"""
买家秀配图生成模块
功能：
  1. 用 qwen3.8-omni-flash 看平铺图，抽出衣服事实
  2. 用 qwen3.7-plus 按槽位写场景句
  3. 用 doubao-seedream-4-5 逐张生图，视觉模型自由评审，不通过就在原提示词后追加约束再生成一次
  4. 将生成的图片上传到飞书，并写入多维表格
"""

import json
import time
import base64
import logging
import os
import random
from pathlib import Path

import requests

from config import settings
from feishu_reader import get_tenant_access_token as get_feishu_token
from lark_cli import BASE_DIR, base_batch_create, base_upload_attachment, write_tmp_bytes
from qwen_client import dashscope_configured, qwen_chat
from buyer_show_prompt import (
    GARMENT_SYSTEM,
    SlotMemory,
    build_borrow_pose_prompt,
    build_product_only_prompt,
    build_reference_scene_prompt,
    build_same_scene_prompt,
    build_scene_messages,
    contains_banned_scene_word,
    garment_from_title,
    normalize_garment,
    parse_json_object,
    sample_slot,
    scene_sentence_from_response,
)
from buyer_show_review import (
    ImageReview,
    build_retry_prompt,
    pick_attempt,
    prompt_diff,
    retry_constraints,
    review_buyer_show,
)

logger = logging.getLogger("image-generator")

PORTRAIT_SIZE = "1728x2304"
SQUARE_SIZE = "2K"
SHOTS_PER_SCENE = 3
SAME_SCENE_VARIATIONS = (
    "换一个机位，拍全身，人站着，手机拿在身前",
    "换一个机位，人对着镜子或玻璃，仍是刚才那个地方",
)


# ========== 2. 调用 doubao 图生图 ==========
def _image_bytes_to_data_uri(image_bytes: bytes, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def _is_size_error(data: dict) -> bool:
    blob = json.dumps(data, ensure_ascii=False).lower()
    return "size" in blob or "dimension" in blob or "resolution" in blob or "像素" in blob


def _post_doubao_payload(payload: dict, *, allow_size_fallback: bool = True) -> str | None:
    if not settings.llm.doubao_api_key:
        raise RuntimeError("未配置 DOUBAO_API_KEY，无法调用豆包生图")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.llm.doubao_api_key}"
    }

    try:
        # 自动重试（应对 429 过载错误）
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(settings.llm.doubao_api_url, headers=headers, json=payload, timeout=120)
            data = response.json()

            # 检查是否 429 过载
            if response.status_code == 429 or data.get("error", {}).get("type") == "engine_overloaded_error":
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s
                    logger.info(f"  ⚠️ doubao API 过载，{wait_time} 秒后自动重试（第 {attempt+1}/{max_retries} 次）...")
                    time.sleep(wait_time)
                    continue

            if "data" in data and len(data["data"]) > 0:
                return data["data"][0].get("url")

            if (
                allow_size_fallback
                and payload.get("size") not in {None, SQUARE_SIZE}
                and response.status_code == 400
                and _is_size_error(data)
            ):
                logger.info("竖图尺寸被拒绝，改用 %s", SQUARE_SIZE)
                return _post_doubao_payload(
                    {**payload, "size": SQUARE_SIZE},
                    allow_size_fallback=False,
                )

            logger.info(f"doubao 生图失败: {json.dumps(data, ensure_ascii=False)}")
            return None

        logger.info("doubao API 多次重试后仍失败")
        return None
    except Exception as e:
        logger.info(f"doubao API 调用异常: {str(e)}")
        return None


def generate_lifestyle_image(prompt: str, image_base64: str) -> str | None:
    """
    调用 doubao-seedream-4-5 API，传入白底图的 base64 数据和场景 Prompt，
    返回生成的生活场景图片 URL。
    """
    payload = {
        "model": "doubao-seedream-4-5-251128",
        "prompt": prompt,
        "image": image_base64,
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": PORTRAIT_SIZE,
        "stream": False,
        "watermark": False
    }

    return _post_doubao_payload(payload)


def generate_lifestyle_image_with_references(
    prompt: str,
    product_image_base64: str,
    outfit_image_base64: str,
) -> tuple[str | None, str]:
    """
    使用商品平铺图 + 单张穿搭参考图进行多图参考生成。
    当前火山方舟接口的单图生图使用 image 字段，这里用同一字段传入两张图数组。
    如果接口不支持双图数组，直接失败，避免未知字段被忽略后变成纯提示词生图。
    """
    base_payload = {
        "model": "doubao-seedream-4-5-251128",
        "prompt": prompt,
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": PORTRAIT_SIZE,
        "stream": False,
        "watermark": False
    }

    reference_images = [product_image_base64, outfit_image_base64]
    field_name = "image"
    payload = {**base_payload, field_name: reference_images}
    logger.info(f"  使用多图字段 {field_name} 进行双图参考生成，输入图片数={len(reference_images)}...")
    url = _post_doubao_payload(payload)
    if url:
        logger.info(f"  ✅ 双图参考生成成功，使用字段: {field_name}")
        return url, field_name

    logger.info("  ⚠️ 双图参考生成失败，未回退到单商品图生图")
    return None, ""


# ========== 飞书图片上传 ==========
def _cli_relpath(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        candidate = BASE_DIR / path
    if not candidate.is_file():
        return None
    try:
        return candidate.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return None


def upload_image_to_feishu(image_source, token: str, filename: str = "image.png") -> str | None:
    """
    将图片落到本地临时文件，返回可供 lark-cli 上传的相对路径。
    image_source 可以是 URL、本地路径或 bytes。
    """
    del token
    try:
        if isinstance(image_source, str):
            if os.path.isfile(image_source):
                with open(image_source, "rb") as f:
                    image_bytes = f.read()
            else:
                img_resp = requests.get(image_source, timeout=30)
                if img_resp.status_code != 200:
                    logger.info(f"下载图片失败: HTTP {img_resp.status_code}")
                    return None
                image_bytes = img_resp.content
        else:
            image_bytes = image_source
        return write_tmp_bytes(image_bytes, filename)
    except Exception as e:
        logger.info(f"准备飞书上传文件失败: {str(e)}")
        return None


def write_to_feishu_table(product_name: str, product_image_token: str | None,
                          ai_image_tokens: list[str], token: str,
                          review_text: str = "") -> bool:
    """
    向飞书结果表写入一行：先创建文本记录，再把本地图片追加到附件字段。
    """
    del token
    app_token = settings.feishu.app_token
    table_id = settings.feishu.table_id
    if not app_token or not table_id:
        logger.info("❌ 飞书结果表未配置")
        return False

    try:
        fields = {"商品名称": product_name}
        if review_text.strip():
            fields["评价内容"] = review_text.strip()
        record_ids = base_batch_create(app_token, table_id, [fields])
        if not record_ids:
            logger.info("❌ 飞书表格写入失败: 未返回 record_id")
            return False
        record_id = record_ids[0]
        product_files = [_cli_relpath(product_image_token)]
        product_files = [p for p in product_files if p]
        if product_files:
            base_upload_attachment(app_token, table_id, record_id, "商品主图", product_files)
        ai_files = [_cli_relpath(path) for path in ai_image_tokens]
        ai_files = [p for p in ai_files if p]
        if ai_files:
            base_upload_attachment(app_token, table_id, record_id, "ai生成的买家秀图片", ai_files)
        logger.info("✅ 飞书表格写入成功")
        return True
    except Exception as e:
        logger.info(f"飞书写入异常: {str(e)}")
        return False


def _chosen_image_source(group: dict) -> str:
    attempts = group.get("attempts") or []
    chosen = group.get("chosen")
    if isinstance(chosen, int) and 0 <= chosen < len(attempts):
        return attempts[chosen].get("image_path") or attempts[chosen].get("image_url") or ""
    return ""


def group_images_by_scene(sources: list, attempts: list | None, shots_per_scene: int) -> list[list]:
    """一个场景一组。有场景标记时按场景分，否则每 shots_per_scene 张切一组。"""
    if attempts:
        buckets: dict[int, list] = {}
        for group in attempts:
            scene = group.get("scene_index")
            if not isinstance(scene, int):
                continue
            source = _chosen_image_source(group)
            if not source:
                continue
            buckets.setdefault(scene, []).append(source)
        if buckets:
            return [buckets[key] for key in sorted(buckets)]
    size = shots_per_scene if shots_per_scene > 1 else 0
    if size <= 1:
        return [list(sources)] if sources else []
    return [list(sources[i:i + size]) for i in range(0, len(sources), size) if sources[i:i + size]]


def commit_images_to_feishu(
    product_name: str,
    product_image_source,
    generated_image_sources: list,
    review_text: str = "",
    attempts: list | None = None,
    shots_per_scene: int = SHOTS_PER_SCENE,
) -> dict:
    """一个场景写飞书一行：这一行只有该场景的图，并带上对应的一条评价。"""
    from utils import split_review_blocks

    result = {
        "status": "error",
        "message": "",
        "product_image_token": None,
        "ai_image_tokens": [],
        "row_count": 0,
    }

    try:
        feishu_token = get_feishu_token()
    except RuntimeError as e:
        result["message"] = f"❌ 获取飞书权限失败：{e}"
        return result

    logger.info("📤 审核通过，正在上传商品图到飞书...")
    product_image_token = upload_image_to_feishu(product_image_source, feishu_token, "product_main.png")
    if not product_image_token:
        result["message"] = "❌ 商品图上传飞书失败"
        return result

    scene_groups = group_images_by_scene(generated_image_sources, attempts, shots_per_scene)
    reviews = split_review_blocks(review_text)
    row_count = max(len(scene_groups), len(reviews))
    if row_count == 0:
        result["message"] = "❌ 没有可写入飞书的图片或评价"
        return result

    written = 0
    ai_image_tokens: list[str] = []
    for row_index in range(row_count):
        sources = scene_groups[row_index] if row_index < len(scene_groups) else []
        review = reviews[row_index] if row_index < len(reviews) else ""
        row_tokens = []
        for image_index, image_source in enumerate(sources, start=1):
            token = upload_image_to_feishu(
                image_source,
                feishu_token,
                f"scene{row_index + 1}_{image_index}.png",
            )
            if token:
                row_tokens.append(token)
                ai_image_tokens.append(token)
        if sources and not row_tokens:
            result["message"] = f"❌ 场景 {row_index + 1} 的图片上传飞书失败"
            return result
        success = write_to_feishu_table(
            product_name,
            product_image_token,
            row_tokens,
            feishu_token,
            review_text=review,
        )
        if not success:
            result["status"] = "partial"
            result["message"] = f"⚠️ 场景 {row_index + 1} 写入飞书表格失败"
            result["ai_image_tokens"] = ai_image_tokens
            result["row_count"] = written
            return result
        written += 1

    result["product_image_token"] = product_image_token
    result["ai_image_tokens"] = ai_image_tokens
    result["row_count"] = written
    result["status"] = "success"
    result["message"] = f"🎉 审核通过，已按场景写入 {written} 行，共 {len(ai_image_tokens)} 张买家秀图"
    return result


# ========== 4. 看图、场景句、逐张生成 ==========
def _guess_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
        return "image/webp"
    return "image/jpeg"


def _image_content(image_bytes: bytes) -> dict:
    mime = _guess_mime(image_bytes)
    data_uri = _image_bytes_to_data_uri(image_bytes, mime)
    return {"type": "image_url", "image_url": {"url": data_uri}}


def extract_garment_facts(image_bytes: bytes, product_name: str) -> dict:
    """看平铺图抽出衣服事实。没有百炼 Key 或调用失败时，退回洗过的标题。"""
    fallback = garment_from_title(product_name)
    if not dashscope_configured():
        logger.warning("未配置 DASHSCOPE_API_KEY，衣服事实退回标题")
        return fallback
    try:
        text = qwen_chat(
            [
                {"role": "system", "content": GARMENT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        _image_content(image_bytes),
                        {"type": "text", "text": f"商品标题：{product_name}"},
                    ],
                },
            ],
            model=settings.llm.vision_model,
            temperature=0.2,
        )
        return normalize_garment(parse_json_object(text), product_name)
    except Exception as exc:
        logger.warning("看图抽取衣服事实失败，退回标题: %s", exc)
        return fallback


def write_scene_sentence(
    garment: dict,
    slot,
    *,
    reject_note: str = "",
    failure_hint_text: str = "",
    rng: random.Random | None = None,
) -> str:
    """用 qwen3.7-plus 把已经抽中的槽位写成一条场景句。"""
    messages = build_scene_messages(
        garment,
        slot,
        reject_note=reject_note,
        failure_hint=failure_hint_text,
        rng=rng,
    )
    if not dashscope_configured():
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法写场景句")
    text = qwen_chat(messages, model=settings.llm.text_model, temperature=0.8)
    sentence = scene_sentence_from_response(text, garment.get("one_line") or "")
    banned = contains_banned_scene_word(sentence)
    if banned:
        logger.warning("场景句含禁用词「%s」，仍使用该句", banned)
    return sentence


def generate_scene_prompts(product_name: str, scene_count: int = 2) -> list[str]:
    """兼容旧调用：没有平铺图时，只按标题槽位写多条场景句。"""
    garment = garment_from_title(product_name)
    memory = SlotMemory()
    rng = random.Random()
    prompts: list[str] = []
    for _ in range(scene_count):
        slot = sample_slot(garment, memory, rng)
        try:
            sentence = write_scene_sentence(garment, slot, rng=rng)
        except Exception as exc:
            logger.info("场景句生成失败: %s", exc)
            return prompts
        prompts.append(build_product_only_prompt(sentence, garment["one_line"]))
    return prompts


def _download_image(url: str) -> bytes | None:
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200 or not response.content:
            logger.info("下载成图失败: HTTP %s", response.status_code)
            return None
        return response.content
    except Exception as exc:
        logger.info("下载成图异常: %s", exc)
        return None


def _save_attempt_bytes(content: bytes, round_no: int) -> str:
    """每次尝试都落成本地文件，豆包链接过期后页面仍能对比。"""
    from utils import save_phone_jpeg

    generated_dir = settings.paths.generated_image_dir
    os.makedirs(generated_dir, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    path = os.path.join(generated_dir, f"ai_r{round_no}_{stamp}.jpg")
    try:
        return save_phone_jpeg(content, path)
    except Exception as exc:
        logger.info("尝试图重压失败，原样保存: %s", exc)
        raw = os.path.join(generated_dir, f"ai_r{round_no}_{stamp}.bin")
        with open(raw, "wb") as handle:
            handle.write(content)
        return raw


def _run_attempt(
    *,
    round_no: int,
    prompt: str,
    previous_prompt: str,
    added_constraints: list[str],
    image_bytes: bytes,
    mode: str,
    product_image_base64: str,
    outfit_bytes: bytes | None,
) -> dict:
    """生成一张、落到本地、交给视觉模型自由评审。"""
    gen_url, reference_field = _generate_one(
        mode=mode,
        prompt=prompt,
        product_image_base64=product_image_base64,
        outfit_bytes=outfit_bytes,
    )
    attempt = {
        "round": round_no,
        "prompt": prompt,
        "added_constraints": added_constraints,
        "diff": prompt_diff(previous_prompt, prompt) if previous_prompt else [],
        "image_url": gen_url or "",
        "image_path": "",
        "reference_field": reference_field,
        "review": ImageReview(skipped=True, error="生图失败").to_dict(),
    }
    if not gen_url:
        return attempt
    content = _download_image(gen_url)
    if content:
        attempt["image_path"] = _save_attempt_bytes(content, round_no)
    attempt["review"] = review_buyer_show(image_bytes, content).to_dict()
    return attempt


def _generate_with_review(
    *,
    index: int,
    first_prompt: str,
    image_bytes: bytes,
    mode: str,
    product_image_base64: str,
    outfit_bytes: bytes | None,
    prior_attempt: dict | None,
    note: str,
) -> dict:
    """一张图最多新生成两次。第二次提示词是第一次原文加上评审约束。"""
    attempts: list[dict] = []
    previous_prompt = ""
    added: list[str] = []
    if prior_attempt and prior_attempt.get("prompt"):
        prior = {**prior_attempt, "from_previous_round": True}
        attempts.append(prior)
        previous_prompt = str(prior.get("prompt") or "")
        added = retry_constraints(ImageReview.from_dict(prior.get("review")), note)
        first_prompt = build_retry_prompt(previous_prompt, added)

    first = _run_attempt(
        round_no=len(attempts) + 1,
        prompt=first_prompt,
        previous_prompt=previous_prompt,
        added_constraints=added,
        image_bytes=image_bytes,
        mode=mode,
        product_image_base64=product_image_base64,
        outfit_bytes=outfit_bytes,
    )
    attempts.append(first)

    first_review = ImageReview.from_dict(first.get("review"))
    if first.get("image_url") and not first_review.passed and first_review.constraints and not first_review.skipped:
        constraints = retry_constraints(first_review)
        logger.info("第 %d 张评审未通过，追加 %d 条约束重做", index + 1, len(constraints))
        attempts.append(_run_attempt(
            round_no=len(attempts) + 1,
            prompt=build_retry_prompt(first_prompt, constraints),
            previous_prompt=first_prompt,
            added_constraints=constraints,
            image_bytes=image_bytes,
            mode=mode,
            product_image_base64=product_image_base64,
            outfit_bytes=outfit_bytes,
        ))

    start = 1 if prior_attempt and prior_attempt.get("prompt") else 0
    picked = pick_attempt(attempts[start:])
    chosen = picked + start if picked >= 0 else -1
    return {"index": index, "chosen": chosen, "attempts": attempts}


def _render_prompt(
    *,
    mode: str,
    garment: dict,
    slot,
    reject_note: str,
    failure_hint_text: str,
    rng: random.Random,
) -> str:
    one_line = garment.get("one_line") or "服装"
    if mode == "use_ref_scene":
        text = build_reference_scene_prompt(one_line, reject_note)
        if failure_hint_text.strip():
            text += f"上一张的问题，这次必须避开：{failure_hint_text.strip()}。"
        return text
    sentence = write_scene_sentence(
        garment,
        slot,
        reject_note=reject_note,
        failure_hint_text=failure_hint_text,
        rng=rng,
    )
    if mode == "borrow_pose":
        return build_borrow_pose_prompt(sentence, one_line)
    return build_product_only_prompt(sentence, one_line)


def _generate_one(
    *,
    mode: str,
    prompt: str,
    product_image_base64: str,
    outfit_bytes: bytes | None,
) -> tuple[str | None, str]:
    if mode == "product_only" or not outfit_bytes:
        return generate_lifestyle_image(prompt, product_image_base64), ""
    outfit_image_base64 = _image_bytes_to_data_uri(outfit_bytes)
    return generate_lifestyle_image_with_references(
        prompt,
        product_image_base64,
        outfit_image_base64,
    )


def _chosen_attempt(group: dict) -> dict | None:
    attempts = group.get("attempts") or []
    chosen = group.get("chosen")
    if not isinstance(chosen, int) or chosen < 0 or chosen >= len(attempts):
        return None
    return attempts[chosen]


def _anchor_ready(group: dict) -> bool:
    chosen = _chosen_attempt(group)
    if not chosen or not os.path.isfile(chosen.get("image_path") or ""):
        return False
    review = ImageReview.from_dict(chosen.get("review"))
    return bool(review.passed and not review.skipped)


def _read_local_bytes(path: str) -> bytes | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return handle.read()


def _keep_group(result: dict, generated_urls: list[str], group: dict) -> None:
    chosen = _chosen_attempt(group)
    if not chosen:
        return
    generated_urls.append(chosen.get("image_url") or "")
    result["attempts"].append(group)
    result["prompts"].append(chosen.get("prompt") or "")
    result["local_image_paths"].append(chosen.get("image_path") or "")
    if chosen.get("reference_field"):
        result["reference_fields"].append(chosen["reference_field"])


def _generate_scene_sets(
    result: dict,
    generated_urls: list[str],
    *,
    indexes: list[int],
    full_run: bool,
    scene_count: int,
    shots_per_scene: int,
    image_bytes: bytes,
    product_image_base64: str,
    garment: dict,
    memory,
    rng: random.Random,
    reject_note: str,
    prior_attempts: list,
) -> None:
    """每个场景先出第 1 张。通过后才用它的本地图生成同组其余张。"""
    index_set = None if full_run else set(indexes)
    one_line = garment.get("one_line") or "服装"
    for scene_i in range(scene_count):
        base = scene_i * shots_per_scene
        wanted = [
            shot for shot in range(shots_per_scene)
            if index_set is None or (base + shot) in index_set
        ]
        if not wanted:
            continue
        sentence = ""
        anchor_bytes = None
        if 0 in wanted:
            slot = sample_slot(garment, memory, rng)
            prior = prior_attempts[base] if base < len(prior_attempts) else None
            if prior and prior.get("scene_sentence"):
                sentence = prior["scene_sentence"]
            else:
                try:
                    sentence = write_scene_sentence(
                        garment, slot, reject_note=reject_note, rng=rng,
                    )
                except Exception as exc:
                    logger.info("场景 %d 的场景句失败: %s", scene_i + 1, exc)
                    continue
            prompt = "" if prior and prior.get("prompt") else build_product_only_prompt(sentence, one_line)
            group = _tag_group(
                _generate_with_review(
                    index=base,
                    first_prompt=prompt,
                    image_bytes=image_bytes,
                    mode="product_only",
                    product_image_base64=product_image_base64,
                    outfit_bytes=None,
                    prior_attempt=prior if prior and prior.get("prompt") else None,
                    note=reject_note,
                ),
                scene_index=scene_i,
                shot_index=0,
                scene_sentence=sentence,
            )
            if group["chosen"] < 0:
                logger.info("场景 %d 的第 1 张生成失败", scene_i + 1)
                continue
            _keep_group(result, generated_urls, group)
            if not _anchor_ready(group):
                result["await_anchor"] = True
                continue
            anchor_bytes = _read_local_bytes(_chosen_attempt(group).get("image_path") or "")
        else:
            prior_anchor = prior_attempts[base] if base < len(prior_attempts) else None
            sentence = (prior_anchor or {}).get("scene_sentence") or ""
            anchor_bytes = _read_local_bytes((prior_anchor or {}).get("image_path") or "")
            if not anchor_bytes:
                logger.info("场景 %d 没有可参考的第 1 张", scene_i + 1)
                continue
        for shot in wanted:
            if shot == 0:
                continue
            variation = SAME_SCENE_VARIATIONS[min(shot - 1, len(SAME_SCENE_VARIATIONS) - 1)]
            slot_index = base + shot
            prior = prior_attempts[slot_index] if slot_index < len(prior_attempts) else None
            prompt = "" if prior and prior.get("prompt") else build_same_scene_prompt(sentence, one_line, variation)
            group = _tag_group(
                _generate_with_review(
                    index=slot_index,
                    first_prompt=prompt,
                    image_bytes=image_bytes,
                    mode="follow_anchor",
                    product_image_base64=product_image_base64,
                    outfit_bytes=anchor_bytes,
                    prior_attempt=prior if prior and prior.get("prompt") else None,
                    note=reject_note,
                ),
                scene_index=scene_i,
                shot_index=shot,
                scene_sentence=sentence,
            )
            if group["chosen"] < 0:
                logger.info("场景 %d 的第 %d 张生成失败", scene_i + 1, shot + 1)
                continue
            _keep_group(result, generated_urls, group)


def _tag_group(group: dict, *, scene_index: int, shot_index: int, scene_sentence: str) -> dict:
    group["scene_index"] = scene_index
    group["shot_index"] = shot_index
    group["scene_sentence"] = scene_sentence
    return group


def run_image_generation(
    image_bytes: bytes,
    product_name: str,
    scene_count: int = 2,
    outfit_image_bytes_list: list[bytes] | None = None,
    commit_to_feishu: bool = True,
    reject_note: str = "",
    only_indexes: list[int] | None = None,
    review_text: str = "",
    prior_attempts: list[dict] | None = None,
    shots_per_scene: int = SHOTS_PER_SCENE,
) -> dict:
    """
    飞书「晒图数量」是场景数。

    没有穿搭参考图时，每个场景固定多张：第 1 张通过后，用它的本地文件当参考，
    再生成同一个人、同一个地方的另外几张。第 1 张没通过就先停住。
    有穿搭参考图时仍按张换衣，不套这组规则。
    """
    outfit_image_bytes_list = outfit_image_bytes_list or []
    prior_attempts = prior_attempts or []
    grouped = not outfit_image_bytes_list and shots_per_scene > 1
    slot_count = scene_count * shots_per_scene if grouped else scene_count
    result = {
        "status": "error",
        "message": "",
        "prompts": [],
        "image_urls": [],
        "local_image_paths": [],
        "attempts": [],
        "reference_mode": "product_plus_outfit" if outfit_image_bytes_list else "product_only",
        "outfit_reference_count": len(outfit_image_bytes_list),
        "reference_fields": [],
        "committed_to_feishu": False,
        "image_indexes": [],
        "expected_count": slot_count,
        "shots_per_scene": shots_per_scene if grouped else 1,
        "scene_count": scene_count if grouped else 0,
        "await_anchor": False,
    }

    if only_indexes is None:
        indexes = list(range(slot_count))
    else:
        indexes = []
        for raw in only_indexes:
            idx = int(raw)
            if idx < 0 or idx >= slot_count:
                raise ValueError(f"图片序号超出范围: {idx}")
            if idx not in indexes:
                indexes.append(idx)
        if not indexes:
            result["message"] = "没有要重做的图片"
            return result
    result["image_indexes"] = []

    garment = extract_garment_facts(image_bytes, product_name)
    product_image_base64 = _image_bytes_to_data_uri(image_bytes)
    memory = SlotMemory()
    rng = random.Random()
    generated_urls: list[str] = []

    for index in ([] if grouped else indexes):
        if outfit_image_bytes_list and index < len(outfit_image_bytes_list):
            mode = "use_ref_scene"
            outfit_bytes = outfit_image_bytes_list[index]
            slot = None
        elif outfit_image_bytes_list:
            mode = "borrow_pose"
            outfit_bytes = outfit_image_bytes_list[index % len(outfit_image_bytes_list)]
            slot = sample_slot(garment, memory, rng)
        else:
            mode = "product_only"
            outfit_bytes = None
            slot = sample_slot(garment, memory, rng)

        prior = prior_attempts[index] if index < len(prior_attempts) else None
        prompt = ""
        if not (prior and prior.get("prompt")):
            try:
                prompt = _render_prompt(
                    mode=mode,
                    garment=garment,
                    slot=slot,
                    reject_note=reject_note,
                    failure_hint_text="",
                    rng=rng,
                )
            except Exception as exc:
                logger.info("第 %d 张场景句失败: %s", index + 1, exc)
                continue

        logger.info("正在生成第 %d/%d 张，模式=%s", index + 1, scene_count, mode)
        group = _generate_with_review(
            index=index,
            first_prompt=prompt,
            image_bytes=image_bytes,
            mode=mode,
            product_image_base64=product_image_base64,
            outfit_bytes=outfit_bytes,
            prior_attempt=prior if prior and prior.get("prompt") else None,
            note=reject_note,
        )
        if group["chosen"] < 0:
            logger.info("第 %d 张生成失败，跳过", index + 1)
            continue
        _keep_group(result, generated_urls, group)

    if grouped:
        _generate_scene_sets(
            result,
            generated_urls,
            indexes=indexes,
            full_run=only_indexes is None,
            scene_count=scene_count,
            shots_per_scene=shots_per_scene,
            image_bytes=image_bytes,
            product_image_base64=product_image_base64,
            garment=garment,
            memory=memory,
            rng=rng,
            reject_note=reject_note,
            prior_attempts=prior_attempts,
        )

    result["image_urls"] = generated_urls
    result["image_indexes"] = [group.get("index") for group in result["attempts"]]
    short = len(generated_urls) < (slot_count if only_indexes is None else len(indexes))
    if result.get("await_anchor"):
        result["message"] = "第 1 张未通过评审，同组其余图片先不生成"

    if not generated_urls:
        result["message"] = "❌ 所有场景图生成均失败，请检查 DOUBAO_API_KEY 是否正确"
        return result

    if not commit_to_feishu:
        result["status"] = "partial" if short else "success"
        if not result.get("message"):
            result["message"] = (
                f"生成了 {len(generated_urls)}/{slot_count} 张，缺的张留给人工"
                if short
                else f"✅ 成功生成 {len(generated_urls)} 张待审核买家秀图，尚未写入飞书"
            )
        return result

    if short or result.get("await_anchor"):
        result["status"] = "partial"
        if not result.get("message"):
            result["message"] = f"生成了 {len(generated_urls)}/{slot_count} 张，缺的张留给人工"
        return result

    commit_result = commit_images_to_feishu(
        product_name,
        image_bytes,
        result.get("local_image_paths") or generated_urls,
        review_text=review_text,
        attempts=result.get("attempts") or [],
        shots_per_scene=int(result.get("shots_per_scene") or 1),
    )

    if commit_result.get("status") == "success":
        result["status"] = "partial" if short else "success"
        result["committed_to_feishu"] = True
        result["message"] = f"🎉 成功生成 {len(generated_urls)} 张买家秀配图，已写入飞书表格！"
    else:
        result["status"] = commit_result.get("status", "partial")
        result["message"] = commit_result.get("message") or f"⚠️ 生成了 {len(generated_urls)} 张配图，但写入飞书失败"

    return result

