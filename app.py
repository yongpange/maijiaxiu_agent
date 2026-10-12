"""
V3.0 买家秀生成 Agent - Web 服务入口

能力：
1. 文本/图片生成
2. 批量生成：从飞书需求表读取，逐行生成评价+晒图并写回飞书
3. 图片人工审核（可选）
"""
from __future__ import annotations
import sys
import io
import os
import json
import asyncio
import logging
import threading

if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from langgraph.errors import GraphInterrupt
except Exception:
    try:
        from langgraph.types import GraphInterrupt
    except Exception:
        class GraphInterrupt(Exception):
            """LangGraph interrupt signal -- raise when graph hits interrupt()."""
            pass

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger("maijiaxiu-api")

from outbox import (
    ensure_outbox_db,
    create_delivery_approval as db_create_approval,
    get_delivery_approval as db_get_approval,
    list_delivery_approvals as db_list_approvals,
    confirm_delivery_approval as db_confirm_approval,
    reject_delivery_approval as db_reject_approval,
)
from utils import (
    materialize_generated_images,
    merge_replaced_images,
    parse_target_contacts,
)
import uvicorn

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ===== FastAPI 应用 =====
app = FastAPI(
    title="买家秀生成 Agent V3.0",
    version="3.0",
    description="从飞书需求表批量读取商品信息，生成评价和晒图，写回飞书。"
)

# 挂载静态资源
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
from config import settings  # pylint: disable=wrong-import-position

OUTBOX_DB_PATH = settings.outbox.db_path
GENERATED_IMAGE_DIR = settings.paths.generated_image_dir
DEFAULT_TARGET_CONTACTS = settings.outbox.default_target_contacts
REQUIRE_DELIVERY_CONFIRMATION = settings.outbox.require_confirmation


@app.on_event("startup")
async def on_startup():
    ensure_outbox_db(OUTBOX_DB_PATH)
    os.makedirs(GENERATED_IMAGE_DIR, exist_ok=True)


def _get_agent_v2():
    """延迟导入 LangGraph V2 Agent 接口。"""
    from agent_graph_v2 import (  # pylint: disable=import-outside-toplevel
        run_agent_v2,
        resume_agent_v2,
        get_agent_state,
        run_agent,
    )
    return run_agent_v2, resume_agent_v2, get_agent_state, run_agent


def _get_batch_graph():
    """延迟导入批量子图接口。"""
    from batch_graph import (  # pylint: disable=import-outside-toplevel
        run_batch_generate,
        get_batch_state,
    )
    return run_batch_generate, get_batch_state


class DeliveryApprovalActionRequest(BaseModel):
    note: str = Field(default="", description="确认/驳回备注")
    image_indexes: List[int] = Field(default_factory=list, description="要重做的图片下标，从 0 开始")
    scene_index: Optional[int] = Field(default=None, description="只补这一个场景的同组图片")


# ===== 根路径：返回聊天页面 =====
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/generated-images/{filename}")
async def serve_generated_image(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(GENERATED_IMAGE_DIR, safe_name)
    if not os.path.isfile(file_path):
        return JSONResponse(content={"error": "not found"}, status_code=404)
    return FileResponse(file_path, media_type="image/jpeg")


# ===== 纯文字聊天 API =====
@app.post("/chat")
async def chat(request: Request):
    """
    纯文字请求，交给 LangGraph Agent 生成评价。
    """
    try:
        payload = await request.json()
        user_input = payload.get("user_input", "")

        if not user_input.strip():
            return JSONResponse(content={"reply": "请输入商品标题。"})

        _, _, _, run_agent = _get_agent_v2()
        result = await asyncio.to_thread(run_agent, user_input)
        return JSONResponse(content={"reply": result})

    except Exception as e:
        return JSONResponse(
            content={"reply": f"⚠️ 处理出错：{str(e)}\n请稍后重试，可能是 API 暂时过载。"},
            status_code=200
        )


# ===== 文字 + 图片的合并请求 =====
@app.post("/chat-with-image")
async def chat_with_image(
    file: UploadFile = File(...),
    user_input: str = Form(default=""),
    target_contacts: str = Form(default=""),
    target_contact: str = Form(default=""),
):
    """
    接收用户输入的文字 + 商品白底图。
    流程：调用 LangGraph V2 Agent 一次性完成评价生成、配图生成、[人工确认]。
    """
    run_agent_v2, _, get_agent_state, _ = _get_agent_v2()
    image_bytes = await file.read()

    raw_contacts = target_contacts or target_contact or DEFAULT_TARGET_CONTACTS
    contacts = parse_target_contacts(raw_contacts)

    thread_id = str(uuid4())
    require_confirmation = REQUIRE_DELIVERY_CONFIRMATION

    try:
        logger.info("启动 Agent V2，thread_id=%s, require_confirmation=%s", thread_id, require_confirmation)
        result = await asyncio.to_thread(
            run_agent_v2,
            user_input=user_input,
            thread_id=thread_id,
            product_image_bytes=image_bytes,
            outfit_image_bytes_list=[],
            target_contacts=contacts,
            review_count=5,
            image_count=2,
            require_confirmation=require_confirmation,
        )

        review_text = result.get("reviews_formatted", "")
        image_result = result.get("image_result")
        local_images = result.get("local_image_paths", [])

        return JSONResponse(content={
            "reply": review_text,
            "image_result": image_result,
            "require_confirmation": False,
            "approval_result": None,
            "materialized_images": local_images,
            "thread_id": thread_id,
        })

    except GraphInterrupt as e:
        logger.info("Agent V2 中断等待人工确认，thread_id=%s", thread_id)
        state = get_agent_state(thread_id)
        review_text = state.get("reviews_formatted", "") if state else ""
        local_images = state.get("local_image_paths", []) if state else []
        product_name = state.get("product_name", "未命名商品") if state else "未命名商品"

        approval_result = _create_delivery_approval(
            source="chat-with-image",
            product_title=product_name,
            target_contacts=contacts,
            review_text=review_text,
            image_paths=local_images,
            file_paths=[],
            extra={"thread_id": thread_id, "interrupt_payload": getattr(e, "value", None)},
        )

        return JSONResponse(content={
            "reply": review_text,
            "image_result": state.get("image_result") if state else None,
            "require_confirmation": True,
            "approval_result": approval_result,
            "materialized_images": local_images,
            "thread_id": thread_id,
        })

    except Exception as e:
        logger.exception("chat-with-image 异常")
        return JSONResponse(
            content={
                "reply": f"⚠️ 处理出错：{str(e)}\n请稍后重试。",
                "image_result": None,
                "thread_id": thread_id,
            },
            status_code=200
        )


def _create_delivery_approval(
    *,
    source: str,
    product_title: str,
    target_contacts: List[str],
    review_text: str,
    image_paths: List[str],
    file_paths: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    approval = db_create_approval(
        OUTBOX_DB_PATH,
        source=source,
        product_title=product_title,
        target_contacts=target_contacts,
        review_text=review_text,
        image_paths=image_paths,
        file_paths=file_paths,
        extra=extra,
    )
    return approval.to_dict()


def _is_image_review_approval(approval) -> bool:
    return (approval.source == "batch-image-review") or (
        (approval.extra or {}).get("approval_type") == "image_review"
    )


def _read_local_file_bytes(path: str) -> bytes:
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"本地候选文件不存在: {path}")
    with open(path, "rb") as f:
        return f.read()


def _chosen_of(group: dict) -> dict:
    attempts = group.get("attempts") or []
    chosen = group.get("chosen")
    if isinstance(chosen, int) and 0 <= chosen < len(attempts):
        return attempts[chosen]
    return {}


def _review_passed(group: dict) -> bool:
    review = _chosen_of(group).get("review") or {}
    return bool(review.get("pass") or review.get("passed")) and not review.get("skipped")


def _scene_index_of(group: dict, shots: int) -> int:
    if isinstance(group.get("scene_index"), int):
        return group["scene_index"]
    if isinstance(group.get("index"), int) and shots > 0:
        return int(group["index"]) // shots
    return 0


def _shot_index_of(group: dict, shots: int) -> int:
    if isinstance(group.get("shot_index"), int):
        return group["shot_index"]
    if isinstance(group.get("index"), int) and shots > 0:
        return int(group["index"]) % shots
    return 0


def _confirm_anchor_and_fill(existing, note: str, scene_index: int):
    """只补被点到的那一个场景。其他场景的图保持不动。"""
    from image_generator import run_image_generation

    extra = existing.extra or {}
    shots = int(extra.get("shots_per_scene") or 1)
    scene_count = int(extra.get("scene_count") or len(existing.image_paths) or 1)
    if scene_index < 0 or scene_index >= scene_count:
        raise RuntimeError(f"场景序号超出范围: {scene_index}")
    product_image_bytes = _read_local_file_bytes(extra.get("product_image_path"))
    priors = [None] * (scene_count * shots)
    for group in extra.get("image_attempts") or []:
        scene_i = _scene_index_of(group, shots)
        if scene_i != scene_index or _shot_index_of(group, shots) != 0:
            continue
        chosen = dict(_chosen_of(group))
        if not chosen.get("image_path"):
            continue
        chosen["scene_sentence"] = group.get("scene_sentence") or ""
        priors[scene_i * shots] = chosen
    if not priors[scene_index * shots]:
        raise RuntimeError(f"场景 {scene_index + 1} 没有可参考的第 1 张")
    follow_indexes = [scene_index * shots + shot for shot in range(1, shots)]
    image_result = run_image_generation(
        image_bytes=product_image_bytes,
        product_name=existing.product_title,
        scene_count=scene_count,
        shots_per_scene=shots,
        commit_to_feishu=False,
        only_indexes=follow_indexes,
        prior_attempts=priors,
        reject_note=note or "",
    )
    if image_result.get("status") not in {"success", "partial"} or not image_result.get("image_urls"):
        raise RuntimeError(image_result.get("message") or "同组图片生成失败")
    groups = _merge_attempt_groups(extra.get("image_attempts") or [], image_result.get("attempts") or [])
    paths = []
    for group in sorted(groups, key=lambda item: (_scene_index_of(item, shots), _shot_index_of(item, shots))):
        path = _chosen_of(group).get("image_path") or ""
        if path:
            paths.append(path)
    filled = {
        scene_i
        for scene_i in range(scene_count)
        if sum(1 for group in groups if _scene_index_of(group, shots) == scene_i) >= shots
    }
    still_waiting = len(filled) < scene_count
    this_follows = [
        group for group in groups
        if _scene_index_of(group, shots) == scene_index and _shot_index_of(group, shots) > 0
    ]
    complete = (not still_waiting) and all(_review_passed(group) for group in this_follows) and len(paths) >= scene_count * shots
    if not complete:
        db_reject_approval(OUTBOX_DB_PATH, existing.approval_id, note=note or "第1张已通过，等待同组其余图片")
        new_approval = db_create_approval(
            OUTBOX_DB_PATH,
            source=existing.source,
            product_title=existing.product_title,
            target_contacts=existing.target_contacts,
            review_text=existing.review_text,
            image_paths=paths,
            file_paths=existing.file_paths,
            max_retry=existing.max_retry,
            extra={
                **extra,
                "phase": "await_anchor" if still_waiting else "review_set",
                "image_count": scene_count * shots,
                "image_attempts": groups,
                "previous_approval_id": existing.approval_id,
            },
        )
        return new_approval, {"status": "expanded", "message": "已补上同组图片，请再看一遍"}

    existing.image_paths = paths
    extra = {**extra, "phase": "review_set", "image_attempts": groups}
    existing.extra = extra
    return _commit_confirmed_images(existing, note)


def _confirm_image_review_approval(existing, note: str, scene_index: Optional[int] = None):
    """整组都在才写飞书。补图必须带场景序号，只生成那一组。"""
    extra = existing.extra or {}
    if extra.get("phase") == "await_anchor" and int(extra.get("shots_per_scene") or 1) > 1:
        if scene_index is None:
            raise RuntimeError("请选择要补图的场景")
        return _confirm_anchor_and_fill(existing, note, scene_index)
    return _commit_confirmed_images(existing, note)


def _commit_confirmed_images(existing, note: str):
    """把已经看过的整组图片写入飞书结果表。"""
    from image_generator import commit_images_to_feishu
    from feishu_reader import get_tenant_access_token, update_source_row_status

    extra = existing.extra or {}
    product_image_path = extra.get("product_image_path")
    if not product_image_path:
        raise RuntimeError("approval 缺少 product_image_path，无法写回飞书")
    if not existing.image_paths:
        raise RuntimeError("approval 缺少候选图片，无法写回飞书")

    commit_result = commit_images_to_feishu(
        existing.product_title,
        product_image_path,
        existing.image_paths,
        review_text=existing.review_text or "",
        attempts=extra.get("image_attempts") or [],
        shots_per_scene=int(extra.get("shots_per_scene") or 3),
    )
    if commit_result.get("status") != "success":
        raise RuntimeError(commit_result.get("message") or "写入飞书失败")

    source_record_id = extra.get("source_record_id")
    if source_record_id:
        try:
            token = get_tenant_access_token()
            update_source_row_status(token, source_record_id, "已处理")
        except Exception as e:
            logger.warning("更新源需求表状态失败，继续确认审核: %s", e)
            commit_result["source_status_warning"] = str(e)

    updated = db_confirm_approval(
        OUTBOX_DB_PATH,
        existing.approval_id,
        note=note,
    )
    if not updated:
        raise RuntimeError("更新审核记录失败")
    return updated, commit_result


def _merge_attempt_groups(previous: list, updated: list) -> list:
    """按图片序号合并尝试记录，没重做的张保留上一轮。"""
    merged = {}
    for group in previous or []:
        if isinstance(group, dict) and isinstance(group.get("index"), int):
            merged[group["index"]] = group
    for group in updated or []:
        if isinstance(group, dict) and isinstance(group.get("index"), int):
            merged[group["index"]] = group
    return [merged[index] for index in sorted(merged)]


def _regenerate_image_review_approval(existing, note: str, image_indexes: List[int]):
    """图片审核驳回：只重做选中的那几张，其余原图保留。"""
    from image_generator import run_image_generation

    extra = existing.extra or {}
    product_image_path = extra.get("product_image_path")
    outfit_image_paths = extra.get("outfit_image_paths") or []
    shots = int(extra.get("shots_per_scene") or 1)
    scene_count = int(extra.get("scene_count") or extra.get("image_count") or len(existing.image_paths) or 1)
    image_count = int(extra.get("image_count") or len(existing.image_paths) or 1)
    bound = len(existing.image_paths) if extra.get("phase") == "await_anchor" else image_count

    indexes: List[int] = []
    for raw in image_indexes:
        idx = int(raw)
        if idx < 0 or idx >= bound:
            raise RuntimeError(f"图片序号超出范围: {idx}")
        if idx not in indexes:
            indexes.append(idx)
    if extra.get("phase") != "await_anchor" and shots > 1 and not outfit_image_paths:
        expanded = set(indexes)
        for idx in list(indexes):
            if idx % shots == 0:
                scene_i = idx // shots
                expanded.update(range(scene_i * shots, (scene_i + 1) * shots))
        indexes = sorted(expanded)
    if not indexes:
        raise RuntimeError("请选择要重做的图片")

    product_image_bytes = _read_local_file_bytes(product_image_path)
    outfit_image_bytes_list = [
        _read_local_file_bytes(path)
        for path in outfit_image_paths
        if path and os.path.isfile(path)
    ]

    priors = [None] * image_count
    for group in extra.get("image_attempts") or []:
        group_index = group.get("index")
        attempts = group.get("attempts") or []
        chosen = group.get("chosen")
        if (
            isinstance(group_index, int)
            and 0 <= group_index < image_count
            and isinstance(chosen, int)
            and 0 <= chosen < len(attempts)
            and attempts[chosen].get("prompt")
        ):
            prior = dict(attempts[chosen])
            prior["scene_sentence"] = group.get("scene_sentence") or prior.get("scene_sentence") or ""
            priors[group_index] = prior

    image_result = run_image_generation(
        image_bytes=product_image_bytes,
        product_name=existing.product_title,
        scene_count=scene_count if shots > 1 and not outfit_image_bytes_list else image_count,
        shots_per_scene=1 if outfit_image_bytes_list else shots,
        outfit_image_bytes_list=outfit_image_bytes_list or None,
        commit_to_feishu=False,
        reject_note=note or "",
        only_indexes=indexes,
        prior_attempts=priors,
    )
    if image_result.get("status") not in {"success", "partial"} or not image_result.get("image_urls"):
        raise RuntimeError(image_result.get("message") or "重新生成候选图片失败")

    replaced_paths = materialize_generated_images(image_result)
    produced = [int(idx) for idx in (image_result.get("image_indexes") or indexes)]
    if len(replaced_paths) != len(produced):
        raise RuntimeError("重新生成的候选图片数量和选中数量不一致")
    if produced == indexes:
        local_image_paths = merge_replaced_images(existing.image_paths, indexes, replaced_paths)
    else:
        merged = list(existing.image_paths)
        width = max([len(merged) - 1, image_count - 1, *produced]) + 1
        merged.extend([""] * (width - len(merged)))
        for idx, path in zip(produced, replaced_paths):
            merged[idx] = path
        if image_result.get("await_anchor") and shots > 1:
            for idx in produced:
                if idx % shots == 0:
                    for shot in range(1, shots):
                        slot = (idx // shots) * shots + shot
                        if slot < len(merged):
                            merged[slot] = ""
        local_image_paths = [path for path in merged if path]

    db_reject_approval(OUTBOX_DB_PATH, existing.approval_id, note=note or "图片驳回，已重新生成")

    new_extra = {
        **extra,
        "previous_approval_id": existing.approval_id,
        "regenerate_count": int(extra.get("regenerate_count") or 0) + 1,
        "image_reference_mode": image_result.get("reference_mode"),
        "image_reference_fields": image_result.get("reference_fields", []),
        "reject_note": note,
        "replaced_indexes": indexes,
        "image_attempts": _merge_attempt_groups(extra.get("image_attempts") or [], image_result.get("attempts") or []),
        "phase": "await_anchor" if image_result.get("await_anchor") else extra.get("phase") or "review_set",
    }
    new_approval = db_create_approval(
        OUTBOX_DB_PATH,
        source=existing.source,
        product_title=existing.product_title,
        target_contacts=existing.target_contacts,
        review_text=existing.review_text,
        image_paths=local_image_paths,
        file_paths=existing.file_paths,
        max_retry=existing.max_retry,
        extra=new_extra,
    )
    return new_approval, image_result


@app.get("/delivery-approvals")
async def list_delivery_approvals(
    status: str = Query(default="pending"),
    limit: int = Query(default=200, ge=1, le=1000),
):
    try:
        normalized_status = (status or "").strip().lower() or "pending"
        items = db_list_approvals(OUTBOX_DB_PATH, status=normalized_status, limit=limit)
        return JSONResponse(
            content={
                "require_confirmation": REQUIRE_DELIVERY_CONFIRMATION,
                "count": len(items),
                "approvals": [a.to_dict() for a in items],
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/delivery-approvals/{approval_id}/confirm")
async def confirm_delivery_approval(
    approval_id: str,
    payload: Optional[DeliveryApprovalActionRequest] = None,
):
    try:
        existing = db_get_approval(OUTBOX_DB_PATH, approval_id)
        if not existing:
            return JSONResponse(content={"error": "approval_not_found"}, status_code=404)
        if existing.status == "rejected":
            return JSONResponse(content={"error": "approval_already_rejected"}, status_code=400)
        if existing.status == "confirmed":
            return JSONResponse(content={"approval": existing.to_dict(), "message": "already_confirmed"})

        note = payload.note if payload else ""
        scene_index = payload.scene_index if payload else None
        if _is_image_review_approval(existing):
            updated, commit_result = await asyncio.to_thread(
                _confirm_image_review_approval,
                existing,
                note,
                scene_index,
            )
            return JSONResponse(content={
                "approval": updated.to_dict(),
                "commit_result": commit_result,
                "message": "confirmed",
            })

        thread_id = existing.extra.get("thread_id") if existing.extra else None
        if thread_id:
            _, resume_agent_v2, _, _ = _get_agent_v2()
            await asyncio.to_thread(resume_agent_v2, thread_id, "confirmed", note)

        updated = db_confirm_approval(
            OUTBOX_DB_PATH,
            approval_id,
            note=note,
        )
        if not updated:
            return JSONResponse(content={"error": "confirm_failed"}, status_code=500)

        return JSONResponse(content={
            "approval": updated.to_dict(),
            "message": "confirmed",
        })
    except Exception as e:
        logger.exception("确认审核失败")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/delivery-approvals/{approval_id}/reject")
async def reject_delivery_approval(
    approval_id: str,
    payload: Optional[DeliveryApprovalActionRequest] = None,
):
    try:
        existing = db_get_approval(OUTBOX_DB_PATH, approval_id)
        if not existing:
            return JSONResponse(content={"error": "approval_not_found"}, status_code=404)
        if existing.status == "confirmed":
            return JSONResponse(content={"error": "approval_already_confirmed"}, status_code=400)

        note = payload.note if payload else ""
        image_indexes = list(payload.image_indexes) if payload else []
        if _is_image_review_approval(existing):
            if not image_indexes:
                return JSONResponse(content={"error": "请选择要重做的图片"}, status_code=400)
            new_approval, image_result = await asyncio.to_thread(
                _regenerate_image_review_approval,
                existing,
                note,
                image_indexes,
            )
            return JSONResponse(content={
                "approval": new_approval.to_dict(),
                "image_result": image_result,
                "message": "regenerated",
            })

        thread_id = existing.extra.get("thread_id") if existing.extra else None

        # 如果存在 LangGraph thread_id，恢复图执行并驳回
        if thread_id:
            _, resume_agent_v2, _, _ = _get_agent_v2()
            await asyncio.to_thread(resume_agent_v2, thread_id, "rejected", note)

        updated = db_reject_approval(OUTBOX_DB_PATH, approval_id, note=note)
        if not updated:
            return JSONResponse(content={"error": "reject_failed"}, status_code=500)

        return JSONResponse(content={"approval": updated.to_dict(), "message": "rejected"})
    except Exception as e:
        logger.exception("驳回审核失败")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/agent/state/{thread_id}")
async def get_agent_thread_state(thread_id: str):
    """查询 LangGraph Agent 某条 thread 的当前执行状态。"""
    try:
        _, _, get_agent_state_fn, _ = _get_agent_v2()
        state = get_agent_state_fn(thread_id)
        if state is None:
            return JSONResponse(content={"error": "thread_not_found"}, status_code=404)
        return JSONResponse(content={"thread_id": thread_id, "state": state})
    except Exception as e:
        logger.exception("查询 Agent 状态失败")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ===== 批量生成 V2（LangGraph Send 并行） =====

def _get_feishu_reader():
    from feishu_reader import (  # pylint: disable=import-outside-toplevel
        fetch_all_source_rows,
        download_attachment,
        update_source_row_status,
    )
    return fetch_all_source_rows, download_attachment, update_source_row_status


_batch_jobs_lock = threading.Lock()
_batch_jobs: Dict[str, Dict[str, Any]] = {}
_ACTIVE_BATCH_PATH = os.path.join(BASE_DIR, "data", "active_batch.json")


def _read_active_batch() -> Optional[Dict[str, Any]]:
    try:
        with open(_ACTIVE_BATCH_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) and data.get("batch_id") else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_active_batch(batch_id: str, status: str) -> None:
    os.makedirs(os.path.dirname(_ACTIVE_BATCH_PATH), exist_ok=True)
    with open(_ACTIVE_BATCH_PATH, "w", encoding="utf-8") as handle:
        json.dump({"batch_id": batch_id, "status": status}, handle, ensure_ascii=False)


def _running_batch_id() -> Optional[str]:
    with _batch_jobs_lock:
        for batch_id, job in _batch_jobs.items():
            if job.get("status") == "running":
                return batch_id
    return None


def _batch_status_payload(batch_id: str) -> Optional[Dict[str, Any]]:
    with _batch_jobs_lock:
        job = dict(_batch_jobs.get(batch_id) or {})
    active = _read_active_batch()
    known = bool(job) or (active and active.get("batch_id") == batch_id)
    _, get_batch_state_fn = _get_batch_graph()
    try:
        state = get_batch_state_fn(batch_id) or {}
    except Exception as exc:
        logger.warning("读取批次进度失败: %s", exc)
        state = {}
    if not known and not state:
        return None

    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    status = job.get("status")
    if not status:
        if active and active.get("batch_id") == batch_id and active.get("status") == "running":
            status = "error"
            job["error"] = "服务已重启，这次生成中断了"
        elif state.get("final_message") or state.get("error_message"):
            status = "error" if state.get("error_message") else "done"
        else:
            status = "running"
    source_rows = result.get("source_rows") or state.get("source_rows") or []
    row_results = result.get("row_results") or state.get("row_results") or []
    return {
        "batch_id": batch_id,
        "status": status,
        "require_confirmation": REQUIRE_DELIVERY_CONFIRMATION,
        "message": result.get("final_message") or state.get("final_message") or "",
        "total": len(source_rows),
        "success_count": result.get("success_count", state.get("success_count", 0)) or 0,
        "error_count": result.get("error_count", state.get("error_count", 0)) or 0,
        "results": row_results,
        "error": result.get("error_message") or state.get("error_message") or job.get("error"),
    }


def _run_batch_job(batch_id: str, default_contacts: List[str]) -> None:
    run_batch_generate, _ = _get_batch_graph()
    try:
        result = run_batch_generate(
            batch_id=batch_id,
            default_contacts=default_contacts,
            require_confirmation=REQUIRE_DELIVERY_CONFIRMATION,
        )
        status = "error" if result.get("error_message") else "done"
        with _batch_jobs_lock:
            _batch_jobs[batch_id] = {"status": status, "result": result}
        _write_active_batch(batch_id, status)
    except Exception as exc:
        logger.exception("[batch-v2-%s] 批量生成异常", batch_id)
        with _batch_jobs_lock:
            _batch_jobs[batch_id] = {"status": "error", "error": str(exc)}
        _write_active_batch(batch_id, "error")


@app.post("/batch-generate-v2")
async def batch_generate_v2():
    """启动批量生成后立即返回批次号。页面刷新后用这个号把进度接回来。"""
    existing = _running_batch_id()
    if existing:
        return JSONResponse(content={
            "batch_id": existing,
            "status": "running",
            "already_running": True,
            "require_confirmation": REQUIRE_DELIVERY_CONFIRMATION,
        })

    batch_id = str(uuid4())[:8]
    default_contacts = parse_target_contacts(DEFAULT_TARGET_CONTACTS)
    with _batch_jobs_lock:
        _batch_jobs[batch_id] = {"status": "running"}
    _write_active_batch(batch_id, "running")
    logger.info("[batch-v2-%s] 启动批量生成", batch_id)
    threading.Thread(
        target=_run_batch_job,
        args=(batch_id, default_contacts),
        name=f"batch-{batch_id}",
        daemon=True,
    ).start()
    return JSONResponse(content={
        "batch_id": batch_id,
        "status": "running",
        "require_confirmation": REQUIRE_DELIVERY_CONFIRMATION,
    })


@app.get("/batch-generate-v2/current")
async def batch_generate_v2_current():
    """返回当前服务进程里最近一次批量任务，供页面刷新后接上。"""
    active = _read_active_batch()
    if not active:
        return JSONResponse(content={"batch_id": None, "status": "idle"})
    payload = _batch_status_payload(active["batch_id"])
    if payload is None:
        return JSONResponse(content={"batch_id": None, "status": "idle"})
    return JSONResponse(content=payload)


@app.get("/batch-generate-v2/status")
async def batch_generate_v2_status(batch_id: str = Query(default="")):
    """查询批量任务进度。生成过程中也可以读到已经完成的行。"""
    if not batch_id:
        return JSONResponse(content={"error": "batch_id required"}, status_code=400)
    try:
        payload = _batch_status_payload(batch_id)
        if payload is None:
            return JSONResponse(content={"error": "batch_id not found"}, status_code=404)
        return JSONResponse(content=payload)
    except Exception as e:
        logger.exception("查询 batch-v2 状态失败")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/batch-generate/preview")
async def batch_generate_preview():
    """
    只读取飞书需求表，返回所有行的预览（不触发生成）。
    """
    try:
        fetch_all_source_rows, _, _ = _get_feishu_reader()
        _, rows = await asyncio.to_thread(fetch_all_source_rows)
        return JSONResponse(content={
            "require_confirmation": REQUIRE_DELIVERY_CONFIRMATION,
            "total": len(rows),
            "processable_total": sum(1 for r in rows if r.should_process),
            "rows": [
                {
                    "record_id": r.record_id,
                    "product_title": r.product_title,
                    "has_image": len(r.image_file_tokens) > 0,
                    "has_outfit_images": len(r.outfit_image_file_tokens) > 0,
                    "outfit_image_count": len(r.outfit_image_file_tokens),
                    "review_count": r.review_count,
                    "image_count": r.image_count,
                    "wechat_contacts": r.wechat_contacts,
                    "processing_status": r.processing_status,
                    "should_process": r.should_process,
                }
                for r in rows
            ],
        })
    except Exception as e:
        logger.exception("预览飞书需求表失败")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
