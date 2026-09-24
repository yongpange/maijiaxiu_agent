"""买家秀成图评审：视觉模型自由分析问题，给出下一次要追加的约束。"""
from __future__ import annotations

import base64
import difflib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger("buyer-show-review")

RETRY_HEADER = "【上一张的问题，这次必须改掉】"
PASS_SCORE = 7

REVIEW_SYSTEM = """你在替一家淘宝服装店审核 AI 生成的买家秀。图1是商品平铺图，图2是生成的买家秀。
这张图会放进商品评价区，给准备下单的顾客看。请像挑剔的店主一样看图2，找出所有会让顾客起疑或不想买的地方。

重点看这些方向，但不限于这些：
- 衣服是否还是图1那件：颜色、图案、领口、袖子、长短、版型、面料
- 衣服穿在人身上是否合理：比如裙摆拖地、衣服被东西缠住、腰上挂着纸箱或胶带、布料穿模
- 像不像普通顾客用手机随手拍：是不是像模特写真、棚拍、广告大片
- 画面里有没有不合常理的东西：多出来的人、乱码文字、变形的手脚、脏得看不清衣服的镜子
- 顾客看到这张图，还会不会想下单

只写图2里真实看得到的问题，每条都要具体到部位和现象。
每个问题都要给出一条下一次生图时可以直接执行的中文约束，写成要怎么做，而不是只说不好。
没有明显问题就通过，不要为了挑错而挑错。

只输出 JSON：
{
  "pass": true 或 false,
  "score": 0 到 10 的整数，10 表示完全可以直接放进评价区,
  "problems": ["具体问题"],
  "constraints": ["下一次生图要遵守的约束"]
}"""


@dataclass
class ImageReview:
    passed: bool = False
    score: int = 0
    problems: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    skipped: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pass"] = data.pop("passed")
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "ImageReview":
        data = data or {}
        return cls(
            passed=bool(data.get("pass", data.get("passed", False))),
            score=_clamp_score(data.get("score")),
            problems=_clean_lines(data.get("problems")),
            constraints=_clean_lines(data.get("constraints")),
            skipped=bool(data.get("skipped", False)),
            error=str(data.get("error") or ""),
        )


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, score))


def _clean_lines(value: Any) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[\n；;]+", value)
    elif isinstance(value, list):
        items = [str(item) for item in value]
    else:
        items = []
    result: list[str] = []
    for item in items:
        text = re.sub(r"^\s*(?:\d+[.、)\]]|[-*•])\s*", "", item).strip()
        if text and text not in result:
            result.append(text)
    return result


def parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if fenced:
        raw = fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON 根节点不是对象")
    return data


def review_from_response(text: str) -> ImageReview:
    """解析评审输出。解析不了时 skipped=True，这张图交给人看，不能默认放行。"""
    try:
        payload = parse_json_object(text)
    except Exception as exc:
        return ImageReview(skipped=True, error=f"评审结果无法解析: {exc}")
    review = ImageReview.from_dict(payload)
    if "pass" not in payload and "passed" not in payload:
        review.passed = review.score >= PASS_SCORE and not review.problems
    if review.passed and review.score and review.score < PASS_SCORE:
        review.passed = False
    if not review.passed and review.problems and not review.constraints:
        review.constraints = [f"避免：{problem}" for problem in review.problems]
    return review


def _guess_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
        return "image/webp"
    return "image/jpeg"


def image_content(image_bytes: bytes) -> dict[str, Any]:
    data = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{_guess_mime(image_bytes)};base64,{data}"},
    }


def review_buyer_show(product_bytes: bytes, generated_bytes: Optional[bytes]) -> ImageReview:
    """用视觉模型对照平铺图评审成图。"""
    from config import settings  # pylint: disable=import-outside-toplevel
    from qwen_client import dashscope_configured, qwen_chat  # pylint: disable=import-outside-toplevel

    if not generated_bytes:
        return ImageReview(skipped=True, error="成图下载失败，无法评审")
    if not dashscope_configured():
        return ImageReview(skipped=True, error="未配置 DASHSCOPE_API_KEY，无法评审")
    try:
        text = qwen_chat(
            [
                {"role": "system", "content": REVIEW_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        image_content(product_bytes),
                        image_content(generated_bytes),
                        {"type": "text", "text": "图1是商品平铺图，图2是生成的买家秀。只输出 JSON。"},
                    ],
                },
            ],
            model=settings.llm.vision_model,
            temperature=0.1,
        )
    except Exception as exc:
        logger.warning("成图评审调用失败: %s", exc)
        return ImageReview(skipped=True, error=f"评审调用失败: {exc}")
    return review_from_response(text)


def build_retry_prompt(previous_prompt: str, constraints: list[str]) -> str:
    """在上一次提示词原文后面追加约束。原文不改，差异只在追加的部分。"""
    base = (previous_prompt or "").rstrip()
    lines = _clean_lines(constraints)
    if not lines:
        return base
    numbered = "".join(f"{i}. {line.rstrip('。')}。" for i, line in enumerate(lines, start=1))
    return f"{base}\n{RETRY_HEADER}{numbered}"


def retry_constraints(review: ImageReview, note: str = "") -> list[str]:
    lines = list(review.constraints)
    if note.strip():
        lines.insert(0, f"人工要求：{note.strip()}")
    return _clean_lines(lines)


def prompt_diff(old: str, new: str) -> list[dict[str, str]]:
    """中文按字符比对，返回 equal/insert/delete 片段，供页面高亮。"""
    old = old or ""
    new = new or ""
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    segments: list[dict[str, str]] = []

    def push(op: str, text: str) -> None:
        if not text:
            return
        if segments and segments[-1]["op"] == op:
            segments[-1]["text"] += text
        else:
            segments.append({"op": op, "text": text})

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            push("equal", old[i1:i2])
        elif tag == "insert":
            push("insert", new[j1:j2])
        elif tag == "delete":
            push("delete", old[i1:i2])
        else:
            push("delete", old[i1:i2])
            push("insert", new[j1:j2])
    return segments


def pick_attempt(attempts: list[dict[str, Any]]) -> int:
    """优先最后一次通过的；都没通过时取分数最高的，同分取较新的。"""
    best_index = -1
    best_key: tuple[int, int, int] = (-1, -1, -1)
    for index, attempt in enumerate(attempts):
        if not attempt.get("image_url") and not attempt.get("image_path"):
            continue
        review = ImageReview.from_dict(attempt.get("review"))
        key = (1 if review.passed and not review.skipped else 0, review.score, index)
        if key > best_key:
            best_key = key
            best_index = index
    return best_index


def chosen_attempts(groups: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """每张图最终采用的那次尝试，给人工重做当作底稿。"""
    picked: list[dict[str, Any]] = []
    for group in groups or []:
        attempts = group.get("attempts") or []
        chosen = group.get("chosen")
        if isinstance(chosen, int) and 0 <= chosen < len(attempts):
            picked.append(attempts[chosen])
    return picked


def decide_delivery(
    image_result: Optional[dict[str, Any]],
    *,
    requested_count: int,
    require_confirmation: bool,
) -> str:
    """auto_accept：张数齐且每张最终都通过评审；human_review：其余情况且开了人工确认。"""
    result = image_result or {}
    urls = list(result.get("image_urls") or [])
    if requested_count > 0 and not urls:
        return "fail"
    groups = list(result.get("attempts") or [])
    complete = requested_count <= 0 or len(urls) >= requested_count
    all_passed = bool(groups) and len(groups) == len(urls) and all(
        _chosen_passed(group) for group in groups
    )
    if complete and all_passed:
        return "auto_accept"
    return "human_review" if require_confirmation else "proceed"


def _chosen_passed(group: dict[str, Any]) -> bool:
    attempts = group.get("attempts") or []
    chosen = group.get("chosen")
    if not isinstance(chosen, int) or chosen < 0 or chosen >= len(attempts):
        return False
    review = ImageReview.from_dict(attempts[chosen].get("review"))
    return review.passed and not review.skipped
