"""
公共工具函数
从 agent_nodes.py 和 app.py 中抽取的共享逻辑。
"""

from __future__ import annotations

import io
import os
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from config import settings

logger = logging.getLogger("utils")


# ============================================================
# 评价文案格式化
# ============================================================

def normalize_review_block(text: str) -> str:
    """清理单条评价块：去掉 markdown 标记、换行合并为一行。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return " ".join(lines).strip()


def format_review_text_for_delivery(raw_text: Optional[str]) -> str:
    """统一评价正文格式：仅保留评价正文，每条用分隔线连接。"""
    text = (raw_text or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    blocks: List[str] = []

    sections = re.split(r"(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*)?评价\s*\d+(?:\*\*)?\s*[：:]\s*", text)
    if len(sections) > 1:
        for section in sections[1:]:
            m = re.search(
                r"(?:^|\n)\s*(?:[-*]\s*)?(?:内容|正文)\s*[：:]\s*(.+?)(?=\n\s*(?:[-*]\s*)?(?:配图建议|图片建议)\s*[：:]|\Z)",
                section,
                re.S,
            )
            candidate = m.group(1) if m else section
            normalized = normalize_review_block(candidate)
            if normalized:
                blocks.append(normalized)

    if not blocks:
        for m in re.finditer(
            r"(?:^|\n)\s*(?:[-*]\s*)?(?:内容|正文)\s*[：:]\s*(.+?)(?=\n\s*(?:[-*]\s*)?(?:配图建议|图片建议|评价\s*\d+)\s*[：:]|\Z)",
            text,
            re.S,
        ):
            normalized = normalize_review_block(m.group(1))
            if normalized:
                blocks.append(normalized)

    if not blocks:
        kept_lines: List[str] = []
        for line in text.splitlines():
            if re.search(r"(以下是为您生成|配图建议|这些评价|评价\s*\d+)", line):
                continue
            kept_lines.append(line)
        fallback = "\n".join(kept_lines).strip()
        parts = [p.strip() for p in re.split(r"\n\s*\n", fallback) if p.strip()]
        blocks = [normalize_review_block(p) for p in parts if normalize_review_block(p)]

    if not blocks:
        return text

    sep = "\n—————————————\n"
    return sep.join(blocks) + "\n—————————————"


def split_review_blocks(text: Optional[str]) -> List[str]:
    """把已经格式化的评价拆回一条一条。"""
    parts = [part.strip() for part in (text or "").split("—————————————")]
    return [part for part in parts if part]


# ============================================================
# 图片下载和格式识别
# ============================================================

def guess_image_ext(content_type: str, url: str) -> str:
    """根据 Content-Type 或 URL 后缀推断图片扩展名。"""
    ct = (content_type or "").lower()
    if "png" in ct:
        return "png"
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"
    if "webp" in ct:
        return "webp"
    tail = url.rsplit(".", 1)[-1].split("?", 1)[0].lower() if "." in url else ""
    if tail in {"png", "jpg", "jpeg", "webp"}:
        return "jpg" if tail == "jpeg" else tail
    return "png"


def save_phone_jpeg(content: bytes, dest_jpg: str) -> str:
    """按手机相册的压缩质量重存为 JPEG，减弱生图模型的锐利边缘。"""
    from PIL import Image

    image = Image.open(io.BytesIO(content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    os.makedirs(os.path.dirname(os.path.abspath(dest_jpg)), exist_ok=True)
    image.save(dest_jpg, "JPEG", quality=75, optimize=True)
    return dest_jpg


def materialize_generated_images(
    image_result: Optional[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> List[str]:
    """将 image_generator 返回的远程 URL 落地到本地，返回本地路径列表。"""
    if not image_result or not isinstance(image_result, dict):
        return []
    urls = image_result.get("image_urls") or []
    if not urls:
        return []
    saved = [p for p in (image_result.get("local_image_paths") or []) if p and os.path.isfile(p)]
    if len(saved) == len(urls):
        return saved

    generated_dir = output_dir or settings.paths.generated_image_dir
    os.makedirs(generated_dir, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths: List[str] = []
    for idx, url in enumerate(urls, start=1):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            filename = f"ai_{now}_{idx}.jpg"
            path = os.path.join(generated_dir, filename)
            try:
                save_phone_jpeg(resp.content, path)
            except Exception as image_error:
                logger.warning("JPEG 重压失败，改为原样保存: %s", image_error)
                ext = guess_image_ext(resp.headers.get("Content-Type", ""), url)
                path = os.path.join(generated_dir, f"ai_{now}_{idx}.{ext}")
                with open(path, "wb") as f:
                    f.write(resp.content)
            saved_paths.append(path)
        except Exception as e:
            logger.warning("下载生成图失败: %s, error=%s", url, e)
    return saved_paths


# ============================================================
# 输入解析
# ============================================================

URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE,
)


def extract_product_name(user_input: str) -> str:
    """从用户的自然语言输入中尽量提取出商品名称。链接只删掉，不访问。"""
    text = URL_PATTERN.sub(" ", user_input or "")
    m = re.search(r'商品[：:叫是]\s*(.+?)(?:[，,。]|生成|$)', text)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r'(帮我|请|生成|写|条评价|条评论|评价|评论|\d+条?|[，,。！])', '', text).strip()
    return cleaned if cleaned else ""


def merge_replaced_images(existing: List[str], indexes: List[int], replacements: List[str]) -> List[str]:
    """只替换选中下标上的图片，其余保持原路径。"""
    if len(indexes) != len(replacements):
        raise RuntimeError("重做张数和选中张数不一致")
    merged = list(existing)
    for idx, path in zip(indexes, replacements):
        if idx < 0 or idx >= len(merged):
            raise RuntimeError(f"图片序号超出范围: {idx}")
        merged[idx] = path
    return merged


def parse_target_contacts(raw: str) -> List[str]:
    """解析微信联系人字符串为去重列表。"""
    if not raw:
        return []
    normalized = raw.replace("，", ",").replace("\n", ",").replace(";", ",")
    parts = [item.strip() for item in normalized.split(",")]
    seen: set[str] = set()
    contacts: List[str] = []
    for part in parts:
        if not part:
            continue
        if part in seen:
            continue
        seen.add(part)
        contacts.append(part)
    return contacts
