"""
LangGraph Agent 节点集合
每个节点只做一件事：读取 AgentState → 执行业务逻辑 → 返回需要更新的字段。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from agent_state import AgentState
from agent_tools import generate_reviews
from buyer_show_review import decide_delivery
from image_generator import extract_garment_facts, run_image_generation
from utils import (
    format_review_text_for_delivery,
    materialize_generated_images,
    extract_product_name,
)

logger = logging.getLogger("agent-nodes")


def parse_input_node(state: AgentState) -> Dict[str, Any]:
    """从输入里取出商品标题。不访问链接。"""
    user_input = state.get("user_input", "")

    product_name = extract_product_name(user_input)

    return {
        "product_name": product_name,
    }


def generate_reviews_node(state: AgentState) -> Dict[str, Any]:
    """根据飞书商品标题和衣服事实生成买家评价。"""

    product_name = state.get("product_name") or "服装商品"
    count = state.get("review_count", 5)
    garment_line = ""
    image_bytes = state.get("product_image_bytes")
    if image_bytes:
        try:
            facts = extract_garment_facts(image_bytes, product_name)
            garment_line = str((facts or {}).get("one_line") or "")
        except Exception as exc:
            logger.warning("评价前看图失败，只按标题写评价: %s", exc)

    try:
        raw = generate_reviews(
            product_name=product_name,
            count=count,
            garment_line=garment_line,
            write_feishu=False,
        )
        formatted = format_review_text_for_delivery(raw)
        return {
            "reviews_raw": raw,
            "reviews_formatted": formatted,
            "garment_one_line": garment_line,
        }
    except Exception as e:
        logger.exception("生成评价失败")
        return {
            "error_message": f"生成评价失败: {str(e)}",
            "reviews_raw": None,
            "reviews_formatted": None,
        }


def generate_images_node(state: AgentState) -> Dict[str, Any]:
    """调用图生图模块生成买家秀配图。"""

    if state.get("skip_image_generation"):
        return {"image_result": None, "local_image_paths": []}

    product_name = state.get("product_name") or "服装商品"
    image_count = state.get("image_count", 2)
    image_bytes = state.get("product_image_bytes")
    outfit_bytes_list = state.get("outfit_image_bytes_list") or []

    if not image_bytes:
        return {
            "image_result": None,
            "local_image_paths": [],
            "error_message": "未提供商品白底图，跳过配图生成",
        }

    try:
        result = run_image_generation(
            image_bytes=image_bytes,
            product_name=product_name,
            scene_count=image_count,
            outfit_image_bytes_list=outfit_bytes_list if outfit_bytes_list else None,
            commit_to_feishu=not state.get("defer_feishu_commit", False),
            review_text=state.get("reviews_formatted") or "",
        )
        # 提取本地图片路径（从 image_result 中的 URL 落地）
        # 批量图片审核模式会先保存候选图，审核通过后再写飞书。
        local_paths: List[str] = []
        if result and result.get("status") in {"success", "partial"} and result.get("image_urls"):
            local_paths = materialize_generated_images(result)
        if result and result.get("status") == "error":
            return {
                "image_result": result,
                "local_image_paths": [],
                "error_message": result.get("message") or "配图生成失败",
            }

        return {
            "image_result": result,
            "local_image_paths": local_paths,
            "delivery_decision": decide_delivery(
                result,
                requested_count=int((result or {}).get("expected_count") or image_count),
                require_confirmation=bool(state.get("require_confirmation")),
            ),
        }
    except Exception as e:
        logger.exception("生成配图失败")
        return {
            "image_result": None,
            "local_image_paths": [],
            "error_message": f"生成配图失败: {str(e)}",
        }


def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """人工确认节点。

    注意：真正的 interrupt 由 graph 层在外部调用前设置，
    这里只做状态的整理和准备，供 interrupt payload 使用。
    实际 interrupt 放在 graph.py 的编译阶段通过 `interrupt` 实现。
    """
    payload = {
        "type": "delivery_approval",
        "review_text": state.get("reviews_formatted", ""),
        "image_paths": state.get("local_image_paths", []),
        "target_contacts": state.get("target_contacts", []),
        "product_name": state.get("product_name", ""),
    }
    # interrupt 会暂停图执行并保存 checkpoint
    # 恢复时，传入的值会作为 result 返回
    result = interrupt(payload)
    return {
        "approval_status": result.get("status", "rejected"),
        "approval_note": result.get("note", ""),
    }


def format_output_node(state: AgentState) -> Dict[str, Any]:
    """最终输出节点：整理所有结果返回给前端。"""
    reply = state.get("reviews_formatted", "")
    if not reply:
        reply = state.get("reviews_raw", "") or state.get("final_reply", "")

    error = state.get("error_message")
    if error and not reply:
        reply = f"⚠️ {error}"

    return {
        "final_reply": reply,
    }


def handle_error_node(state: AgentState) -> Dict[str, Any]:
    """错误处理节点：收集错误信息，优雅降级。"""
    error = state.get("error_message", "未知错误")
    logger.error("流程异常终止: %s", error)
    return {
        "final_reply": f"⚠️ 处理失败：{error}",
    }


# =====================================================================
# 错误路由
# =====================================================================

def route_on_error(state: AgentState) -> str:
    """若 error_message 非空则进入错误处理。"""
    if state.get("error_message"):
        return "handle_error"
    return "continue"


# =====================================================================
# 条件边路由函数
# =====================================================================

def route_after_generate_reviews(state: AgentState) -> str:
    """评价生成后继续配图、人工确认或输出。"""
    if state.get("error_message"):
        return "handle_error"
    if state.get("skip_image_generation"):
        return "human_approval" if state.get("require_confirmation") else "format_output"
    if state.get("product_image_bytes"):
        return "generate_images"
    return "human_approval" if state.get("require_confirmation") else "format_output"


def route_after_generate_images(state: AgentState) -> str:
    """看图通过就自己收下。不完整、评审被跳过或仍未通过时，才进人审。"""
    if state.get("error_message"):
        return "handle_error"
    decision = state.get("delivery_decision") or decide_delivery(
        state.get("image_result"),
        requested_count=int(state.get("image_count") or 0),
        require_confirmation=bool(state.get("require_confirmation")),
    )
    if decision == "fail":
        return "handle_error"
    if decision == "human_review":
        return "human_approval"
    return "format_output"


def route_after_human_approval(state: AgentState) -> str:
    """人工确认后进入最终输出。"""
    if state.get("error_message"):
        return "handle_error"
    return "format_output"
