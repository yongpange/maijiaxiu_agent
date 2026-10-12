"""Agent 自己做的路由和收下决定。不调用真实模型。"""
from __future__ import annotations

from unittest.mock import patch

from buyer_show_review import ImageReview, decide_delivery
from agent_nodes import (
    generate_reviews_node,
    parse_input_node,
    route_after_generate_images,
)
from agent_graph_v2 import _build_workflow
from image_generator import run_image_generation


def _group(index: int, review: ImageReview) -> dict:
    return {
        "index": index,
        "chosen": 0,
        "attempts": [{
            "image_url": f"http://img/{index}",
            "image_path": f"img-{index}.jpg",
            "review": review.to_dict(),
        }],
    }


def _clean_result(count: int = 2) -> dict:
    review = ImageReview(passed=True, score=8)
    return {
        "image_urls": [f"http://img/{i}" for i in range(count)],
        "attempts": [_group(i, review) for i in range(count)],
    }


def test_missing_doubao_key_does_not_count_as_processed():
    from batch_graph import batch_failure_reason

    reason = batch_failure_reason(
        {"error_message": "未配置 DOUBAO_API_KEY，无法调用豆包生图", "image_result": None},
        image_count=2,
        deferred=False,
    )
    assert "DOUBAO_API_KEY" in reason
    assert batch_failure_reason(
        {"image_result": {"image_urls": [], "committed_to_feishu": False, "message": "配图未生成"}},
        image_count=2,
        deferred=False,
    )


def test_product_link_is_not_crawled():
    update = parse_input_node({"user_input": "帮我生成2条评价，商品：红裙 https://item.example/1"})
    assert update["product_name"] == "红裙"
    assert "crawl_url" not in _build_workflow().nodes
    assert "extract_selling_points" not in _build_workflow().nodes


def test_clean_images_are_auto_accepted_even_if_review_is_enabled():
    decision = decide_delivery(_clean_result(), requested_count=2, require_confirmation=True)
    assert decision == "auto_accept"
    assert route_after_generate_images({
        "image_result": _clean_result(),
        "image_count": 2,
        "require_confirmation": True,
        "delivery_decision": decision,
    }) == "format_output"


def test_uncertain_images_still_wait_for_a_person():
    result = _clean_result()
    result["attempts"][1] = _group(1, ImageReview(passed=False, score=3, problems=["裙摆拖地"], constraints=["裙摆不要拖地"]))
    assert decide_delivery(result, requested_count=2, require_confirmation=True) == "human_review"
    short = _clean_result(1)
    assert decide_delivery(short, requested_count=2, require_confirmation=True) == "human_review"
    skipped = _clean_result()
    skipped["attempts"][0] = _group(0, ImageReview(skipped=True, error="评审结果无法解析"))
    assert decide_delivery(skipped, requested_count=2, require_confirmation=True) == "human_review"
    assert decide_delivery(None, requested_count=2, require_confirmation=True) == "fail"


def test_reviews_use_garment_facts_from_the_flat_lay():
    with patch(
        "agent_nodes.extract_garment_facts",
        return_value={"one_line": "深酒红丝绒连衣裙"},
    ), patch("agent_nodes.generate_reviews", return_value="评价1：\n内容：收到了") as writer:
        update = generate_reviews_node({
            "product_name": "法式连衣裙",
            "review_count": 1,
            "product_image_bytes": b"\xff\xd8jpeg",
        })
    assert update["garment_one_line"] == "深酒红丝绒连衣裙"
    assert writer.call_args.kwargs["garment_line"] == "深酒红丝绒连衣裙"


def test_missing_image_marks_partial_instead_of_full_success():
    with patch(
        "image_generator.extract_garment_facts",
        return_value={"one_line": "黑色羽绒服", "season": "冬", "occasion": "通勤"},
    ), patch(
        "image_generator.write_scene_sentence",
        side_effect=["第一张", "第二张"],
    ), patch(
        "image_generator.generate_lifestyle_image",
        side_effect=["http://ok", None],
    ), patch(
        "image_generator.review_buyer_show",
        return_value=ImageReview(passed=True, score=8),
    ), patch(
        "image_generator._download_image",
        return_value=b"\xff\xd8jpeg",
    ):
        result = run_image_generation(
            image_bytes=b"\xff\xd8jpeg",
            product_name="羽绒服",
            scene_count=2,
            shots_per_scene=1,
            commit_to_feishu=False,
        )
    assert result["status"] == "partial"
    assert result["image_urls"] == ["http://ok"]
    assert decide_delivery(result, requested_count=2, require_confirmation=True) == "human_review"
