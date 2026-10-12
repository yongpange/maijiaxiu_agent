"""评审驱动重做与提示词对比。不调用真实的生图和评审模型。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import image_generator
from buyer_show_review import (
    RETRY_HEADER,
    ImageReview,
    build_retry_prompt,
    chosen_attempts,
    decide_delivery,
    prompt_diff,
    review_from_response,
)
from config import settings

FLOOR_REVIEW = ImageReview(
    passed=False,
    score=3,
    problems=["裙摆拖在地上约 20 厘米，正常穿着不会这样"],
    constraints=["裙长按图1，裙摆停在脚踝上方，不要拖地"],
)
PASS_REVIEW = ImageReview(passed=True, score=8)


@pytest.fixture(autouse=True)
def _tmp_image_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.paths, "generated_image_dir", str(tmp_path))


def _patched_generation(reviews, urls):
    return [
        patch.object(
            image_generator,
            "extract_garment_facts",
            return_value={"one_line": "白色斜肩连衣裙", "season": "夏", "occasion": "日常出门"},
        ),
        patch.object(image_generator, "write_scene_sentence", return_value="阴天在楼下便利店门口随手拍。"),
        patch.object(image_generator, "generate_lifestyle_image", side_effect=list(urls)),
        patch.object(image_generator, "_download_image", return_value=b"\xff\xd8fake-jpeg"),
        patch.object(image_generator, "review_buyer_show", side_effect=list(reviews)),
    ]


def _run(reviews, urls, **kwargs):
    managers = _patched_generation(reviews, urls)
    started = [m.__enter__() for m in managers]
    try:
        result = image_generator.run_image_generation(
            image_bytes=b"\x89PNGproduct",
            product_name="白色斜肩连衣裙",
            scene_count=kwargs.pop("scene_count", 1),
            shots_per_scene=kwargs.pop("shots_per_scene", 1),
            commit_to_feishu=False,
            **kwargs,
        )
    finally:
        for m in reversed(managers):
            m.__exit__(None, None, None)
    return result, started


def test_free_form_problem_triggers_retry_with_constraints():
    review = review_from_response(
        '{"pass": false, "score": 3, "problems": ["腰上挂着一个纸箱"], '
        '"constraints": ["腰部不要出现纸箱、胶带或包装"]}'
    )
    assert review.passed is False
    assert review.problems == ["腰上挂着一个纸箱"]
    assert review.constraints == ["腰部不要出现纸箱、胶带或包装"]


def test_problems_without_constraints_still_become_constraints():
    review = review_from_response('{"pass": false, "score": 2, "problems": ["裙摆拖地"]}')
    assert review.constraints == ["避免：裙摆拖地"]


def test_low_score_cannot_pass():
    assert review_from_response('{"pass": true, "score": 4, "problems": []}').passed is False


def test_broken_review_json_is_not_auto_accepted():
    review = review_from_response("这张图挺好的")
    assert review.skipped is True
    result, _ = _run([review], ["http://img/1"])
    assert decide_delivery(result, requested_count=1, require_confirmation=True) == "human_review"


def test_second_prompt_is_first_prompt_plus_constraints():
    result, (_, scene_mock, _, _, _) = _run([FLOOR_REVIEW, PASS_REVIEW], ["http://img/1", "http://img/2"])
    group = result["attempts"][0]
    first, second = group["attempts"]
    assert second["prompt"].startswith(first["prompt"])
    assert RETRY_HEADER in second["prompt"]
    assert "裙摆停在脚踝上方" in second["prompt"]
    assert second["added_constraints"] == FLOOR_REVIEW.constraints
    assert scene_mock.call_count == 1
    assert group["chosen"] == 1
    assert result["image_urls"] == ["http://img/2"]
    assert first["image_path"] and second["image_path"]
    assert decide_delivery(result, requested_count=1, require_confirmation=True) == "auto_accept"


def test_passing_first_image_is_not_regenerated():
    result, (_, _, gen_mock, _, _) = _run([PASS_REVIEW], ["http://img/1"])
    assert gen_mock.call_count == 1
    assert len(result["attempts"][0]["attempts"]) == 1


def test_both_fail_keeps_higher_score_and_goes_to_human():
    worse = ImageReview(passed=False, score=2, problems=["像棚拍写真"], constraints=["改成手机随手拍"])
    result, _ = _run([FLOOR_REVIEW, worse], ["http://img/1", "http://img/2"])
    group = result["attempts"][0]
    assert group["chosen"] == 0
    assert result["image_urls"] == ["http://img/1"]
    assert decide_delivery(result, requested_count=1, require_confirmation=True) == "human_review"


def test_manual_redo_builds_on_previous_prompt_and_note():
    first, _ = _run([FLOOR_REVIEW, FLOOR_REVIEW], ["http://img/1", "http://img/2"])
    prior = chosen_attempts(first["attempts"])
    redo, (_, scene_mock, _, _, _) = _run(
        [PASS_REVIEW],
        ["http://img/3"],
        prior_attempts=prior,
        reject_note="不要让人物站在马路中间",
    )
    group = redo["attempts"][0]
    assert scene_mock.call_count == 0
    assert group["attempts"][0]["from_previous_round"] is True
    new_prompt = group["attempts"][1]["prompt"]
    assert new_prompt.startswith(prior[0]["prompt"])
    assert "人工要求：不要让人物站在马路中间" in new_prompt
    assert group["chosen"] == 1
    assert redo["image_urls"] == ["http://img/3"]


def test_batch_row_auto_accepts_when_every_image_passes():
    import batch_graph
    from feishu_reader import SourceRow

    image_result, _ = _run([PASS_REVIEW], ["http://img/1"])
    row = SourceRow(
        record_id="rec1",
        product_title="白色斜肩连衣裙",
        image_file_tokens=[],
        outfit_image_file_tokens=[],
        review_count=2,
        image_count=1,
        wechat_contacts=[],
        processing_status="",
        should_process=True,
    )
    agent_result = {
        "reviews_formatted": "穿着很舒服",
        "image_result": image_result,
        "local_image_paths": image_result["local_image_paths"],
    }
    with patch.object(batch_graph, "run_agent_v2", return_value=agent_result), \
            patch.object(batch_graph, "commit_images_to_feishu", return_value={"status": "success"}) as commit, \
            patch.object(batch_graph, "create_delivery_approval") as approval:
        out = batch_graph.process_row_node({
            "row": row,
            "batch_id": "b1",
            "feishu_token": "",
            "default_contacts": [],
            "require_confirmation": True,
        })
    row_result = out["row_results"][0]
    assert row_result["status"] == "success"
    assert row_result["auto_accepted"] is True
    assert row_result["image_attempts"] == image_result["attempts"]
    assert commit.call_count == 1
    assert approval.call_count == 0


def test_failed_anchor_does_not_start_the_rest_of_the_scene():
    result, _ = _run(
        [FLOOR_REVIEW, FLOOR_REVIEW],
        ["http://img/1", "http://img/2"],
        shots_per_scene=3,
    )
    assert result["await_anchor"] is True
    assert result["expected_count"] == 3
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["shot_index"] == 0
    assert decide_delivery(result, requested_count=1, require_confirmation=False) == "fail"
    assert decide_delivery(result, requested_count=1, require_confirmation=True) == "human_review"


def test_passed_anchor_is_the_reference_for_later_shots():
    with patch.object(
        image_generator,
        "generate_lifestyle_image_with_references",
        return_value=("http://img/follow", "image"),
    ) as ref:
        result, (_, _, gen_mock, _, review_mock) = _run(
            [PASS_REVIEW, PASS_REVIEW, PASS_REVIEW],
            ["http://img/1"],
            shots_per_scene=3,
        )
    assert gen_mock.call_count == 1
    assert ref.call_count == 2
    prompts = [group["attempts"][group["chosen"]]["prompt"] for group in result["attempts"][1:]]
    assert all("保持图2" in prompt for prompt in prompts)
    assert [group["shot_index"] for group in result["attempts"]] == [0, 1, 2]
    assert result["await_anchor"] is False
    assert review_mock.call_count == 3


def test_prompt_diff_marks_insert_and_delete():
    segments = prompt_diff("在楼下拍的，裙子到脚踝。", "在楼下拍的，裙子停在脚踝上方。")
    ops = {seg["op"] for seg in segments}
    assert {"equal", "insert", "delete"} <= ops
    rebuilt_new = "".join(s["text"] for s in segments if s["op"] != "delete")
    rebuilt_old = "".join(s["text"] for s in segments if s["op"] != "insert")
    assert rebuilt_new == "在楼下拍的，裙子停在脚踝上方。"
    assert rebuilt_old == "在楼下拍的，裙子到脚踝。"


def test_retry_prompt_diff_is_pure_insertion():
    first = "阴天在楼下便利店门口随手拍。"
    second = build_retry_prompt(first, ["裙摆停在脚踝上方"])
    segments = prompt_diff(first, second)
    assert [s["op"] for s in segments] == ["equal", "insert"]
    assert segments[1]["text"].startswith("\n" + RETRY_HEADER)
