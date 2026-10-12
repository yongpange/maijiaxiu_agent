"""场景槽位、硬伤判定和逐张生图编排。不调用真实模型。"""
from __future__ import annotations

import io
import random
from unittest.mock import patch

from PIL import Image

from buyer_show_review import ImageReview
from buyer_show_prompt import (
    SCENE_SYSTEM,
    ImageChecklist,
    SlotMemory,
    build_scene_messages,
    checklist_from_response,
    garment_from_title,
    pick_fewshots,
    prefer_candidate,
    sample_slot,
    scene_sentence_from_response,
    wash_title,
)
from image_generator import run_image_generation
from utils import save_phone_jpeg


def test_prompt_forbids_packaging_and_dirty_mirror():
    from buyer_show_prompt import phone_camera_tail, checklist_from_response

    text = phone_camera_tail()
    assert "纸板" in text
    assert "水珠" in text
    checklist = checklist_from_response(
        '{"garment_issues":[],"hard_defects":["衣服被包装物缠住","镜子布满水渍"]}'
    )
    assert checklist.needs_retry is True


def test_wash_title_drops_marketing_words():
    assert "法式" not in wash_title("法式显瘦丝绒连衣裙")
    assert garment_from_title("高级感衬衫")["one_line"] == "衬衫"


def test_slots_do_not_repeat_place_light_or_shot():
    garment = {"one_line": "黑色羽绒服", "season": "冬", "occasion": "通勤"}
    memory = SlotMemory()
    rng = random.Random(7)
    slots = [sample_slot(garment, memory, rng) for _ in range(4)]
    assert len({item.place for item in slots}) == 4
    assert len({item.light for item in slots}) == 4
    assert len({item.shot for item in slots}) == 4
    assert all(item.place != "正午夏日街头" for item in slots)
    assert all(len(item.flaws) == 2 for item in slots)


def test_scene_messages_lock_slot_and_include_fewshots():
    garment = {"one_line": "深酒红丝绒连衣裙"}
    from buyer_show_prompt import SceneSlot

    slot = SceneSlot(
        shot="对着落地镜自拍，手机挡住大半张脸",
        place="小区单元门口",
        light="阴天发灰",
        body="微胖",
        flaws=["坐出来的横褶", "门口的电动车"],
    )
    rng = random.Random(1)
    messages = build_scene_messages(
        garment,
        slot,
        reject_note="脸太假",
        failure_hint="正脸被磨平",
        rng=rng,
    )
    assert messages[0]["content"] == SCENE_SYSTEM
    assert "深酒红丝绒连衣裙" in messages[-1]["content"]
    assert "小区单元门口" in messages[-1]["content"]
    assert "正脸被磨平" in messages[-1]["content"]
    assert "脸太假" in messages[-1]["content"]
    places = [item["place"] for item in pick_fewshots("小区单元门口", random.Random(3))]
    assert "小区单元门口" not in places


def test_checklist_ignores_unknown_complaints():
    checklist = checklist_from_response(
        '{"garment_issues":["颜色","氛围感"],"hard_defects":["影棚柔光","不够高级"]}'
    )
    assert checklist.garment_issues == ["颜色"]
    assert checklist.hard_defects == ["影棚柔光"]
    assert checklist.needs_retry is True
    assert checklist_from_response("不是 json").skipped is True


def test_prefer_candidate_keeps_old_when_retry_is_not_better():
    broken = ImageChecklist(garment_issues=["颜色"])
    clean = ImageChecklist()
    still_broken = ImageChecklist(garment_issues=["领口"])
    two = ImageChecklist(hard_defects=["影棚柔光", "背景被清空"])
    one = ImageChecklist(hard_defects=["影棚柔光"])
    assert prefer_candidate(broken, clean) is True
    assert prefer_candidate(broken, still_broken) is False
    assert prefer_candidate(two, one) is True
    assert prefer_candidate(one, two) is False
    assert prefer_candidate(one, ImageChecklist(skipped=True)) is False


def test_scene_sentence_appends_garment_line():
    text = scene_sentence_from_response('{"prompt":"在楼道里拍的。"}', "黑色羽绒服")
    assert "黑色羽绒服" in text


def test_outfit_refs_do_not_write_a_new_scene():
    with patch(
        "image_generator.extract_garment_facts",
        return_value={"one_line": "测试卫衣", "season": "", "occasion": ""},
    ), patch("image_generator.write_scene_sentence") as writer, patch(
        "image_generator.generate_lifestyle_image_with_references",
        return_value=("https://example.com/a.jpg", "image"),
    ) as ref, patch(
        "image_generator.review_buyer_show",
        return_value=ImageReview(passed=True, score=8),
    ), patch(
        "image_generator._download_image",
        return_value=b"outfit-jpeg",
    ):
        result = run_image_generation(
            image_bytes=b"product-image-bytes" * 20,
            product_name="测试卫衣",
            scene_count=2,
            outfit_image_bytes_list=[b"outfit-a" * 30, b"outfit-b" * 30],
            commit_to_feishu=False,
        )
    assert result["status"] == "success"
    assert writer.call_count == 0
    assert ref.call_count == 2
    assert "测试卫衣" in result["prompts"][0]


def test_failed_image_retries_once_on_the_same_prompt():
    from buyer_show_review import RETRY_HEADER

    floor = ImageReview(passed=False, score=3, problems=["裙摆拖地"], constraints=["裙摆停在脚踝上方"])
    still = ImageReview(passed=False, score=2, problems=["裙摆仍拖地"], constraints=["裙摆离开地面"])
    with patch(
        "image_generator.extract_garment_facts",
        return_value={"one_line": "深酒红丝绒连衣裙", "season": "冬", "occasion": "约会吃饭"},
    ), patch(
        "image_generator.write_scene_sentence",
        return_value="第一句",
    ) as writer, patch(
        "image_generator.generate_lifestyle_image",
        side_effect=["http://old", "http://new"],
    ), patch(
        "image_generator._download_image",
        return_value=b"\xff\xd8jpeg",
    ), patch(
        "image_generator.review_buyer_show",
        side_effect=[floor, still],
    ):
        result = run_image_generation(
            image_bytes=b"\xff\xd8jpeg",
            product_name="连衣裙",
            scene_count=1,
            shots_per_scene=1,
            commit_to_feishu=False,
        )
    first, second = result["attempts"][0]["attempts"]
    assert result["image_urls"] == ["http://old"]
    assert writer.call_count == 1
    assert second["prompt"].startswith(first["prompt"])
    assert RETRY_HEADER in second["prompt"]
    assert "裙摆停在脚踝上方" in second["prompt"]


def test_only_selected_index_is_generated():
    with patch(
        "image_generator.extract_garment_facts",
        return_value={"one_line": "深酒红丝绒连衣裙", "season": "冬", "occasion": "约会吃饭"},
    ), patch(
        "image_generator.write_scene_sentence",
        return_value="只写这一张",
    ) as writer, patch(
        "image_generator.generate_lifestyle_image",
        return_value="http://only",
    ) as gen, patch(
        "image_generator.review_buyer_show",
        return_value=ImageReview(passed=True, score=8),
    ), patch(
        "image_generator._download_image",
        return_value=b"\xff\xd8jpeg",
    ):
        result = run_image_generation(
            image_bytes=b"\xff\xd8jpeg",
            product_name="连衣裙",
            scene_count=4,
            shots_per_scene=1,
            only_indexes=[2],
            commit_to_feishu=False,
        )
    assert result["status"] == "success"
    assert result["image_indexes"] == [2]
    assert result["image_urls"] == ["http://only"]
    assert writer.call_count == 1
    assert gen.call_count == 1


def test_scene_rows_keep_three_images_each():
    from image_generator import group_images_by_scene
    from utils import split_review_blocks

    attempts = []
    for scene, shots in ((0, ["a1", "a2", "a3"]), (1, ["b1", "b2", "b3"])):
        for shot, path in enumerate(shots):
            attempts.append({
                "scene_index": scene,
                "shot_index": shot,
                "chosen": 0,
                "attempts": [{"image_path": path}],
            })
    groups = group_images_by_scene(["a1", "a2", "a3", "b1", "b2", "b3"], attempts, 3)
    assert groups == [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
    assert split_review_blocks("第一句\n—————————————\n第二句\n—————————————") == ["第一句", "第二句"]


def test_merge_replaced_images_keeps_the_rest():
    from utils import merge_replaced_images

    merged = merge_replaced_images(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], [2], ["c2.jpg"])
    assert merged == ["a.jpg", "b.jpg", "c2.jpg", "d.jpg"]


def test_reviews_skip_critique_node():
    from agent_graph_v2 import _build_workflow

    workflow = _build_workflow()
    assert "critique_reviews" not in workflow.nodes
    assert "generate_reviews" in workflow.nodes


def test_save_phone_jpeg(tmp_path):
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    dest = tmp_path / "shot.jpg"
    saved = save_phone_jpeg(buf.getvalue(), str(dest))
    assert saved.endswith(".jpg")
    with Image.open(saved) as opened:
        assert opened.format == "JPEG"
