"""买家秀提示词：衣服事实、场景槽位、few-shot、成图硬伤判定。"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
SLOTS_PATH = ROOT / "data" / "buyer_show_slots.json"
FEWSHOTS_PATH = ROOT / "data" / "buyer_show_fewshots.json"

MARKETING_WORDS = (
    "显瘦", "高级感", "高级", "法式", "气质", "爆款", "ins", "韩系", "小众",
    "氛围感", "网红", "爆款推荐",
)

BANNED_SCENE_WORDS = (
    "高级", "氛围感", "电影感", "质感", "精致", "完美", "唯美", "ins", "大片",
    "柔光", "光影", "浅景深", "虚化", "清新", "治愈", "时尚", "模特", "写真",
    "构图", "优雅", "气质",
)

GARMENT_ISSUE_KEYS = ("颜色", "图案", "领口", "袖长", "衣长")
HARD_DEFECTS = (
    "正脸被磨平", "影棚柔光", "背景被清空", "衣服没有任何褶皱",
    "衣服被包装物缠住", "镜子布满水渍", "乱码招牌",
)

OCCASION_GROUPS = {
    "居家": ["居家"],
    "通勤": ["出门", "居家"],
    "日常出门": ["出门", "居家"],
    "约会吃饭": ["出门"],
    "运动": ["出门"],
    "度假": ["出门"],
}

SCENE_SYSTEM = """你在帮淘宝买家写一张晒图的画面描述。
这张图要像买家收到衣服后，用自己的手机拍、直接发到评价区的照片。
它不是商家图，不是小红书封面，也不是写真。

你会收到：
- 衣服：已经确定的一句描述。原样照抄，不改词，不加形容词。
- 本张设定：拍法、地点、光线、身材、两处瑕疵。全部写进去，不能换成别的。

写法：
1. 顺序是：谁用什么方式拍的，在哪、光从哪来、是什么颜色的光，人的身材和动作，衣服穿在身上的状态，背景里的杂物和画面瑕疵。
2. 只写镜头看得见的东西。
3. 衣服要能完整看清。褶皱只能是穿着带来的轻微布料褶，不能在身上加纸板、胶带、快递袋、塑料膜、吊牌或包装。镜子可以有一点指纹，不能布满水珠。店招不要写出能读的错字，模糊即可。颜色和款式写「图1那件」加上给定的衣服描述。
4. 脸和皮肤按原相机写：肤色不均、有细小毛孔。
5. 最后写一句拍法本身的毛病，比如竖拍、手机拿歪、切掉了哪里、哪里有点糊。
6. 输出一段中文，120 到 200 字。
7. 下面这些词一个都不能出现：高级、氛围感、电影感、质感、精致、完美、唯美、ins、大片、柔光、光影、浅景深、虚化、清新、治愈、时尚、模特、写真、构图、优雅、气质。
8. 示例只示范写法，不要照搬示例里的地点、物件和句子。

只输出 JSON：{"prompt": "..."}"""

GARMENT_SYSTEM = """你只描述图中这件衣服本身，参考商品标题补充品类和面料。
标题里的营销词一律忽略：显瘦、高级、法式、气质、爆款、ins、韩系、小众。
颜色和图案以图为准；标题和图片冲突时，写图里看到的。看不清就写「看不清」，不要猜。

只输出 JSON：
{
  "category": "连衣裙/衬衫/T恤/卫衣/毛衣/外套/羽绒服/裤子/半身裙/套装",
  "color": "",
  "pattern": "",
  "material_look": "",
  "drape": "",
  "neckline": "",
  "sleeve": "",
  "length": "",
  "details": [],
  "season": "夏/春秋/冬",
  "occasion": "居家/通勤/日常出门/约会吃饭/运动/度假",
  "one_line": "不带形容词的一句话"
}"""

JUDGE_SYSTEM = """你在检查一张淘宝买家秀成图。图1是商品平铺图，图2是生成的买家秀。
只根据看得见的事实判断。画面不够精致、光线发灰、人头被切掉、背景杂乱，都算正常买家秀，不要因此判失败。

garment_issues：仅当图2的衣服相对图1明显不对时，从以下选择，没有就空数组：
颜色、图案、领口、袖长、衣长

hard_defects：仅当非常明显时，从以下选择，没有就空数组：
正脸被磨平、影棚柔光、背景被清空、衣服没有任何褶皱、衣服被包装物缠住、镜子布满水渍、乱码招牌

只输出 JSON：{"garment_issues": [], "hard_defects": []}"""


@dataclass
class SceneSlot:
    shot: str
    place: str
    light: str
    body: str
    flaws: list[str]


@dataclass
class ImageChecklist:
    garment_issues: list[str] = field(default_factory=list)
    hard_defects: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def needs_retry(self) -> bool:
        return (not self.skipped) and bool(self.garment_issues or self.hard_defects)


@dataclass
class SlotMemory:
    shots: set[str] = field(default_factory=set)
    places: set[str] = field(default_factory=set)
    lights: set[str] = field(default_factory=set)


def load_slots(path: Optional[Path] = None) -> dict[str, Any]:
    slots_path = path or SLOTS_PATH
    return json.loads(slots_path.read_text(encoding="utf-8"))


def load_fewshots(path: Optional[Path] = None) -> list[dict[str, str]]:
    fewshot_path = path or FEWSHOTS_PATH
    data = json.loads(fewshot_path.read_text(encoding="utf-8"))
    return list(data)


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


def wash_title(title: str) -> str:
    text = (title or "").strip()
    for word in MARKETING_WORDS:
        text = text.replace(word, "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,。！!、]{2,}", "，", text).strip("，,。 ")
    return text or (title or "").strip() or "服装"


def garment_from_title(title: str) -> dict[str, str]:
    one_line = wash_title(title)
    return {
        "category": "",
        "color": "",
        "pattern": "",
        "material_look": "",
        "drape": "",
        "neckline": "",
        "sleeve": "",
        "length": "",
        "season": "",
        "occasion": "",
        "one_line": one_line,
    }


def normalize_garment(payload: dict[str, Any], title: str) -> dict[str, str]:
    fallback = garment_from_title(title)
    one_line = str(payload.get("one_line") or "").strip() or fallback["one_line"]
    for word in MARKETING_WORDS:
        one_line = one_line.replace(word, "")
    one_line = one_line.strip("，,。 ") or fallback["one_line"]
    season = str(payload.get("season") or "").strip()
    if season not in {"夏", "春秋", "冬"}:
        season = ""
    occasion = str(payload.get("occasion") or "").strip()
    if occasion not in OCCASION_GROUPS:
        occasion = ""
    result = dict(fallback)
    result.update({
        "category": str(payload.get("category") or ""),
        "color": str(payload.get("color") or ""),
        "pattern": str(payload.get("pattern") or ""),
        "material_look": str(payload.get("material_look") or ""),
        "drape": str(payload.get("drape") or ""),
        "neckline": str(payload.get("neckline") or ""),
        "sleeve": str(payload.get("sleeve") or ""),
        "length": str(payload.get("length") or ""),
        "season": season,
        "occasion": occasion,
        "one_line": one_line,
    })
    return result


def _eligible_places(slots: dict[str, Any], garment: dict[str, str]) -> list[str]:
    season = garment.get("season") or ""
    groups = OCCASION_GROUPS.get(garment.get("occasion") or "", [])
    picked: list[str] = []
    for place in slots.get("places") or []:
        text = str(place.get("text") or "")
        if not text:
            continue
        place_seasons = place.get("seasons") or []
        place_groups = place.get("groups") or []
        if season and place_seasons and season not in place_seasons:
            continue
        if groups and place_groups and not set(groups).intersection(place_groups):
            continue
        picked.append(text)
    if picked:
        return picked
    return [str(p.get("text")) for p in slots.get("places") or [] if p.get("text")]


def _pick_unused(options: list[str], used: set[str], rng: random.Random) -> str:
    fresh = [item for item in options if item not in used]
    pool = fresh or list(options)
    if not fresh:
        used.clear()
    choice = rng.choice(pool)
    used.add(choice)
    return choice


def sample_slot(
    garment: dict[str, str],
    memory: SlotMemory,
    rng: Optional[random.Random] = None,
    slots: Optional[dict[str, Any]] = None,
) -> SceneSlot:
    """抽一组与本商品已用拍法、地点、光线不重复的拍摄设定。"""
    rng = rng or random.Random()
    slots = slots or load_slots()
    shot = _pick_unused(list(slots.get("shots") or []), memory.shots, rng)
    place = _pick_unused(_eligible_places(slots, garment), memory.places, rng)
    light = _pick_unused(list(slots.get("lights") or []), memory.lights, rng)
    bodies = list(slots.get("bodies") or ["匀称"])
    flaw_groups = slots.get("flaws") or {}
    categories = [name for name, items in flaw_groups.items() if items]
    rng.shuffle(categories)
    flaws: list[str] = []
    for name in categories[:2]:
        flaws.append(rng.choice(list(flaw_groups[name])))
    return SceneSlot(
        shot=shot,
        place=place,
        light=light,
        body=rng.choice(bodies),
        flaws=flaws,
    )


def pick_fewshots(
    place: str,
    rng: Optional[random.Random] = None,
    fewshots: Optional[list[dict[str, str]]] = None,
    k: int = 2,
) -> list[dict[str, str]]:
    rng = rng or random.Random()
    fewshots = fewshots if fewshots is not None else load_fewshots()
    pool = [item for item in fewshots if item.get("place") != place]
    if len(pool) < k:
        pool = list(fewshots)
    if not pool:
        return []
    count = min(k, len(pool))
    return rng.sample(pool, count)


def build_scene_messages(
    garment: dict[str, str],
    slot: SceneSlot,
    *,
    reject_note: str = "",
    failure_hint: str = "",
    fewshots: Optional[list[dict[str, str]]] = None,
    rng: Optional[random.Random] = None,
) -> list[dict[str, str]]:
    examples = fewshots if fewshots is not None else pick_fewshots(slot.place, rng)
    messages: list[dict[str, str]] = [{"role": "system", "content": SCENE_SYSTEM}]
    for example in examples:
        messages.append({"role": "user", "content": example.get("user") or ""})
        prompt = example.get("prompt") or ""
        messages.append({
            "role": "assistant",
            "content": json.dumps({"prompt": prompt}, ensure_ascii=False),
        })
    flaws = "；".join(slot.flaws)
    lines = [
        f"衣服：{garment.get('one_line') or '服装'}",
        f"拍法：{slot.shot}",
        f"地点：{slot.place}",
        f"光线：{slot.light}",
        f"身材：{slot.body}",
        f"瑕疵：{flaws}",
    ]
    if reject_note.strip():
        lines.append(f"人工驳回，这次必须改掉：{reject_note.strip()}")
    if failure_hint.strip():
        lines.append(f"上一张的问题，这次必须避开：{failure_hint.strip()}")
    messages.append({"role": "user", "content": "\n".join(lines)})
    return messages


def scene_sentence_from_response(text: str, one_line: str) -> str:
    sentence = ""
    try:
        payload = parse_json_object(text)
        sentence = str(payload.get("prompt") or "").strip()
    except Exception:
        sentence = (text or "").strip()
    if one_line and one_line not in sentence:
        sentence = f"{sentence} 衣服是图1那件{one_line}。"
    return sentence.strip()


def contains_banned_scene_word(text: str) -> str:
    for word in BANNED_SCENE_WORDS:
        if word in (text or ""):
            return word
    return ""


def phone_camera_tail() -> str:
    return (
        "这是一张普通手机照片，竖拍，颜色没有调过。"
        "衣服必须完整、干净、能看清款式。允许轻微褶皱和手机稍歪。"
        "不要在腰上或身上缠纸板、胶带、快递袋、塑料膜，不要挂吊牌。"
        "镜子和镜头保持基本干净，不要布满水珠或污点。"
        "画面里不要多出一个挡在身前的无关的人。"
        "店招和包装文字模糊，不要写出错别字。"
    )


def garment_lock_tail(one_line: str) -> str:
    return (
        f"衣服完全按照图1这件{one_line}：颜色、图案、领口、袖长、衣长、纽扣和面料都和图1一致，"
        "只是穿在人身上，有身体动作带出来的褶皱和垂坠。"
    )


def build_product_only_prompt(scene_sentence: str, one_line: str) -> str:
    return f"{scene_sentence}{garment_lock_tail(one_line)}{phone_camera_tail()}"


def build_same_scene_prompt(scene_sentence: str, one_line: str, variation: str) -> str:
    """后几张锁住第 1 张的人和地点，只换机位。图1 是平铺图，图2 是已通过的第 1 张。"""
    return (
        "图2是已经拍好的第一张买家秀。人物、脸、发型、体型、地点和光线都保持图2，"
        "不要换成另一个人，不要换到另一个地方。"
        f"{variation}。"
        f"地点仍是：{scene_sentence}"
        f"{garment_lock_tail(one_line)}{phone_camera_tail()}"
    )


def build_reference_scene_prompt(one_line: str, reject_note: str = "") -> str:
    text = (
        f"把图2里人物身上的衣服换成图1这件{one_line}。"
        "图2的人物、姿势、手机拍摄角度、背景和光线保持原样。"
        "衣服的颜色、图案、领口、袖长、衣长按图1，亮暗跟着图2的光线变化；衣服上要有图2姿势压出来的褶皱。"
        "保留图2原本的清晰度和噪点，人脸保持图2原样。"
        f"{phone_camera_tail()}"
    )
    if reject_note.strip():
        text += f"人工要求改掉这些问题：{reject_note.strip()}。"
    return text


def build_borrow_pose_prompt(scene_sentence: str, one_line: str) -> str:
    return (
        f"{scene_sentence}"
        "人物站姿和手的位置参考图2；图2里的衣服、人脸和背景都不采用。"
        f"{garment_lock_tail(one_line)}{phone_camera_tail()}"
    )


def checklist_from_response(text: str) -> ImageChecklist:
    try:
        payload = parse_json_object(text)
    except Exception:
        return ImageChecklist(skipped=True)
    issues = _filter_labels(payload.get("garment_issues"), GARMENT_ISSUE_KEYS)
    defects = _filter_labels(payload.get("hard_defects"), HARD_DEFECTS)
    return ImageChecklist(garment_issues=issues, hard_defects=defects, skipped=False)


def _filter_labels(value: Any, allowed: tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        raw_items = re.split(r"[，,、\s]+", value)
    elif isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []
    result: list[str] = []
    allowed_set = set(allowed)
    for item in raw_items:
        name = item.strip()
        if name in allowed_set and name not in result:
            result.append(name)
    return result


def decide_delivery(
    image_result: Optional[dict[str, Any]],
    *,
    requested_count: int,
    require_confirmation: bool,
) -> str:
    """决定这批图要不要等人。

    auto_accept: 张数齐，且每张都看过、没有硬伤。Agent 自己收下。
    human_review: 开了人工确认，但这批图不完整，或看图被跳过，或仍有硬伤。
    proceed: 没开人工确认，直接继续。
    fail: 要图，但一张都没有。
    """
    urls = list((image_result or {}).get("image_urls") or [])
    checks = list((image_result or {}).get("checks") or [])
    if requested_count > 0 and not urls:
        return "fail"
    complete = requested_count <= 0 or len(urls) >= requested_count
    confident = bool(checks) and len(checks) == len(urls) and all(
        not item.get("skipped")
        and not item.get("garment_issues")
        and not item.get("hard_defects")
        for item in checks
        if isinstance(item, dict)
    )
    if complete and confident:
        return "auto_accept"
    if require_confirmation:
        return "human_review"
    return "proceed"


def failure_hint(checklist: ImageChecklist) -> str:
    parts = list(checklist.garment_issues) + list(checklist.hard_defects)
    if not parts:
        return ""
    return "、".join(parts)


def prefer_candidate(previous: ImageChecklist, new: ImageChecklist) -> bool:
    """新图衣服必须正确，且硬伤比旧图少，才替换旧图。"""
    if new.skipped or new.garment_issues:
        return False
    if previous.garment_issues:
        return True
    return len(new.hard_defects) < len(previous.hard_defects)
