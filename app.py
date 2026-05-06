import io
import os
import tempfile
import time
import uuid
import urllib.request
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import replicate
import streamlit as st
from dotenv import load_dotenv
from PIL import Image


APP_TITLE = "AI 短剧角色一致性助手"
MODEL_VERSION = "google/nano-banana"
# Replicate throttles accounts with low credit (e.g. < $5): ~6 predictions/min, burst 1.
# Three sequential calls need spacing; tune with env REPLICATE_REQUEST_INTERVAL_SEC (default 12).
_DEFAULT_REPLICATE_INTERVAL = float(os.getenv("REPLICATE_REQUEST_INTERVAL_SEC", "12"))
_REPLICATE_429_RETRIES = int(os.getenv("REPLICATE_429_RETRIES", "6"))
# Do NOT ask for a multi-panel "character sheet" here — we generate front/left/back as 3 separate API calls.
BASE_SUFFIX = (
    "High quality character illustration, full body standing, head to toe visible, highly detailed"
)
CONSISTENCY_SUFFIX = (
    "Same character identity, same face, same hairstyle, same outfit, same colors, "
    "same art style as the reference image, keep the exact character design, "
    "identical footwear in every view: exact same shoe model, same shoe color, same laces, "
    "same sole thickness and sole color, same socks or bare feet as reference, "
    "same accessories (bag, belt, jewelry, gloves) in all three views, "
    "no outfit drift between views, full body standing, clean white background"
)
OUTFIT_LOCK_SUFFIX = (
    "Critical: the shoes and clothing must match the reference image pixel-consistent in design. "
    "Do not change shoe type, heel height, or shoe color between views. "
    "If shoes are not visible in the reference, invent ONE plausible shoe design and "
    "use that exact same design in front, left, and back views. "
    "Pants/skirt hem length and how it meets the shoes must be identical across views."
)
NEGATIVE_PROMPT = (
    "different character, different outfit, different face, different hairstyle, "
    "different shoes, mismatched footwear, changing shoe style, wrong shoe color, "
    "barefoot in one view and shod in another, inconsistent accessories, "
    "character turnaround sheet, model sheet, reference sheet, orthographic views, "
    "three views in one image, multiple views in one image, side-by-side panels, "
    "grid layout, comic strip, storyboard, collage, triptych, split screen, "
    "multiple characters, cropped, close-up, extra limbs, deformed, blurry, low quality, "
    "text, watermark, logo"
)

# Per-angle hints: keep one clear camera direction so the model does not fall back to a lineup sheet.
VIEW_EXTRA_HINTS = {
    "front view": (
        "Camera straight-on from the front. One single figure facing the viewer. "
        "Not a lineup, not multiple angles in one frame."
    ),
    "left side view": (
        "Camera strictly from the character's LEFT side: a single left-profile or 3/4-left full body. "
        "The character's nose points toward the RIGHT side of the image. "
        "Exactly ONE person, ONE pose, ONE panel — no front+side+back together, no turnaround sheet."
    ),
    "back view": (
        "Camera straight-on from behind. One single figure with back toward the viewer. "
        "Not a lineup, not multiple angles in one frame."
    ),
}

# After front view is generated, pass it as an extra reference for side/back to lock shoes & hem.
SHOE_LOCK_FROM_FRONT = (
    "Reference image order: the LAST image is the canonical FRONT full-body render from this same run. "
    "Copy its footwear EXACTLY in this new camera angle — same shoe model, silhouette, material, color, "
    "laces or buckles, sole thickness and sole color, socks or tights, and how pants or skirt hem meets the shoes. "
    "Do not redesign shoes. If the FIRST image is the user's original upload, keep identity from it but use the "
    "LAST image as ground truth for shoes and lower-hem details."
)
TEXTONLY_SHOE_LOCK = (
    "The reference image is the canonical FRONT full-body render from this session. "
    "Match shoes, soles, laces, socks, and hem-to-shoe transition exactly for this new viewing angle."
)


def _round_down_to_multiple(x: int, m: int) -> int:
    return max(m, (x // m) * m)


# Local dev: load .env into os.environ (Streamlit Cloud uses st.secrets instead)
load_dotenv()


def get_replicate_api_token() -> Optional[str]:
    """Streamlit Cloud: Secrets first; else environment (e.g. .env / shell)."""
    try:
        if "REPLICATE_API_TOKEN" in st.secrets:
            v = st.secrets["REPLICATE_API_TOKEN"]
            if v and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    t = os.environ.get("REPLICATE_API_TOKEN")
    return t.strip() if t and str(t).strip() else None


def save_upload_to_temp_file(uploaded_file, max_side: int = 768) -> str:
    """
    Streamlit uploads are in-memory.
    Save to a real temp file AND downscale to avoid SDXL img2img OOM on the worker GPU.
    SDXL expects sizes aligned to 64.
    """
    suffix = ".png"  # normalize output to png for consistency

    # UploadedFile may have been read elsewhere in the same run; reset for PIL.
    try:
        uploaded_file.seek(0)
    except (OSError, io.UnsupportedOperation, AttributeError):
        pass

    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    # Downscale large images aggressively; OOM is usually caused by huge input dimensions.
    image.thumbnail((max_side, max_side))
    w, h = image.size
    w = _round_down_to_multiple(w, 64)
    h = _round_down_to_multiple(h, 64)
    image = image.resize((w, h))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="temp_input_") as f:
        image.save(f, format="PNG", optimize=True)
        return f.name


def build_prompt(user_traits: str) -> str:
    user_traits = (user_traits or "").strip()
    if user_traits:
        return f"{user_traits}. {BASE_SUFFIX}. {CONSISTENCY_SUFFIX}. {OUTFIT_LOCK_SUFFIX}"
    return f"{BASE_SUFFIX}. {CONSISTENCY_SUFFIX}. {OUTFIT_LOCK_SUFFIX}"


def has_complex_background(image_path: str) -> bool:
    """
    Heuristic: sample border pixels; if border color varies a lot, assume there's a background.
    This is intentionally simple and conservative.
    """
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < 64 or h < 64:
                return True

            # sample thin border strips
            top = im.crop((0, 0, w, 10))
            bottom = im.crop((0, h - 10, w, h))
            left = im.crop((0, 0, 10, h))
            right = im.crop((w - 10, 0, w, h))
            border = Image.new("RGB", (w * 2 + h * 2, 10))
            border.paste(top.resize((w, 10)), (0, 0))
            border.paste(bottom.resize((w, 10)), (w, 0))
            border.paste(left.resize((h, 10)), (2 * w, 0))
            border.paste(right.resize((h, 10)), (2 * w + h, 0))

            # compute simple per-channel variance proxy
            pixels = list(border.getdata())
            n = len(pixels)
            mean = [sum(p[i] for p in pixels) / n for i in range(3)]
            var = [sum((p[i] - mean[i]) ** 2 for p in pixels) / n for i in range(3)]
            # threshold tuned for "lots of color" vs "flat background"
            return (var[0] + var[1] + var[2]) > 900.0
    except Exception:
        return True


def _save_url_as_temp_png(url: str) -> str:
    """Download model output to a temp PNG for chaining as additional image_input."""
    png = url_to_png_bytes(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="front_lock_") as tf:
        tf.write(png)
        return tf.name


def _replicate_run_with_backoff(
    client: Any,
    model_input: Dict[str, Any],
    image_paths: Optional[List[str]] = None,
):
    """Call Replicate; on 429 throttle, wait and retry. Opens local files as image_input when given."""
    last_err: Optional[Exception] = None
    for attempt in range(_REPLICATE_429_RETRIES):
        try:
            if image_paths:
                with ExitStack() as stack:
                    files = [stack.enter_context(open(p, "rb")) for p in image_paths]
                    inp = dict(model_input)
                    inp["image_input"] = files
                    return client.run(MODEL_VERSION, input=inp)
            return client.run(MODEL_VERSION, input=model_input)
        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            if (
                "429" in str(e)
                or "throttl" in err_text
                or "rate limit" in err_text
                or "too many requests" in err_text
            ):
                # Back off: low-credit tier resets roughly per minute; grow with attempts.
                wait_s = min(90.0, 15.0 * (attempt + 1))
                time.sleep(wait_s)
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("Replicate call failed with no exception recorded")


def run_three_views(base_prompt: str, temp_path: Optional[str], client: Any) -> List[str]:
    """
    Always return 3 images: front / left / back.
    Front is generated first; side/back use [upload + front PNG] to lock shoes and hem (text-only: [front] only).
    """
    views = [
        ("正视图（Front）", "front view"),
        ("左视图（Left）", "left side view"),
        ("后视图（Back）", "back view"),
    ]

    urls: List[str] = []
    front_lock_path: Optional[str] = None

    # Background rule:
    # - If reference image seems to contain a background, force gray background output.
    # - Otherwise use clean white background.
    background_instruction = "clean white background"
    if temp_path is not None and has_complex_background(temp_path):
        background_instruction = "solid neutral light gray background"

    try:
        for idx, (_, view_tag) in enumerate(views):
            # Space out predictions for low-credit rate limits (6/min, burst 1).
            if idx > 0:
                time.sleep(_DEFAULT_REPLICATE_INTERVAL)

            if idx > 0 and front_lock_path is None:
                raise RuntimeError("正视图未成功生成，无法继续生成侧视/后视。")

            view_hint = VIEW_EXTRA_HINTS.get(
                view_tag,
                "Single full-body figure only, one camera angle only, no multi-panel layout.",
            )

            shoe_extra = ""
            if idx > 0 and front_lock_path:
                shoe_extra = f" {SHOE_LOCK_FROM_FRONT}" if temp_path else f" {TEXTONLY_SHOE_LOCK}"

            if idx > 0 and temp_path is None:
                identity_clause = (
                    "Match the reference FRONT render: same face, hair, body proportions, outfit, and art style. "
                )
            else:
                identity_clause = (
                    "Keep the exact same character identity and the exact same art style as the reference image(s). "
                )

            prompt = (
                f"{base_prompt}. "
                f"Output exactly ONE single-panel image: {view_tag}, full body standing. "
                "Composition: one character, one pose, one viewpoint — NOT a character design sheet, "
                "NOT a turnaround, NOT front+side+back in the same picture. "
                f"{view_hint} "
                "Zoomed out to show the full body from head to toe, including shoes. "
                "Uncropped, centered character. "
                f"{identity_clause}"
                "Same complete outfit as the other renders: identical top, bottom, coat, and especially "
                "identical shoes (same shape, color, laces, sole). "
                f"{shoe_extra}"
                f"Avoid: {NEGATIVE_PROMPT}. "
                f"{background_instruction}."
            )

            model_input = {
                "prompt": prompt,
                "seed": 12345,
            }

            if temp_path is not None:
                image_paths: Optional[List[str]] = (
                    [temp_path] if idx == 0 else [temp_path, front_lock_path]
                )
            else:
                image_paths = None if idx == 0 else [front_lock_path]

            out = _replicate_run_with_backoff(client, model_input, image_paths)

            if isinstance(out, (list, tuple)):
                u = to_image_url(out[0]) if out else ""
            else:
                u = to_image_url(out)
            urls.append(u)

            if idx == 0 and u:
                front_lock_path = _save_url_as_temp_png(u)

        return [u for u in urls if u]
    finally:
        if front_lock_path and os.path.exists(front_lock_path):
            try:
                os.remove(front_lock_path)
            except OSError:
                pass


def to_image_url(obj) -> str:
    """
    Replicate may return URLs as strings, or FileOutput-like objects.
    Streamlit st.image expects a URL string / bytes / PIL image, so normalize here.
    """
    if obj is None:
        return ""
    url = getattr(obj, "url", None)
    if isinstance(url, str) and url:
        return url
    if isinstance(obj, str):
        return obj
    # Fallback: many SDK objects stringify to a URL
    return str(obj)


def url_to_png_bytes(url: str) -> bytes:
    """Fetch remote image and normalize to PNG bytes for download."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ai_character/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    im = Image.open(io.BytesIO(raw))
    out = io.BytesIO()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im.convert("RGBA").save(out, format="PNG", optimize=True)
    else:
        im.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


VIEW_CAPTIONS = ["正视图（Front）", "左视图（Left）", "后视图（Back）"]
VIEW_FILENAMES = ["character_front.png", "character_left.png", "character_back.png"]

# Keys for structured text → prompt (文生图)
TEXT_PART_KEYS = ("subject", "face", "hair", "outfit", "vibe", "style_render")


def enrich_character_description(parts: Dict[str, str]) -> str:
    """对已填项保留原文；空项按整体语境做简单、合理的默认补齐（不调用外网）。"""
    subject = (parts.get("subject") or "").strip()
    face = (parts.get("face") or "").strip()
    hair = (parts.get("hair") or "").strip()
    outfit = (parts.get("outfit") or "").strip()
    vibe = (parts.get("vibe") or "").strip()
    style = (parts.get("style_render") or "").strip()

    blob = "".join(parts.get(k, "") for k in TEXT_PART_KEYS).lower()
    anime = any(x in blob for x in ("二次元", "动漫", "漫画", "赛璐璐", "插画", "日系"))
    realistic = any(x in blob for x in ("真人", "写实", "电影", "胶片", "摄影"))

    if not subject:
        subject = (
            "青年角色，性别与后续气质描述一致，身份为短剧主角型人物"
            if not anime
            else "青年动漫角色，性别与气质设定一致，身份为短剧主角型人物"
        )
    if not face:
        if anime:
            face = "五官比例偏动漫范式，轮廓清晰；皮肤上色干净、无明显脏色"
        elif realistic:
            face = "五官比例接近真人，皮肤质感细腻自然，微表情克制"
        else:
            face = "五官比例协调，面部结构与所选画风一致；皮肤质感自然"
    if not hair:
        hair = "发型与脸型、身份匹配，层次清楚；发色与整体色调协调"
    if not outfit:
        outfit = "穿搭与身份和气质统一，上装/下装/鞋层次清晰，配饰数量克制"
    if not vibe:
        vibe = "气质有辨识度、适合镜头叙事，情绪稳定不浮夸"
    if not style:
        if anime:
            style = "二次元角色立绘完成度，线条与上色统一，细节干净"
        elif realistic:
            style = "偏写实插画，材质与光影自然，高清细节"
        else:
            style = "高完成度角色全身插画，光影干净，整体风格统一"

    return (
        f"主体：{subject}。"
        f"五官与肤质：{face}。"
        f"发型：{hair}。"
        f"穿搭与配饰：{outfit}。"
        f"气质与风格：{vibe}。"
        f"画风与质感：{style}。"
    )


def _text_parts_digest(parts: Dict[str, str]) -> int:
    return hash(tuple((parts.get(k) or "").strip() for k in TEXT_PART_KEYS))


HISTORY_MAX_ITEMS = 200
TEXT_PART_LABELS = {
    "subject": "主体",
    "face": "五官",
    "hair": "发型",
    "outfit": "穿搭",
    "vibe": "气质",
    "style_render": "画风",
}


def _append_gen_history(mode: str, input_payload: Dict[str, Any], pngs: List[bytes]) -> None:
    """Newest-first list in session; stores PNG bytes for re-download."""
    now = datetime.now(timezone.utc)
    title = now.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    gh = st.session_state.get("gen_history")
    if not isinstance(gh, list):
        st.session_state["gen_history"] = []
    png_stored = [bytes(x) for x in pngs[:3] if x]
    if len(png_stored) != 3:
        raise ValueError("需要 3 张 PNG 才会写入历史记录")
    st.session_state["gen_history"].insert(
        0,
        {
            "entry_id": str(uuid.uuid4()),
            "ts_iso": now.isoformat(),
            "title": title,
            "mode": mode,
            "input_payload": dict(input_payload),
            "pngs": png_stored,
        },
    )
    if len(st.session_state["gen_history"]) > HISTORY_MAX_ITEMS:
        st.session_state["gen_history"] = st.session_state["gen_history"][:HISTORY_MAX_ITEMS]


def _format_history_input_md(payload: Dict[str, Any]) -> str:
    kind = payload.get("kind")
    if kind == "image":
        fn = payload.get("filename") or "未命名"
        return f"**方式**：图生图\n\n**参考图文件**：`{fn}`"
    parts = payload.get("parts") or {}
    lines: List[str] = ["**方式**：文生图", "", "**分项输入**："]
    any_part = False
    for k in TEXT_PART_KEYS:
        v = (parts.get(k) or "").strip()
        if v:
            any_part = True
            lines.append(f"- **{TEXT_PART_LABELS.get(k, k)}**：{v}")
    if not any_part:
        lines.append("- （无分项，仅完整描述）")
    enriched = (payload.get("enriched") or "").strip()
    if enriched:
        lines.extend(["", "**完整描述（用于生图）**：", enriched])
    return "\n".join(lines)


st.set_page_config(page_title=APP_TITLE, layout="centered")

st.session_state.setdefault("gen_history", [])


def render_history_panel() -> None:
    """历史列表（放在 popover 内；仅点击「历史记录」展开，其它操作不会自动弹出）。"""
    st.caption("按时间倒序（最新在上）。以下为每次的输入摘要与三视图输出，可再次下载 PNG。")
    hist: List[Dict[str, Any]] = st.session_state.get("gen_history", [])
    if not hist:
        st.info("暂无历史记录。")
        return

    c_a, c_b, c_c = st.columns([1, 1, 2])
    with c_a:
        page_size = st.selectbox("每页条数", [10, 20, 50], index=0, key="hist_page_size_sel")
    prev_ps = st.session_state.get("_hist_prev_page_size")
    if prev_ps != page_size:
        st.session_state["hist_page_pick"] = 1
    st.session_state["_hist_prev_page_size"] = page_size

    total = len(hist)
    total_pages = max(1, (total + page_size - 1) // page_size)
    # Streamlit forbids mutating session_state[key] after the widget with that key is created.
    # Clamp page only *before* number_input.
    _hp = int(st.session_state.get("hist_page_pick", 1))
    if _hp < 1:
        st.session_state["hist_page_pick"] = 1
    elif _hp > total_pages:
        st.session_state["hist_page_pick"] = total_pages

    with c_b:
        st.number_input(
            "页码",
            min_value=1,
            max_value=total_pages,
            step=1,
            key="hist_page_pick",
        )

    page = max(1, min(int(st.session_state.get("hist_page_pick", 1)), total_pages))

    with c_c:
        st.write(f"共 **{total}** 条 · 第 **{page}** / **{total_pages}** 页")

    start = (page - 1) * page_size
    chunk = hist[start : start + page_size]

    for local_i, entry in enumerate(chunk):
        global_idx = start + local_i
        eid = entry.get("entry_id") or f"legacy_{global_idx}"
        exp_title = f"{entry.get('title', '')} · {entry.get('mode', '')}"
        with st.expander(exp_title, expanded=False):
            st.markdown(_format_history_input_md(entry.get("input_payload") or {}))
            st.caption("输出 · 三视图")
            cols = st.columns(3)
            pngs = entry.get("pngs") or []
            for j in range(3):
                buf = pngs[j] if j < len(pngs) else None
                with cols[j]:
                    if buf:
                        st.image(buf, caption=VIEW_CAPTIONS[j], width="stretch")
                        st.download_button(
                            label=f"下载 {VIEW_CAPTIONS[j]}",
                            data=buf,
                            file_name=f"hist_{global_idx:04d}_{VIEW_FILENAMES[j]}",
                            mime="image/png",
                            key=f"hdl_{eid}_{j}",
                        )
                    else:
                        st.caption("（无图）")


_h_left, _h_right = st.columns([5, 1])
with _h_left:
    st.title(APP_TITLE)
with _h_right:
    st.write("")
    with st.popover("历史记录", use_container_width=True):
        render_history_panel()

st.caption("输入方式：上传图片 或 输入文字（二选一）。输出：正视 / 左视 / 后视 全身三视图。")

if "text_gen_step" not in st.session_state:
    st.session_state["text_gen_step"] = 0
if "_last_gen_mode" not in st.session_state:
    st.session_state["_last_gen_mode"] = None

st.subheader("输入区")
with st.container(border=True):
    mode = st.radio(
        "选择生成方式",
        ["图生图", "文生图"],
        horizontal=True,
        key="gen_mode",
    )

    prev_mode = st.session_state.get("_last_gen_mode")
    if prev_mode is not None and prev_mode != mode:
        st.session_state["text_gen_step"] = 0
        st.session_state.pop("text_enriched_body", None)
        st.session_state.pop("confirmed_parts_digest", None)
    st.session_state["_last_gen_mode"] = mode

    uploaded = None
    text_parts: Dict[str, str] = {k: "" for k in TEXT_PART_KEYS}

    if mode == "图生图":
        uploaded = st.file_uploader("上传参考图片（jpg/png）", type=["jpg", "jpeg", "png"])
    else:
        st.markdown("**角色描述**（可只填部分；第一次点击将自动补齐空项并供你确认，**再次点击**开始生图。）")
        c1, c2 = st.columns(2)
        with c1:
            text_parts["subject"] = st.text_area(
                "主体（年龄、性别、身份）",
                placeholder="例：28 岁女性，都市白领",
                height=88,
                key="tp_subject",
            )
            text_parts["face"] = st.text_area(
                "五官（脸型、皮肤、五官）",
                placeholder="例：鹅蛋脸，冷白皮，杏眼",
                height=88,
                key="tp_face",
            )
            text_parts["hair"] = st.text_area(
                "发型",
                placeholder="例：及肩黑直发，空气刘海",
                height=88,
                key="tp_hair",
            )
        with c2:
            text_parts["outfit"] = st.text_area(
                "穿搭配饰",
                placeholder="例：黑色长大衣，细跟靴，银色耳钉",
                height=88,
                key="tp_outfit",
            )
            text_parts["vibe"] = st.text_area(
                "气质风格",
                placeholder="例：疏离、克制、赛博冷感",
                height=88,
                key="tp_vibe",
            )
            text_parts["style_render"] = st.text_area(
                "画风 & 质感",
                placeholder="例：日系赛璐璐，干净线稿，电影打光",
                height=88,
                key="tp_style",
            )

        text_parts = {k: (text_parts.get(k) or "").strip() for k in TEXT_PART_KEYS}
        digest = _text_parts_digest(text_parts)
        if st.session_state.get("text_gen_step") == 1:
            if digest != st.session_state.get("confirmed_parts_digest"):
                st.session_state["text_gen_step"] = 0
                st.session_state.pop("text_enriched_body", None)
                st.session_state.pop("confirmed_parts_digest", None)

        if st.session_state.get("text_gen_step") == 1 and st.session_state.get("text_enriched_body"):
            st.success("已生成完整描述。请核对后**再次点击下方按钮**开始生成三视图。")
            st.text_area(
                "完整描述（将用于生图）",
                value=st.session_state["text_enriched_body"],
                height=160,
                disabled=True,
                key="enriched_readonly_display",
            )

    if mode == "图生图":
        do_action = st.button("生成三视图", type="primary", key="btn_image_gen")
    else:
        do_action = st.button("确认描述并生成三视图", type="primary", key="btn_text_gen")

st.write("")
st.divider()
st.subheader("输出区 · 三视图结果")

if do_action:
    _token = get_replicate_api_token()
    if not _token:
        st.error("请在 Streamlit Cloud 的 Settings -> Secrets 中配置 REPLICATE_API_TOKEN")
        st.stop()
    _replicate_client = replicate.Client(auth=_token)

    temp_path = None
    try:
        if mode == "图生图":
            if uploaded is None:
                st.error("请先上传一张参考图片。")
                st.stop()

            prompt = build_prompt("")
            img_name = getattr(uploaded, "name", None) or "upload.png"
            temp_path = save_upload_to_temp_file(uploaded)

            with st.spinner("生成中，请稍候（图生图：约 3 次模型请求）…"):
                output_urls = run_three_views(prompt, temp_path, _replicate_client)

            if not output_urls:
                st.warning("没有拿到模型输出，请稍后重试。")
            else:
                try:
                    pngs = [url_to_png_bytes(u) for u in output_urls[:3]]
                    if len(pngs) != 3:
                        raise ValueError(f"期望 3 张结果图，实际 {len(pngs)} 张")
                    st.session_state["three_views_pngs"] = pngs
                    st.success("生成完成，可在下方预览并下载 PNG。")
                    try:
                        _append_gen_history(
                            "图生图",
                            {"kind": "image", "filename": img_name},
                            pngs,
                        )
                    except Exception as hist_err:
                        st.warning(f"已生成图片，但写入历史记录失败：{hist_err}")
                except Exception as fetch_err:
                    st.session_state.pop("three_views_pngs", None)
                    st.error(f"结果图下载转换失败：{fetch_err}")

        else:
            if not any(text_parts.get(k, "").strip() for k in TEXT_PART_KEYS):
                st.error("请至少填写一项描述。")
                st.stop()

            if st.session_state.get("text_gen_step") == 0:
                enriched = enrich_character_description(text_parts)
                st.session_state["text_enriched_body"] = enriched
                st.session_state["text_gen_step"] = 1
                st.session_state["confirmed_parts_digest"] = _text_parts_digest(text_parts)
                st.rerun()
            else:
                traits_final = (st.session_state.get("text_enriched_body") or "").strip()
                if not traits_final:
                    st.session_state["text_gen_step"] = 0
                    st.error("描述状态异常，请重新点击「确认描述并生成三视图」。")
                    st.stop()

                prompt = build_prompt(traits_final)

                with st.spinner("生成中，请稍候（文生图：约 3 次模型请求）…"):
                    output_urls = run_three_views(prompt, None, _replicate_client)

                if not output_urls:
                    st.warning("没有拿到模型输出，请稍后重试（已保留确认描述，可直接再次点击生成）。")
                else:
                    try:
                        pngs = [url_to_png_bytes(u) for u in output_urls[:3]]
                        if len(pngs) != 3:
                            raise ValueError(f"期望 3 张结果图，实际 {len(pngs)} 张")
                        st.session_state["three_views_pngs"] = pngs
                        st.success("生成完成，可在下方预览并下载 PNG。")
                        try:
                            _append_gen_history(
                                "文生图",
                                {
                                    "kind": "text",
                                    "parts": {k: text_parts.get(k, "") for k in TEXT_PART_KEYS},
                                    "enriched": traits_final,
                                },
                                pngs,
                            )
                        except Exception as hist_err:
                            st.warning(f"已生成图片，但写入历史记录失败：{hist_err}")
                        st.session_state["text_gen_step"] = 0
                        st.session_state.pop("text_enriched_body", None)
                        st.session_state.pop("confirmed_parts_digest", None)
                    except Exception as fetch_err:
                        st.session_state.pop("three_views_pngs", None)
                        st.error(f"结果图下载转换失败：{fetch_err}")

    except Exception as e:
        st.error(f"生成失败：{e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

with st.container(border=True):
    pngs = st.session_state.get("three_views_pngs")
    if pngs:
        cols = st.columns(3)
        for i, buf in enumerate(pngs[:3]):
            with cols[i]:
                st.image(buf, caption=VIEW_CAPTIONS[i], width="stretch")
                st.download_button(
                    label=f"下载 {VIEW_CAPTIONS[i]}",
                    data=buf,
                    file_name=VIEW_FILENAMES[i],
                    mime="image/png",
                    key=f"download_view_{i}",
                )
    else:
        st.info("尚无生成结果。请先在上方「输入区」完成操作。")
