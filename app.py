# -*- coding: utf-8 -*-
"""
ArchMuse - AI 建筑图库与审美推荐系统 (MVP)
Step 3: 专业级三栏式图库浏览与检索 (Pro-Gallery Layout)
"""
import os
import re
import json
import random
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from st_keyup import st_keyup
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# 常量与路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DB_PATH = PROJECT_ROOT / "image_db.json"
USER_PROFILE_PATH = PROJECT_ROOT / "user_profile.json"
BENCHMARK_GT_PATH = PROJECT_ROOT / "benchmark_gt.json"
BAD_CASE_LOG_PATH = PROJECT_ROOT / "bad_case_logs.json"
UPLOAD_BASE = PROJECT_ROOT / "images" / "upload"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Gemini 自动打标 Prompt（全局：本地上传 + Unsplash 收藏均使用）
TAGGING_PROMPT = (
    "作为一名务实且专业的建筑摄影与设计图库打标专家，请解析这张图片，提取最多 20 个核心标签。\n\n"
    "【核心打标纪律（必须严格遵守）】：\n"
    "1. 基于视觉证据：拒绝过度推测！你写的每一个风格或功能词，都必须有画面里的物理实体作为支撑。看图说话，图里没有的绝对不能编。\n"
    "2. 禁用空泛黑话（黑名单）：绝对禁止输出类似「人本设计」、「功能性美学」、「企业文化」、「可持续设计」、「被动式设计」、「互动体验」、「精致主义」这类毫无视觉对应物的假大空营销词汇。\n"
    "3. 词汇颗粒度（六分写实，四分写意）：\n"
    "   - 写实（客观组件/材质）：如 钢结构、混凝土楼板、圆形立柱、穿孔金属板、大理石墙面、移门、落地玻璃...\n"
    "   - 写意（专业风格/氛围）：如 极简主义、粗野主义、暖色灯光、自然采光、延伸视觉、宁静氛围...\n\n"
    "【标准打标示例（不虚不假，精准描述）】：\n"
    "- 示例 A（现代极简室内/厨房）：\n"
    "  [\"厨房空间\", \"木质橱柜\", \"极简主义\", \"水磨石地面\", \"吊灯照明\", \"移门\", \"自然采光\", \"延伸视觉\"]\n"
    "- 示例 B（商业中庭空间）：\n"
    "  [\"商业空间\", \"共享办公\", \"中庭空间\", \"多功能楼梯\", \"钢结构\", \"玻璃栏板\", \"圆形立柱\", \"自然光\"]\n\n"
    "【输出要求】：\n"
    "必须全部使用中文，仅以平铺的 JSON 字符串数组格式返回，禁止输出任何解释性文本。"
)

# 扩展名 -> MIME 类型
EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# 每日推荐：基石词库 + 中文标签对应英文（用于混合搜索）
BASE_SEEDS = [
    "Architecture",
    "Interior Design",
    "Modern Building",
    "Architectural Photography",
    "Landscape Architecture",
]
CHINESE_TAG_TO_ENGLISH = {
    "极简": "Minimalist", "现代": "Modern", "中式": "Chinese style", "清水混凝土": "Concrete",
    "自然光": "Natural light", "住宅": "Residential", "商业建筑": "Commercial",
    "办公建筑": "Office", "玻璃": "Glass", "石材": "Stone", "铝板": "Aluminum",
    "客厅": "Living room", "光影": "Light and shadow", "氛围": "Atmosphere",
    "别墅": "Villa", "合院": "Courtyard", "极简主义": "Minimalism", "原木": "Wood",
    "北欧": "Nordic", "工业风": "Industrial", "新中式": "New Chinese",
}


def init_data_files():
    """项目初始化：确保两个 JSON 数据文件存在。"""
    if not IMAGE_DB_PATH.exists():
        IMAGE_DB_PATH.write_text("[]", encoding="utf-8")
    if not USER_PROFILE_PATH.exists():
        USER_PROFILE_PATH.write_text("{}", encoding="utf-8")
    if not BENCHMARK_GT_PATH.exists():
        BENCHMARK_GT_PATH.write_text("{}", encoding="utf-8")
    if not BAD_CASE_LOG_PATH.exists():
        BAD_CASE_LOG_PATH.write_text("[]", encoding="utf-8")


@st.cache_data
def load_image_db():
    """读取本地图库 JSON。"""
    with open(IMAGE_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(record: dict) -> Path:
    """将 record 中的 file_path 解析为本地绝对路径。"""
    path_str = (record.get("file_path") or "").strip().lstrip("./")
    return PROJECT_ROOT / path_str if path_str else PROJECT_ROOT


def get_image_hash(image_bytes: bytes) -> str:
    """计算图片内容的 MD5 哈希，用于去重。"""
    return hashlib.md5(image_bytes).hexdigest()


@st.cache_data
def get_unique_folders(db: list) -> list[str]:
    """从图库中提取不重复的 folder 列表，排序。"""
    folders = sorted({r.get("folder") or "未分类" for r in db if r})
    return folders


@st.cache_data
def filter_by_folder(db: list, folder: str | None) -> list:
    """按文件夹筛选；None 或 '全部图片' 表示不过滤。"""
    if not folder or folder == "全部图片":
        return list(db)
    return [r for r in db if (r.get("folder") or "未分类") == folder]


@st.cache_data
def get_record_aesthetic_scores(db: list, user_profile: dict) -> dict:
    """预计算每条记录的审美得分（仅当 db 或 user_profile 变化时重算）。"""
    return {
        r.get("id"): sum(user_profile.get(tag, 0) for tag in (r.get("tags") or []))
        for r in db if r and r.get("id")
    }


def filter_by_search(db: list, keyword: str) -> list:
    """多标签交集检索：按 folder / id+文件名 / tags 进行 AND 过滤。"""
    if not keyword or not keyword.strip():
        return list(db)

    # 将中文 / 英文逗号统一替换为空格，并拆分为关键词列表
    normalized = keyword.replace("，", " ").replace(",", " ")
    keywords = [k.strip().lower() for k in normalized.split() if k.strip()]
    if not keywords:
        return list(db)

    out = []
    for r in db:
        folder = (r.get("folder") or "").lower()
        rid = (r.get("id") or "").lower()
        file_path = (r.get("file_path") or "").lower()
        file_name = Path(file_path).name.lower() if file_path else ""
        tags_list = [str(t).lower() for t in (r.get("tags") or [])]

        def matches_one(kw: str) -> bool:
            # a) folder 名中包含
            if kw in folder:
                return True
            # b) id 或 文件名 中包含
            if kw in rid or kw in file_name:
                return True
            # c) 任一标签中模糊匹配
            for t in tags_list:
                if kw in t:
                    return True
            return False

        # AND 逻辑：所有关键词都需命中（对每个 kw，满足 a/b/c 任一条件）
        if all(matches_one(kw) for kw in keywords):
            out.append(r)
    return out


def save_image_db(data):
    """写入本地图库 JSON。"""
    with open(IMAGE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    load_image_db.clear()
    get_unique_folders.clear()
    filter_by_folder.clear()
    get_record_aesthetic_scores.clear()


@st.cache_data
def load_user_profile():
    """读取用户审美画像，不存在则返回 {}。"""
    if not USER_PROFILE_PATH.exists():
        return {}
    with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_user_profile(profile: dict):
    """写入用户审美画像。"""
    with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    load_user_profile.clear()
    get_record_aesthetic_scores.clear()


def load_benchmark_gt():
    if not BENCHMARK_GT_PATH.exists():
        return {}
    with open(BENCHMARK_GT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_benchmark_gt(data):
    with open(BENCHMARK_GT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_bad_case(image_id, action, tag_value, original_tags):
    """
    记录用户修改标签的日志。
    action: "DELETE" (代表 AI 幻觉/误报), "ADD" (代表 AI 漏看/漏报)
    """
    if not BAD_CASE_LOG_PATH.exists():
        return
    try:
        with open(BAD_CASE_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        logs = []
    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_id": image_id,
        "action": action,
        "tag_value": tag_value,
        "original_tags": original_tags,
    })
    with open(BAD_CASE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def next_image_id(existing_list):
    """生成下一个图片 ID，格式 img_001, img_002, ..."""
    if not existing_list:
        return "img_001"
    ids = [item["id"] for item in existing_list if item.get("id", "").startswith("img_")]
    nums = []
    for i in ids:
        try:
            nums.append(int(i.replace("img_", "")))
        except ValueError:
            pass
    next_num = max(nums, default=0) + 1
    return f"img_{next_num:03d}"


def ensure_upload_dir(folder_name):
    """确保 images/upload/{文件夹名称}/ 存在。"""
    safe_name = folder_name.strip() or "未分类"
    dir_path = UPLOAD_BASE / safe_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path, safe_name


def get_tags_for_image(image_bytes: bytes, mime_type: str) -> list[str]:
    """
    使用 Gemini 1.5 Flash 对图片进行视觉解析，返回标签列表。
    使用 .env 中的 GEMINI_API_KEY；解析失败时返回空列表。
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        return []
    try:
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                TAGGING_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
        text = (response.text or "").strip()
        if not text:
            return []
        # 去掉可能的 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        tags = json.loads(text)
        if isinstance(tags, list):
            return [str(t).strip() for t in tags if t]
        return []
    except Exception as e:
        st.error(f"打标失败详细报错：{e}")
        return []


def analyze_image_with_gemini(image_input, mime_type: str = "image/jpeg", max_tags: int = 20) -> list[str]:
    """
    统一打标入口：支持图片二进制或图片 URL，使用 PRD 建筑师 Prompt 返回中文标签（默认最多 20 个）。
    风格、材质、功能、氛围，与 get_tags_for_image 共用同一套 Prompt。
    """
    if isinstance(image_input, str) and image_input.startswith("http"):
        try:
            r = requests.get(image_input, timeout=15)
            r.raise_for_status()
            image_bytes = r.content
        except Exception:
            return []
    else:
        image_bytes = image_input
    tags = get_tags_for_image(image_bytes, mime_type)
    return (tags or [])[:max_tags]


def semantic_evaluate_tags(gt_tags: list, ai_tags: list, image_bytes: bytes, mime_type: str) -> dict:
    """
    使用多模态 Gemini 看着原图当裁判，解决 GT 稀疏性问题。
    返回包含四类结果的字典。
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        gt_set, ai_set = set(gt_tags), set(ai_tags)
        return {"matched_gt": list(gt_set & ai_set), "unlisted_correct": [], "ai_fp": list(ai_set - gt_set), "gt_fn": list(gt_set - ai_set)}

    prompt = f"""
    你是一个极其专业的建筑打标阅卷裁判。
    请你【看着提供的原图】，并对比【人类专家标准标签(GT)】和【AI提取标签(AI)】。

    分类规则：
    1. matched_gt (精准命中): AI 标签与 GT 标签完全相同、同义、或是 GT 的子集/组合词。
    2. unlisted_correct (合理补充): AI 标签【不在 GT 中】，但你看着原图，发现这个标签非常准确地描述了图中的客观实体或氛围（例如 GT 没写"玻璃栏板"，但图里确实有）。这属于 AI 的加分项！
    3. ai_fp (真实幻觉/误报): AI 标签既不在 GT 中，在原图里也【根本找不到、胡编乱造、或严重不符】。
    4. gt_fn (真实漏看/漏报): GT 里明确写了，且图里确实有，但 AI 完全没有提取到相关概念。

    人类专家的标准标签 (GT)：{gt_tags}
    AI 提取的标签 (AI)：{ai_tags}

    请必须只返回如下 JSON 格式（不要带任何markdown代码块）：
    {{
      "matched_gt": [],
      "unlisted_correct": [],
      "ai_fp": [],
      "gt_fn": []
    }}
    """
    try:
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        )
        text = (response.text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        return json.loads(text)
    except Exception:
        gt_set, ai_set = set(gt_tags), set(ai_tags)
        return {"matched_gt": list(gt_set & ai_set), "unlisted_correct": [], "ai_fp": list(ai_set - gt_set), "gt_fn": list(gt_set - ai_set)}


def save_uploaded_files(folder_name, uploaded_files):
    """
    将上传的图片保存到 images/upload/{folder_name}/，调用 Gemini 打标后追加写入 image_db.json。
    返回 (保存数量, 本次打标结果列表 [{id, tags}, ...])。
    """
    if not uploaded_files:
        return 0, []
    dir_path, safe_folder = ensure_upload_dir(folder_name)
    db = load_image_db()
    # 兼容旧数据：仅收集已有的 file_hash，用于快速去重判断
    existing_hashes = {r.get("file_hash") for r in db if r.get("file_hash")}
    saved = 0
    tag_results = []
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        image_bytes = f.getvalue()
        file_hash = get_image_hash(image_bytes)
        # 如果哈希已存在，则跳过本次图片，避免重复调用 Gemini
        if file_hash in existing_hashes:
            st.warning(f"图片 {f.name} 已存在图库中，已为您跳过。")
            continue
        img_id = next_image_id(db)
        # 保存到本地
        dest_path = dir_path / f"{img_id}{ext}"
        dest_path.write_bytes(image_bytes)
        # 立刻触发打标
        mime = EXT_TO_MIME.get(ext, "image/jpeg")
        tags = get_tags_for_image(image_bytes, mime)
        rel_path = f"./images/upload/{safe_folder}/{img_id}{ext}"
        record = {
            "id": img_id,
            "folder": safe_folder,
            "file_path": rel_path,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tags": tags,
            "file_hash": file_hash,
        }
        db.append(record)
        existing_hashes.add(file_hash)
        tag_results.append({"id": img_id, "tags": tags})
        saved += 1
    save_image_db(db)
    return saved, tag_results


# ---------------------------------------------------------------------------
# 页面配置与初始化
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ArchMuse - 建筑图库",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_data_files()

# ---------------------------------------------------------------------------
# 状态管理：current_folder / selected_image / 批量模式
# ---------------------------------------------------------------------------
if "current_folder" not in st.session_state:
    st.session_state.current_folder = "全部图片"
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "batch_selected" not in st.session_state:
    st.session_state.batch_selected = set()
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "last_gallery_search" not in st.session_state:
    st.session_state.last_gallery_search = ""

# ---------------------------------------------------------------------------
# 侧边栏：文件夹命名 + 图片上传
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 上传图片")
    folder_name = st.text_input(
        "文件夹名称 / 项目名",
        value="",
        placeholder="例如：上海别墅项目（留空为「未分类」）",
    )
    uploaded = st.file_uploader(
        "选择图片",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("保存到图库"):
        safe_folder = folder_name.strip() or "未分类"
        n, tag_results = save_uploaded_files(folder_name, uploaded)
        if n:
            st.success("打标成功！")
            st.caption(f"已保存 {n} 张图片到文件夹「{safe_folder}」。")
            # 展示本次生成的 AI 标签
            for item in tag_results:
                with st.container():
                    st.caption(f"**{item['id']}**")
                    if item["tags"]:
                        tag_html = " ".join(
                            f'<span style="display:inline-block;background:#e8e8e8;color:#333;padding:2px 8px;border-radius:12px;margin:2px;font-size:0.85em;">{t}</span>'
                            for t in item["tags"]
                        )
                        st.markdown(tag_html, unsafe_allow_html=True)
                    else:
                        st.caption("（未获取到标签）")
        else:
            st.warning("未保存任何文件（请上传 jpg/png/webp）。")

# ---------------------------------------------------------------------------
# 弹窗：点击放大展示高清原图
# ---------------------------------------------------------------------------
@st.dialog("查看大图")
def show_image_dialog(record):
    """模态框展示单张图片高清原图。"""
    if not record:
        return
    path = resolve_image_path(record)
    if path.exists():
        st.image(str(path), use_container_width=True)
        st.caption(f"📁 {record.get('folder', '')} · {record.get('id', '')}")
    else:
        st.warning(f"本地文件不存在：{path}")


@st.dialog("编辑标签")
def edit_tags_dialog(image_record):
    """独立修改 / 精准删除标签，支持添加新标签并保存到 image_db.json。"""
    if not image_record:
        return
    img_id = image_record.get("id", "") or ""
    tags = list(image_record.get("tags") or [])

    for idx, tag in enumerate(tags):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input("", value=tag, key=f"tag_edit_{img_id}_{idx}", label_visibility="collapsed")
        with col2:
            if st.button("🗑️", key=f"del_tag_{img_id}_{idx}"):
                new_tags = [t for i, t in enumerate(tags) if i != idx]
                log_bad_case(img_id, "DELETE", tag, tags)
                db = load_image_db()
                for r in db:
                    if r.get("id") == img_id:
                        r["tags"] = new_tags
                        break
                save_image_db(db)
                if st.session_state.selected_image and st.session_state.selected_image.get("id") == img_id:
                    st.session_state.selected_image["tags"] = new_tags
                st.toast("标签已更新")
                st.rerun()
                return

    col_new1, col_new2 = st.columns([4, 1])
    with col_new1:
        st.text_input("输入新标签", key=f"new_tag_input_{img_id}", placeholder="输入后点击右侧添加")
    with col_new2:
        if st.button("➕ 添加", key=f"add_tag_{img_id}"):
            v = st.session_state.get(f"new_tag_input_{img_id}", "").strip()
            if v:
                new_tags = tags + [v]
                log_bad_case(img_id, "ADD", v, tags)
                db = load_image_db()
                for r in db:
                    if r.get("id") == img_id:
                        r["tags"] = new_tags
                        break
                save_image_db(db)
                if st.session_state.selected_image and st.session_state.selected_image.get("id") == img_id:
                    st.session_state.selected_image["tags"] = new_tags
                st.toast("标签已更新")
                st.rerun()
                return

    if st.button("💾 保存所有修改", type="primary", key=f"save_tags_{img_id}", use_container_width=True):
        new_tags = []
        for idx in range(len(tags)):
            val = st.session_state.get(f"tag_edit_{img_id}_{idx}", "").strip()
            if val:
                new_tags.append(val)
        db = load_image_db()
        for r in db:
            if r.get("id") == img_id:
                r["tags"] = new_tags
                break
        save_image_db(db)
        if st.session_state.selected_image and st.session_state.selected_image.get("id") == img_id:
            st.session_state.selected_image["tags"] = new_tags
        st.toast("标签已更新")
        st.rerun()


# ---------------------------------------------------------------------------
# 画廊内容局部渲染（@st.fragment：仅此区域刷新，减少全页闪烁）
# ---------------------------------------------------------------------------
_fragment = getattr(st, "fragment", lambda f: f)


def _go_prev_page():
    st.session_state.current_page -= 1


def _go_next_page():
    st.session_state.current_page += 1


@_fragment
def render_gallery_content(display_list):
    """分页 + 图片网格 + 右侧详情，fragment 内交互仅刷新本区域。"""
    placeholder = st.empty()
    placeholder.empty()
    with placeholder.container():
        col_c, col_r = st.columns([6, 1])
        with col_c:
            if not display_list:
                st.info("暂无图片，请从侧边栏上传；或更换文件夹 / 搜索词。")
            else:
                ITEMS_PER_PAGE = 16
                total_pages = max(1, (len(display_list) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
                current_page = max(1, min(st.session_state.current_page, total_pages))
                st.session_state.current_page = current_page
                page_items = display_list[(current_page - 1) * ITEMS_PER_PAGE : current_page * ITEMS_PER_PAGE]
                batch_mode = st.session_state.get("batch_mode_toggle", False)

                COLS_PER_ROW = 4
                for i in range(0, len(page_items), COLS_PER_ROW):
                    row_records = page_items[i : i + COLS_PER_ROW]
                    cols = st.columns(COLS_PER_ROW)
                    for j, record in enumerate(row_records):
                        with cols[j]:
                            path = resolve_image_path(record)
                            st.image(
                                str(path),
                                width=300,
                                output_format="JPEG",
                            )
                            img_id = record.get("id", "") or ""
                            if batch_mode:
                                checked = st.checkbox("选中", value=(img_id in st.session_state.batch_selected), key=f"chk_{img_id}")
                                if checked:
                                    st.session_state.batch_selected.add(img_id)
                                else:
                                    st.session_state.batch_selected.discard(img_id)
                            else:
                                is_selected = (
                                    st.session_state.selected_image
                                    and st.session_state.selected_image.get("id") == img_id
                                )
                                col1, col2 = st.columns(2)
                                with col1:
                                    if is_selected:
                                        st.button("✅ 已选", key=f"sel_{img_id}", type="primary", use_container_width=True)
                                    else:
                                        if st.button("选择", key=f"sel_{img_id}", use_container_width=True):
                                            st.session_state.selected_image = record
                                with col2:
                                    if st.button("🗑️ 删除", key=f"del_{img_id}", use_container_width=True):
                                        if path.exists():
                                            os.remove(str(path))
                                        db_new = [r for r in load_image_db() if r.get("id") != img_id]
                                        save_image_db(db_new)
                                        if st.session_state.selected_image and st.session_state.selected_image.get("id") == img_id:
                                            st.session_state.selected_image = None
                                        st.toast("图片已彻底删除")
                                        st.rerun()

                pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
                with pcol1:
                    st.button("⬅️ 上一页", key="page_prev", use_container_width=True, disabled=(current_page <= 1), on_click=_go_prev_page)
                with pcol2:
                    st.caption(f"第 **{current_page}** / **{total_pages}** 页")
                with pcol3:
                    st.button("下一页 ➡️", key="page_next", use_container_width=True, disabled=(current_page >= total_pages), on_click=_go_next_page)

        with col_r:
            st.subheader("📋 详情")
            sel = st.session_state.selected_image
            if sel:
                path = resolve_image_path(sel)
                if path.exists():
                    st.image(str(path), use_container_width=True, output_format="JPEG")
                if st.button("🔍 查看大图", key="btn_view_big", use_container_width=True):
                    show_image_dialog(sel)
                if st.button("✏️ 编辑标签", key="btn_edit_tags", use_container_width=True):
                    edit_tags_dialog(sel)
                st.caption(f"**📁 文件夹** {sel.get('folder', '—')}")
                st.caption(f"**🕐 上传时间** {sel.get('upload_time', '—')}")
                tags = sel.get("tags") or []
                st.caption("**🏷️ 标签**")
                if tags:
                    tag_html = (
                        '<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">'
                        + "".join(
                            f'<span style="display:inline-block;background:#e8e8e8;color:#333;padding:2px 8px;border-radius:12px;font-size:0.85em;">{t}</span>'
                            for t in tags
                        )
                        + "</div>"
                    )
                    st.markdown(tag_html, unsafe_allow_html=True)
                else:
                    st.caption("无")
                st.divider()
                st.write("")
                if st.button("👍 喜欢", key="btn_like", use_container_width=True):
                    profile = load_user_profile()
                    for tag in (sel.get("tags") or []):
                        profile[tag] = profile.get(tag, 0) + 1
                    save_user_profile(profile)
                    st.toast("已记录偏好")
                    st.rerun()
                if st.button("👎 不喜欢", key="btn_dislike", use_container_width=True):
                    profile = load_user_profile()
                    for tag in (sel.get("tags") or []):
                        profile[tag] = profile.get(tag, 0) - 1
                    save_user_profile(profile)
                    st.toast("减少此类推荐")
                    st.rerun()
            else:
                st.caption("在中间画廊点击「选择」后，此处显示属性。")


# ---------------------------------------------------------------------------
# 主界面：左侧目录 + 中间主区；图库页内部分为 画廊 | 详情 两栏
# ---------------------------------------------------------------------------
def change_folder(name):
    st.session_state.current_folder = name
    st.session_state.current_page = 1


db = load_image_db()
col_left, col_main = st.columns([1, 7])

# ---------- 左侧：目录（文件夹列表）----------
with col_left:
    st.subheader("📂 目录")
    folders = ["全部图片"] + get_unique_folders(db)
    for name in folders:
        count = len(db) if name == "全部图片" else len(filter_by_folder(db, name))
        label = f"{name} ({count})"
        is_current = st.session_state.current_folder == name
        st.button(
            label,
            key=f"folder_{name}",
            type="primary" if is_current else "secondary",
            use_container_width=True,
            on_click=change_folder,
            args=(name,),
        )
    # 管理当前文件夹：重命名 + 删除（仅当当前选中的不是「全部图片」时显示）
    if st.session_state.current_folder != "全部图片":
        current_folder = st.session_state.current_folder
        with st.expander("⚙️ 管理文件夹"):
            # 重命名
            rename_default = st.session_state.get("rename_folder_input", current_folder)
            new_name = st.text_input("新文件夹名称", value=rename_default, key="rename_folder_input", placeholder=current_folder)
            if st.button("确认重命名", key="btn_rename_folder", use_container_width=True):
                new_name = (new_name or "").strip()
                if not new_name:
                    st.error("名称不能为空")
                elif new_name == current_folder:
                    st.info("名称未变更")
                else:
                    # 非法字符（Windows 常见）
                    invalid = any(c in new_name for c in '\\/:*?"<>|')
                    if invalid:
                        st.error("名称不能包含 \\ / : * ? \" < > | 等字符")
                    else:
                        existing = get_unique_folders(load_image_db())
                        if new_name in existing:
                            st.error(f"已存在同名文件夹「{new_name}」")
                        else:
                            old_dir = UPLOAD_BASE / current_folder
                            new_dir = UPLOAD_BASE / new_name
                            try:
                                if not old_dir.exists():
                                    st.error("原文件夹路径不存在，请刷新后重试")
                                else:
                                    os.rename(old_dir, new_dir)
                                    db = load_image_db()
                                    for r in db:
                                        if (r.get("folder") or "未分类") == current_folder:
                                            r["folder"] = new_name
                                            fp = r.get("file_path") or ""
                                            if current_folder in fp:
                                                r["file_path"] = fp.replace(
                                                    f"upload/{current_folder}/",
                                                    f"upload/{new_name}/",
                                                    1
                                                ).replace(
                                                    f"upload/{current_folder}\\",
                                                    f"upload/{new_name}\\",
                                                    1
                                                )
                                    save_image_db(db)
                                    st.session_state.current_folder = new_name
                                    if st.session_state.selected_image and (st.session_state.selected_image.get("folder") or "未分类") == current_folder:
                                        st.session_state.selected_image["folder"] = new_name
                                        fp = st.session_state.selected_image.get("file_path") or ""
                                        if current_folder in fp:
                                            st.session_state.selected_image["file_path"] = fp.replace(f"upload/{current_folder}/", f"upload/{new_name}/", 1).replace(f"upload/{current_folder}\\", f"upload/{new_name}\\", 1)
                                    st.toast("文件夹已重命名")
                                    st.rerun()
                            except OSError as e:
                                st.error(f"重命名失败（可能被占用或权限不足）：{e}")

            st.divider()
            # 删除（二次确认）
            st.caption("删除为危险操作，需输入文件夹名确认。")
            delete_confirm = st.text_input("输入当前文件夹名以确认删除", key="delete_confirm_input", placeholder=current_folder)
            if st.button("🗑️ 删除当前文件夹", key="delete_current_folder", type="primary", use_container_width=True):
                if (delete_confirm or "").strip() != current_folder:
                    st.error("请输入正确的文件夹名称以确认删除")
                else:
                    folder_to_remove = current_folder
                    dir_to_remove = UPLOAD_BASE / folder_to_remove
                    try:
                        if dir_to_remove.exists():
                            shutil.rmtree(dir_to_remove)
                        db_new = [r for r in load_image_db() if (r.get("folder") or "未分类") != folder_to_remove]
                        save_image_db(db_new)
                        st.session_state.current_folder = "全部图片"
                        st.session_state.selected_image = None
                        st.toast("文件夹及内部图片已全部删除")
                        st.rerun()
                    except OSError as e:
                        st.error(f"删除失败（可能被占用或权限不足）：{e}")

# ---------- 中间主区：Tabs（灵感图库 | 审美基因图谱 | 外部探索）----------
with col_main:
    tab_gallery, tab_dashboard, tab_unsplash, tab_eval = st.tabs(["🖼️ 灵感图库", "📈 审美基因图谱", "🌍 外部探索 (Unsplash)", "🧪 AI 评测中心"])

    with tab_gallery:
        st.subheader("🖼️ 画廊")
        search_term = st_keyup("🔍 输入关键词筛选…", key="gallery_search", debounce=500) or ""
        if search_term != st.session_state.last_gallery_search:
            st.session_state.last_gallery_search = search_term
            st.session_state.current_page = 1
        batch_mode = st.toggle("✅ 开启批量管理", key="batch_mode_toggle")

        if search_term.strip():
            display_list = filter_by_search(db, search_term)
        else:
            display_list = filter_by_folder(db, st.session_state.current_folder)

        user_profile = load_user_profile()
        _scores = get_record_aesthetic_scores(db, user_profile)
        display_list = sorted(
            display_list,
            key=lambda r: _scores.get(r.get("id"), 0),
            reverse=True,
        )

        if batch_mode:
            with st.container(border=True):
                st.caption(f"**已选中 {len(st.session_state.batch_selected)} 张图片**")
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    st.text_input("输入目标文件夹名称（若不存在将自动创建）", key="target_folder_input", placeholder="例如：新项目")
                with bc2:
                    do_batch_move = st.button("📁 批量移动", type="primary", key="batch_move_btn", use_container_width=True)
                with bc3:
                    do_batch_del = st.button("🗑️ 批量删除", key="batch_del_btn", use_container_width=True)

            if do_batch_move:
                target = (st.session_state.get("target_folder_input") or "").strip()
                if not target:
                    st.warning("请输入目标文件夹名称。")
                else:
                    dir_dest = UPLOAD_BASE / target
                    dir_dest.mkdir(parents=True, exist_ok=True)
                    db = load_image_db()
                    for rid in list(st.session_state.batch_selected):
                        rec = next((r for r in db if r.get("id") == rid), None)
                        if not rec:
                            continue
                        old_path = resolve_image_path(rec)
                        if not old_path.exists():
                            continue
                        ext = Path(rec.get("file_path", "")).suffix or ".jpg"
                        new_path = dir_dest / f"{rid}{ext}"
                        shutil.move(str(old_path), str(new_path))
                        rec["folder"] = target
                        rec["file_path"] = f"./images/upload/{target}/{rid}{ext}"
                    save_image_db(db)
                    st.session_state.batch_selected.clear()
                    st.toast("批量移动成功")
                    st.rerun()

            if do_batch_del:
                db = load_image_db()
                to_remove = [r for r in db if r.get("id") in st.session_state.batch_selected]
                for rec in to_remove:
                    p = resolve_image_path(rec)
                    if p.exists():
                        os.remove(str(p))
                db_new = [r for r in db if r.get("id") not in st.session_state.batch_selected]
                save_image_db(db_new)
                deleted_ids = st.session_state.batch_selected.copy()
                st.session_state.batch_selected.clear()
                if st.session_state.selected_image and st.session_state.selected_image.get("id") in deleted_ids:
                    st.session_state.selected_image = None
                st.toast("批量删除成功")
                st.rerun()

        render_gallery_content(display_list)

    with tab_dashboard:
        st.subheader("你的专属审美基因分布")
        profile = load_user_profile()
        if not profile:
            st.info("去图库中为喜欢的图片点赞，即可生成你的审美基因分布。")
        else:
            df_all = pd.DataFrame(list(profile.items()), columns=["标签", "得分"])
            df_all = df_all.sort_values("得分", ascending=False)
            top15 = df_all.head(15)
            if top15.empty or len(top15) < 2:
                st.info("数据尚少，多去图库为喜欢的图片点赞，再来查看审美基因图谱。")
            else:
                # 统计指标卡片
                total_genes = len(profile)
                top_tag = top15.iloc[0]["标签"]
                top_score = int(top15.iloc[0]["得分"])
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("已识别审美基因总数", total_genes)
                with m2:
                    st.metric("最显著的审美风格", top_tag)
                with m3:
                    st.metric("该风格累计得分", top_score)

                # 左右两栏：气泡图 + 雷达图
                col_left, col_right = st.columns(2)

                with col_left:
                    st.caption("审美词云 · 气泡大小代表偏好强度")
                    df_bubble = top15.copy()
                    df_bubble = df_bubble.sort_values("得分", ascending=False).reset_index(drop=True)
                    n_b = len(df_bubble)
                    # 物理隔绝：X 轴 1.8 倍等差间距，彻底推开气泡防重叠
                    df_bubble["x"] = [i * 2.2 for i in range(n_b)]
                    df_bubble["size_scaled"] = (df_bubble["得分"] / df_bubble["得分"].max() * 18 + 5).clip(5, 20)
                    # 竖排标签：仅对中文字符按字拆分，英文保持单词完整，前 8 个静态显示
                    def _is_cjk(c):
                        return "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf"

                    def _vertical_text(s):
                        if not s:
                            return ""
                        segments = []
                        word = []
                        for c in str(s):
                            if _is_cjk(c):
                                if word:
                                    segments.append("".join(word))
                                    word = []
                                segments.append(c)
                            else:
                                word.append(c)
                        if word:
                            segments.append("".join(word))
                        return "<br>".join(segments)

                    n_show = min(8, n_b)
                    df_bubble["text_display"] = [
                        _vertical_text(df_bubble["标签"].iloc[i]) if i < n_show else ""
                        for i in range(n_b)
                    ]
                    # 高对比现代配色，色相区分明显
                    high_contrast_colors = [
                        "#264653", "#2A9D8F", "#E9C46A", "#F4A261",
                        "#E76F51", "#8AB17D", "#B5179E", "#4361EE",
                    ]
                    fig_bubble = px.scatter(
                        df_bubble, x="x", y="得分", size="size_scaled", text="text_display",
                        color="标签",
                        color_discrete_sequence=high_contrast_colors,
                        size_max=22,
                        hover_data={"标签": True, "得分": True, "x": False, "size_scaled": False, "text_display": False},
                    )
                    fig_bubble.update_traces(
                        textposition="top center", textfont=dict(size=11), cliponaxis=False,
                        marker=dict(line=dict(width=2, color="white"), opacity=1.0),
                        hovertemplate="<b>标签</b>: %{customdata[0]}<br><b>得分</b>: %{customdata[1]}<extra></extra>",
                    )
                    fig_bubble.update_layout(
                        showlegend=False, margin=dict(l=20, r=20, t=40, b=20),
                        width=1200, height=500,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        hoverlabel=dict(font=dict(size=12), bgcolor="white"),
                    )
                    fig_bubble.update_xaxes(visible=False)
                    fig_bubble.update_yaxes(
                        title_text="得分",
                        showgrid=True, gridcolor="#F0F0F0", zeroline=False,
                    )
                    st.plotly_chart(fig_bubble, use_container_width=False, key="dashboard_bubble")

                with col_right:
                    st.caption("审美偏向 · 极坐标雷达图")
                    n_radar = min(8, len(top15))
                    df_radar = top15.head(n_radar)
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=df_radar["得分"].tolist(),
                        theta=df_radar["标签"].tolist(),
                        fill="toself",
                        line=dict(color="rgb(99, 110, 250)"),
                        fillcolor="rgba(99, 110, 250, 0.4)"
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, max(df_radar["得分"]) * 1.1]),
                                   angularaxis=dict(tickfont=dict(size=10))),
                        showlegend=False, margin=dict(l=20, r=20, t=40, b=20),
                        height=500, paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key="dashboard_radar")

                # 底部：渐变色柱状图（Plotly 悬停精确分值）
                st.caption("Top 15 标签得分 · 悬停查看精确分值")
                fig_bar = px.bar(
                    top15, x="标签", y="得分",
                    color="得分", color_continuous_scale="Blues",
                    text_auto=".0f"
                )
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=40, b=80),
                    height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                )
                fig_bar.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True, key="dashboard_bar")

    with tab_unsplash:
        if "feedback_state" not in st.session_state:
            st.session_state.feedback_state = {}
        st.markdown(
            """<style> button[kind="primary"] { background-color: #e76f51 !important; border-color: #e76f51 !important; color: white !important; } </style>""",
            unsafe_allow_html=True,
        )
        st.subheader("🌍 外部探索 (Unsplash)")
        access_key = os.getenv("UNSPLASH_ACCESS_KEY") or ""
        unsplash_query = (st.text_input("探索外部图库 (支持英文关键词更佳)...", key="unsplash_search") or "").strip()

        def _photo_words(photo):
            text = (photo.get("alt_description") or "") + " " + " ".join(
                (t.get("title", "") or "" for t in photo.get("tags") or [])
            )
            return [w for w in re.findall(r"\w+", text.lower()) if w]

        def _apply_feedback(photo, delta):
            pid = photo.get("id") or ""
            st.session_state.feedback_state[pid] = "liked" if delta == 1 else "disliked"
            words = _photo_words(photo)
            profile = load_user_profile()
            for w in words:
                profile[w] = profile.get(w, 0) + delta
            save_user_profile(profile)
            st.toast("已记录反馈")

        def _mark_saved(pid):
            st.session_state.feedback_state[pid] = "saved"

        _frag_unsplash = getattr(st, "fragment", lambda f: f)

        @_frag_unsplash
        def render_unsplash_explorer(access_key, unsplash_query):
            if not access_key:
                st.warning("未配置 UNSPLASH_ACCESS_KEY，请在 .env 中设置。")
                return

            if not unsplash_query:
                # ======= 模式 1：今日灵感推荐 =======
                st.subheader("🌟 今日灵感推荐")
                need_new_picks = (
                    "unsplash_picks" not in st.session_state
                    or st.session_state.get("refresh_unsplash_picks", False)
                )
                if need_new_picks:
                    profile = load_user_profile()
                    all_results = []
                    if profile:
                        top_2 = sorted(profile.items(), key=lambda x: -x[1])[:2]
                        for tag_cn, _ in top_2:
                            en = CHINESE_TAG_TO_ENGLISH.get(tag_cn.strip(), tag_cn.strip()) or "Architecture"
                            seed = random.choice(BASE_SEEDS)
                            query = f"{en} {seed}"
                            try:
                                r = requests.get(
                                    "https://api.unsplash.com/search/photos",
                                    params={
                                        "query": query,
                                        "client_id": access_key,
                                        "per_page": 15,
                                        "content_filter": "high",
                                        "orientation": "landscape",
                                    },
                                    timeout=10,
                                )
                                r.raise_for_status()
                                data = r.json()
                                all_results.extend(data.get("results") or [])
                            except Exception:
                                pass
                    if not all_results:
                        try:
                            r = requests.get(
                                "https://api.unsplash.com/search/photos",
                                params={
                                    "query": "Architecture Interior Design",
                                    "client_id": access_key,
                                    "per_page": 20,
                                    "content_filter": "high",
                                    "orientation": "landscape",
                                },
                                timeout=10,
                            )
                            r.raise_for_status()
                            all_results = (r.json() or {}).get("results") or []
                        except Exception:
                            pass
                    seen_ids = set()
                    unique_results = []
                    for p in all_results:
                        pid = p.get("id")
                        if pid and pid not in seen_ids:
                            seen_ids.add(pid)
                            unique_results.append(p)
                    n_target = 8
                    picks = random.sample(unique_results, min(n_target, len(unique_results))) if unique_results else []
                    st.session_state.unsplash_picks = picks
                    st.session_state.refresh_unsplash_picks = False
                else:
                    picks = st.session_state.unsplash_picks

                if picks:
                    if st.button("🔄 换一批", key="btn_refresh_unsplash_picks", use_container_width=False):
                        st.session_state.refresh_unsplash_picks = True
                        st.session_state.feedback_state = {}
                        st.rerun()
                    COLS_U = 4
                    for i in range(0, len(picks), COLS_U):
                        row = picks[i : i + COLS_U]
                        cols_u = st.columns(COLS_U)
                        for j, photo in enumerate(row):
                            with cols_u[j]:
                                img_url = photo.get("urls", {}).get("regular") or photo.get("urls", {}).get("small") or ""
                                if img_url:
                                    st.image(img_url, use_container_width=True)
                                user = photo.get("user") or {}
                                name = user.get("name") or "Unknown"
                                user_url = (user.get("links") or {}).get("html") or "https://unsplash.com"
                                user_url = user_url + ("&" if "?" in user_url else "?") + "utm_source=ArchMuse&utm_medium=referral"
                                st.markdown(
                                    f'Photo by [{name}]({user_url}) on [Unsplash](https://unsplash.com/?utm_source=ArchMuse&utm_medium=referral)'
                                )
                                pid = photo.get("id") or ""
                                fb = st.session_state.feedback_state
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.button(
                                        "👍",
                                        key=f"rec_like_{pid}",
                                        type="primary" if fb.get(pid) == "liked" else "secondary",
                                        use_container_width=True,
                                        on_click=_apply_feedback,
                                        args=(photo, 1),
                                    )
                                with c2:
                                    st.button(
                                        "👎",
                                        key=f"rec_dis_{pid}",
                                        type="primary" if fb.get(pid) == "disliked" else "secondary",
                                        use_container_width=True,
                                        on_click=_apply_feedback,
                                        args=(photo, -1),
                                    )
                                with c3:
                                    if st.button(
                                        "📁",
                                        key=f"rec_save_{pid}",
                                        type="primary" if fb.get(pid) == "saved" else "secondary",
                                        use_container_width=True,
                                        on_click=_mark_saved,
                                        args=(pid,),
                                    ):
                                        img_url = photo.get("urls", {}).get("regular") or photo.get("urls", {}).get("small")
                                        image_bytes = None
                                        if img_url:
                                            try:
                                                resp = requests.get(img_url, timeout=15)
                                                resp.raise_for_status()
                                                image_bytes = resp.content
                                            except Exception as e:
                                                st.error(f"保存失败：{e}")
                                                return
                                        if not image_bytes:
                                            st.error("图片下载失败，请重试")
                                            return
                                        with st.spinner("AI 正在深度解析中..."):
                                            tags = analyze_image_with_gemini(image_bytes, mime_type="image/jpeg", max_tags=20)
                                        profile = load_user_profile()
                                        for t in tags:
                                            profile[t] = profile.get(t, 0) + 1
                                        save_user_profile(profile)
                                        dl_url = (photo.get("links") or {}).get("download_location")
                                        if dl_url:
                                            try:
                                                requests.get(dl_url, params={"client_id": access_key}, timeout=5)
                                            except Exception:
                                                pass
                                        save_dir, safe_folder = ensure_upload_dir("Unsplash")
                                        file_id = f"unsplash_{pid}"
                                        ext = ".jpg"
                                        local_path = save_dir / f"{file_id}{ext}"
                                        try:
                                            local_path.write_bytes(image_bytes)
                                        except Exception as e:
                                            st.error(f"写入磁盘失败: {e}")
                                        else:
                                            db = load_image_db()
                                            rel_path = f"./images/upload/{safe_folder}/{file_id}{ext}"
                                            record = {
                                                "id": file_id,
                                                "folder": safe_folder,
                                                "file_path": rel_path,
                                                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                "tags": tags,
                                            }
                                            db.append(record)
                                            save_image_db(db)
                                            st.session_state.current_folder = "Unsplash"
                                            st.toast("已保存到收藏")
                else:
                    st.info("暂无推荐，去完善偏好后再来。")

            else:
                # ======= 模式 2：搜索外部图库 =======
                try:
                    r = requests.get(
                        "https://api.unsplash.com/search/photos",
                        params={"query": unsplash_query.strip(), "client_id": access_key, "per_page": 20},
                        timeout=10,
                    )
                    r.raise_for_status()
                    data = r.json()
                    results = data.get("results") or []
                except Exception as e:
                    st.error(f"请求 Unsplash 失败：{e}")
                    results = []
                if results:
                    profile = load_user_profile()

                    def unsplash_score(item):
                        text = (item.get("alt_description") or "") + " " + " ".join(
                            (t.get("title", "") or "" for t in item.get("tags") or [])
                        )
                        words = re.findall(r"\w+", text.lower())
                        return sum(profile.get(w, 0) for w in words)

                    results = sorted(results, key=unsplash_score, reverse=True)
                    COLS_U = 4
                    for i in range(0, len(results), COLS_U):
                        row = results[i : i + COLS_U]
                        cols_u = st.columns(COLS_U)
                        for j, photo in enumerate(row):
                            with cols_u[j]:
                                img_url = photo.get("urls", {}).get("regular") or photo.get("urls", {}).get("small") or ""
                                if img_url:
                                    st.image(img_url, use_container_width=True)
                                user = photo.get("user") or {}
                                name = user.get("name") or "Unknown"
                                user_url = (user.get("links") or {}).get("html") or "https://unsplash.com"
                                user_url = user_url + ("&" if "?" in user_url else "?") + "utm_source=ArchMuse&utm_medium=referral"
                                st.markdown(
                                    f'Photo by [{name}]({user_url}) on [Unsplash](https://unsplash.com/?utm_source=ArchMuse&utm_medium=referral)'
                                )
                                pid = photo.get("id") or ""
                                fb = st.session_state.feedback_state
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.button(
                                        "👍",
                                        key=f"like_{pid}",
                                        type="primary" if fb.get(pid) == "liked" else "secondary",
                                        use_container_width=True,
                                        on_click=_apply_feedback,
                                        args=(photo, 1),
                                    )
                                with c2:
                                    st.button(
                                        "👎",
                                        key=f"dis_{pid}",
                                        type="primary" if fb.get(pid) == "disliked" else "secondary",
                                        use_container_width=True,
                                        on_click=_apply_feedback,
                                        args=(photo, -1),
                                    )
                                with c3:
                                    if st.button(
                                        "📁",
                                        key=f"save_{pid}",
                                        type="primary" if fb.get(pid) == "saved" else "secondary",
                                        use_container_width=True,
                                        on_click=_mark_saved,
                                        args=(pid,),
                                    ):
                                        img_url = photo.get("urls", {}).get("regular") or photo.get("urls", {}).get("small")
                                        image_bytes = None
                                        if img_url:
                                            try:
                                                resp = requests.get(img_url, timeout=15)
                                                resp.raise_for_status()
                                                image_bytes = resp.content
                                            except Exception as e:
                                                st.error(f"保存失败：{e}")
                                                return
                                        if not image_bytes:
                                            st.error("图片下载失败，请重试")
                                            return
                                        with st.spinner("AI 正在深度解析中..."):
                                            tags = analyze_image_with_gemini(image_bytes, mime_type="image/jpeg", max_tags=20)
                                        profile = load_user_profile()
                                        for t in tags:
                                            profile[t] = profile.get(t, 0) + 1
                                        save_user_profile(profile)
                                        dl_url = (photo.get("links") or {}).get("download_location")
                                        if dl_url:
                                            try:
                                                requests.get(dl_url, params={"client_id": access_key}, timeout=5)
                                            except Exception:
                                                pass
                                        save_dir, safe_folder = ensure_upload_dir("Unsplash")
                                        file_id = f"unsplash_{pid}"
                                        ext = ".jpg"
                                        local_path = save_dir / f"{file_id}{ext}"
                                        try:
                                            local_path.write_bytes(image_bytes)
                                        except Exception as e:
                                            st.error(f"写入磁盘失败: {e}")
                                        else:
                                            db = load_image_db()
                                            rel_path = f"./images/upload/{safe_folder}/{file_id}{ext}"
                                            record = {
                                                "id": file_id,
                                                "folder": safe_folder,
                                                "file_path": rel_path,
                                                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                "tags": tags,
                                            }
                                            db.append(record)
                                            save_image_db(db)
                                            st.session_state.current_folder = "Unsplash"
                                            st.toast("已保存到收藏")
                else:
                    st.info("未找到相关图片或暂无结果，请换关键词重试。")

        render_unsplash_explorer(access_key, unsplash_query)

    with tab_eval:
        st.header("🧪 AI 模型评估看板 (Auto-Eval Dashboard)")
        eval_tab1, eval_tab2 = st.tabs(["1. 构建黄金测试集 (Ground Truth)", "2. 运行自动化评测"])

        with eval_tab1:
            st.markdown("请从本地图库中选择典型图片，并作为**人类专家**为其打上绝对准确的标签。")
            db = load_image_db()
            gt_data = load_benchmark_gt()

            if not db:
                st.warning("本地图库为空，请先上传图片。")
            else:
                # 1. 顶部状态筛选器
                filter_opt = st.radio("筛选图片状态", ["⏳ 未加入测试集", "✅ 已加入测试集", "全部图片"], horizontal=True)

                # 初始化/处理通过底部画廊点击传上来的编辑目标
                if "eval_edit_target" not in st.session_state:
                    st.session_state.eval_edit_target = None

                # 2. 过滤下拉列表
                filtered_options = {}
                for r in db:
                    rid = r.get("id")
                    if not rid:
                        continue
                    is_in_gt = rid in gt_data
                    if filter_opt == "✅ 已加入测试集" and not is_in_gt:
                        continue
                    if filter_opt == "⏳ 未加入测试集" and is_in_gt:
                        continue
                    filtered_options[rid] = r

                if not filtered_options:
                    st.info(f"当前「{filter_opt}」分类下没有图片。")
                else:
                    options_list = list(filtered_options.keys())
                    # 如果有来自底部画廊的点击，强制跳转到该图片
                    default_index = 0
                    if st.session_state.eval_edit_target in options_list:
                        default_index = options_list.index(st.session_state.eval_edit_target)
                        st.session_state.eval_edit_target = None  # 消费掉该状态

                    sel_id = st.selectbox(
                        "选择并编辑测试图片",
                        options=options_list,
                        index=default_index,
                        format_func=lambda x: f"{'✅' if x in gt_data else '⏳'} {x} ({filtered_options[x].get('folder', '未分类')})",
                    )

                    if sel_id:
                        rec = filtered_options[sel_id]
                        path = resolve_image_path(rec)
                        col_img, col_gt = st.columns([1, 2])
                        with col_img:
                            if path.exists():
                                st.image(str(path), use_container_width=True)
                        with col_gt:
                            st.caption(f"当前 AI 提取的标签：{', '.join(rec.get('tags', []))}")
                            default_gt = ", ".join(gt_data.get(sel_id, rec.get("tags", [])))
                            gt_input = st.text_area("专家标定标签 (Ground Truth)，请用逗号分隔", value=default_gt, height=100)

                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button("💾 保存 / 更新黄金标准", key=f"save_gt_{sel_id}", type="primary", use_container_width=True):
                                    cleaned_gt = [t.strip() for t in gt_input.replace("，", ",").split(",") if t.strip()]
                                    gt_data[sel_id] = cleaned_gt
                                    save_benchmark_gt(gt_data)
                                    st.toast(f"已更新 {sel_id} 的专家标签！")
                                    st.rerun()
                            with c2:
                                if sel_id in gt_data:
                                    if st.button("🗑️ 从测试集中移除", key=f"del_gt_{sel_id}", use_container_width=True):
                                        del gt_data[sel_id]
                                        save_benchmark_gt(gt_data)
                                        st.toast(f"已将 {sel_id} 移出测试集")
                                        st.rerun()

                # 3. 底部展示：已加入的黄金测试集画廊
                st.divider()
                st.subheader(f"📚 已建立的黄金测试集 ({len(gt_data)} 张)")
                if gt_data:
                    gt_cols = st.columns(6)
                    gt_items = list(gt_data.items())
                    for i, (gid, gtags) in enumerate(gt_items):
                        with gt_cols[i % 6]:
                            grec = next((r for r in db if r.get("id") == gid), None)
                            if grec:
                                gpath = resolve_image_path(grec)
                                if gpath.exists():
                                    st.image(str(gpath), use_container_width=True)
                                    if st.button("✏️ 编辑", key=f"edit_gallery_{gid}", use_container_width=True):
                                        st.session_state.eval_edit_target = gid
                                        st.rerun()
                else:
                    st.caption("暂无数据，请在上方添加。")

        with eval_tab2:
            st.markdown("使用当前的 `TAGGING_PROMPT` 重新让 AI 对测试集打标，并计算准确率与召回率。")
            gt_data = load_benchmark_gt()

            if not gt_data:
                st.info("尚未建立测试集，请先在左侧 Tab 中添加 Ground Truth。")
            else:
                st.metric("黄金测试集规模", f"{len(gt_data)} 张图片")
                if st.button("🚀 开始自动化跑分", type="primary", use_container_width=True):
                    db = load_image_db()
                    results = []
                    total_tp, total_fp, total_fn = 0, 0, 0

                    progress_text = "AI 评测运行中..."
                    my_bar = st.progress(0, text=progress_text)

                    items = list(gt_data.items())
                    for i, (img_id, gt_tags) in enumerate(items):
                        rec = next((r for r in db if r.get("id") == img_id), None)
                        if not rec:
                            continue
                        path = resolve_image_path(rec)
                        if not path.exists():
                            continue

                        image_bytes = path.read_bytes()
                        ext = path.suffix.lower()
                        mime = "image/png" if ext == ".png" else "image/jpeg"
                        ai_tags = analyze_image_with_gemini(image_bytes, mime_type=mime)

                        eval_res = semantic_evaluate_tags(gt_tags, ai_tags, image_bytes, mime)

                        matched = eval_res.get("matched_gt", [])
                        unlisted = eval_res.get("unlisted_correct", [])
                        fp_list = eval_res.get("ai_fp", [])
                        fn_list = eval_res.get("gt_fn", [])

                        tp_len = len(matched)
                        correct_len = tp_len + len(unlisted)

                        total_tp += correct_len
                        total_fp += len(fp_list)
                        total_fn += len(fn_list)

                        results.append({
                            "图片 ID": img_id,
                            "真实准确率": f"{(correct_len / len(ai_tags) * 100) if ai_tags else 0:.1f}%",
                            "召回率 (Recall)": f"{(tp_len / len(gt_tags) * 100) if gt_tags else 0:.1f}%",
                            "合理补充 (AI加分)": ", ".join(unlisted),
                            "真实幻觉 (胡编乱造)": ", ".join(fp_list),
                            "AI 漏看 (漏报)": ", ".join(fn_list),
                        })
                        my_bar.progress((i + 1) / len(items), text=f"正在评测 {img_id}...")

                    my_bar.empty()

                    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
                    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
                    f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) else 0

                    st.subheader("📊 评测总览")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("宽容准确率 (Lenient Precision)", f"{overall_precision * 100:.1f}%", help="AI 给出的标签里，有多少是对的（含合理补充；越低代表幻觉越严重）")
                    c2.metric("整体召回率 (Recall)", f"{overall_recall * 100:.1f}%", help="专家标注的标签里，AI 找到了多少（越低代表 AI 漏看的越多）")
                    c3.metric("综合 F1-Score", f"{f1_score * 100:.1f}%", help="准确率和召回率的调和平均数，综合衡量模型打标质量")

                    st.subheader("🔍 Bad Case 详细诊断")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)

