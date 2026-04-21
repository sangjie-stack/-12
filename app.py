import streamlit as st

from utils.stage3_streamlit import (
    APP_CSS,
    bgr_to_rgb,
    build_about_context,
    build_badge_row,
    build_empty_state,
    build_key_value_list,
    build_panel_card,
    build_stage3_model_info,
    build_stat_grid,
    image_bytes_to_pil,
    pil_to_bgr,
    run_geometry_detection,
)


DETECT_MODE_OPTIONS = {
    "single": "单主体：尺寸 + 层高",
    "size_only": "单主体：只看尺寸",
    "height_only": "单主体：只看层高",
    "multi_stack": "多主体：堆叠层数",
}


st.set_page_config(
    page_title="乐高检测与识别应用",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)


def render_summary_chips(items):
    if not items:
        st.markdown(build_empty_state("当前没有可展示的识别摘要", "请重新上传图片并执行一次检测。"), unsafe_allow_html=True)
        return
    cards = []
    for item in items:
        if "：" in item:
            label, value = item.split("：", 1)
        else:
            label, value = "结果", item
        cards.append(
            {
                "label": label,
                "value": value,
                "tone": "violet" if "置信度" in label else "blue",
            }
        )
    st.markdown(build_stat_grid(cards), unsafe_allow_html=True)


def main() -> None:
    model_info = build_stage3_model_info()
    about_context = build_about_context()
    stage1 = about_context["stage1"]
    stage2 = about_context["stage2"]

    st.markdown(
        """
        <div class="hero-card">
          <p style="margin:0; letter-spacing:0.08em; text-transform:uppercase; opacity:0.84; font-weight:700;">Stage 3 / Streamlit Application</p>
          <h1 style="margin:0.35rem 0 0.7rem 0;">乐高检测与识别应用</h1>
          <p style="margin:0;">首页保留几何检测能力，分类模型、示例体验和阶段总览已迁移到 Streamlit 多页面结构中。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        build_badge_row(
            [
                {"label": "多页面导航", "tone": "blue"},
                {"label": "几何检测", "tone": "violet"},
                {"label": "7 类分类模型", "tone": "success"},
                {"label": "课堂演示友好", "tone": "slate"},
            ]
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.success("当前已切换为 Streamlit 多页面应用。")
    st.sidebar.caption("左侧侧边栏可切换到分类识别、示例体验和关于阶段页面。")
    st.sidebar.metric("当前模型准确率", f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录")
    st.sidebar.metric("当前类别数", model_info["class_count"])
    st.sidebar.metric("阶段 1 图片数", about_context["stage1"]["total_images"])

    st.markdown(
        build_stat_grid(
            [
                {"label": "应用框架", "value": "Streamlit", "note": "多页面结构", "tone": "blue"},
                {
                    "label": "当前分类模型",
                    "value": model_info["checkpoint_name"],
                    "note": f"设备 {model_info['device']}",
                    "tone": "violet",
                },
                {
                    "label": "测试准确率",
                    "value": f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录",
                    "note": f"Loss {model_info['loss']}" if model_info["loss"] is not None else "评估文件未记录",
                    "tone": "success",
                },
                {
                    "label": "当前类别数",
                    "value": model_info["class_count"],
                    "note": "默认接入 7 类 Brick",
                    "tone": "slate",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    overview_tab, geometry_tab = st.tabs(["项目概览", "几何检测"])

    with overview_tab:
        left_col, right_col = st.columns([1.1, 0.9], gap="large")
        with left_col:
            st.markdown(
                build_panel_card(
                    "当前阶段能力",
                    """
                    <ul class="feature-list">
                      <li>首页继续保留单主体尺寸/层高检测和多主体堆叠检测。</li>
                      <li><span class="inline-chip">Classification</span> 页面接入第二阶段 7 类 CNN 模型做图片识别。</li>
                      <li><span class="inline-chip">Examples</span> 页面直接展示测试集示例和概率分布图。</li>
                      <li><span class="inline-chip">About</span> 页面汇总第一、二、三阶段关键结果，方便展示。</li>
                    </ul>
                    """,
                    eyebrow="Overview",
                    footer_html=build_badge_row(
                        [
                            {"label": "检测 + 识别", "tone": "blue"},
                            {"label": "示例 + 汇总", "tone": "success"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )

        with right_col:
            st.markdown(
                build_panel_card(
                    "模型与数据快照",
                    build_key_value_list(
                        [
                            ("当前 checkpoint", model_info["checkpoint_name"]),
                            ("数据根目录", model_info["data_root"]),
                            ("自动裁剪", "开启" if model_info["auto_crop"] else "关闭"),
                            ("阶段 1 总图片", stage1["total_images"]),
                        ]
                    ),
                    eyebrow="Live Snapshot",
                    footer_html=build_badge_row(
                        [
                            {"label": f"Train {stage1['split_totals']['train']}", "tone": "blue"},
                            {"label": f"Val {stage1['split_totals']['val']}", "tone": "slate"},
                            {"label": f"Test {stage1['split_totals']['test']}", "tone": "success"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )

        quick_col, stage_col = st.columns([0.95, 1.05], gap="large")
        with quick_col:
            st.markdown(
                build_panel_card(
                    "运行方式",
                    (
                        "<div class='command-block'>streamlit run app.py</div>"
                        "<p class='muted-note'>分类识别、示例体验和关于阶段页面都可以通过左侧侧边栏直接切换。</p>"
                    ),
                    eyebrow="Quick Start",
                    tone="success",
                ),
                unsafe_allow_html=True,
            )
        with stage_col:
            st.markdown(
                build_panel_card(
                    "阶段摘要",
                    build_key_value_list(
                        [
                            ("阶段 1 划分", f"{stage1['split_totals']['train']} / {stage1['split_totals']['val']} / {stage1['split_totals']['test']}"),
                            ("阶段 2 划分", f"{stage2['split_totals']['train']} / {stage2['split_totals']['val']} / {stage2['split_totals']['test']}"),
                            ("格式分布", str(stage1["format_distribution"])),
                            ("当前设备", model_info["device"]),
                        ]
                    ),
                    eyebrow="Stage Summary",
                    tone="violet",
                ),
                unsafe_allow_html=True,
            )

    with geometry_tab:
        intro_col, mode_col = st.columns([1.05, 0.95], gap="large")
        with intro_col:
            st.markdown(
                build_panel_card(
                    "图片几何检测",
                    "<p class='muted-note'>这个页面保留原来的几何检测能力，适合展示尺寸、层高和多主体堆叠分析结果。</p>",
                    eyebrow="Geometry",
                    footer_html=build_badge_row(
                        [
                            {"label": "尺寸检测", "tone": "blue"},
                            {"label": "层高检测", "tone": "violet"},
                            {"label": "多主体堆叠", "tone": "success"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )
        with mode_col:
            st.markdown(
                build_panel_card(
                    "检测模式说明",
                    build_key_value_list(
                        [
                            ("single", "同时输出尺寸和层高"),
                            ("size_only", "只关注平面尺寸"),
                            ("height_only", "只判断层高"),
                            ("multi_stack", "识别多主体堆叠层数"),
                        ]
                    ),
                    eyebrow="Modes",
                    tone="slate",
                ),
                unsafe_allow_html=True,
            )

        with st.form("geometry_detection_form"):
            upload_col, mode_col = st.columns([1.05, 0.95], gap="large")
            with upload_col:
                uploaded_file = st.file_uploader(
                    "上传待检测图片",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    help="支持真实乐高照片或实验演示图片。",
                )
            with mode_col:
                selected_mode = st.selectbox(
                    "选择检测模式",
                    options=list(DETECT_MODE_OPTIONS.keys()),
                    format_func=lambda key: DETECT_MODE_OPTIONS[key],
                )
            submitted = st.form_submit_button("开始检测", use_container_width=True)

        if submitted:
            if uploaded_file is None:
                st.warning("请先上传一张图片。")
            else:
                try:
                    image = image_bytes_to_pil(uploaded_file.getvalue())
                    original_bgr = pil_to_bgr(image)
                    result = run_geometry_detection(original_bgr, selected_mode)

                    original_col, annotated_col = st.columns(2, gap="large")
                    with original_col:
                        st.markdown("#### 原图")
                        st.image(image, use_container_width=True)
                    with annotated_col:
                        st.markdown("#### 标注图")
                        st.image(bgr_to_rgb(result["annotated_image"]), use_container_width=True)

                    st.markdown("#### 识别摘要")
                    render_summary_chips(result["summary"])

                    if result["objects"]:
                        st.markdown("#### 对象明细")
                        st.markdown(
                            build_stat_grid(
                                [
                                    {
                                        "label": item["name"],
                                        "value": item["shape"],
                                        "note": f"置信度 {item['confidence']}",
                                        "tone": "success",
                                    }
                                    for item in result["objects"]
                                ]
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            build_panel_card(
                                "检测说明",
                                "<p class='muted-note'>当前模式主要返回摘要信息；如果没有对象明细，这是正常结果。</p>",
                                eyebrow="Result Note",
                                tone="slate",
                            ),
                            unsafe_allow_html=True,
                        )
                except Exception as exc:
                    st.error(str(exc))


main()
