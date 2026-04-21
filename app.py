import streamlit as st

from utils.stage3_streamlit import (
    APP_CSS,
    bgr_to_rgb,
    build_about_context,
    build_stage3_model_info,
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
        st.info("当前没有可展示的识别摘要。")
        return
    columns = st.columns(min(4, len(items)))
    for index, item in enumerate(items):
        columns[index % len(columns)].markdown(
            f"<div class='soft-card'><strong>{item}</strong></div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    model_info = build_stage3_model_info()
    about_context = build_about_context()

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

    st.sidebar.success("当前已切换为 Streamlit 多页面应用。")
    st.sidebar.caption("左侧侧边栏可切换到分类识别、示例体验和关于阶段页面。")
    st.sidebar.metric("当前模型准确率", f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录")
    st.sidebar.metric("当前类别数", model_info["class_count"])
    st.sidebar.metric("阶段 1 图片数", about_context["stage1"]["total_images"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("应用框架", "Streamlit")
    metric_columns[1].metric("当前分类模型", model_info["checkpoint_name"])
    metric_columns[2].metric("测试准确率", f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录")
    metric_columns[3].metric("类别数", model_info["class_count"])

    overview_tab, geometry_tab = st.tabs(["项目概览", "几何检测"])

    with overview_tab:
        left_col, right_col = st.columns([1.1, 0.9], gap="large")
        with left_col:
            st.subheader("当前阶段能力")
            st.markdown(
                """
                <ul class="feature-list">
                  <li>首页提供单主体尺寸/层高检测和多主体堆叠检测。</li>
                  <li><span class="inline-chip">Classification</span> 页面调用第二阶段 7 类 CNN 模型做图片识别。</li>
                  <li><span class="inline-chip">Examples</span> 页面展示测试集样本和概率分布图。</li>
                  <li><span class="inline-chip">About</span> 页面汇总第一、二、三阶段的关键结果。</li>
                </ul>
                """
                ,
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='muted-note'>现在运行入口已经改为 Streamlit，多页面导航由 Streamlit 侧边栏自动管理。</p>",
                unsafe_allow_html=True,
            )

        with right_col:
            st.subheader("阶段数据摘要")
            stage1 = about_context["stage1"]
            split_totals = stage1["split_totals"]
            stage_columns = st.columns(3)
            stage_columns[0].metric("Train", split_totals["train"])
            stage_columns[1].metric("Val", split_totals["val"])
            stage_columns[2].metric("Test", split_totals["test"])
            st.caption(f"阶段 1 当前总图像数：{stage1['total_images']}，格式分布：{stage1['format_distribution']}")

        st.subheader("运行方式")
        st.markdown("<div class='command-block'>streamlit run app.py</div>", unsafe_allow_html=True)
        st.info("分类识别、示例体验和关于阶段页面可通过左侧边栏直接切换。")

    with geometry_tab:
        st.subheader("图片几何检测")
        st.markdown(
            "<p class='muted-note'>这个页面保留原来的几何检测能力，适合展示尺寸、层高和多主体堆叠分析结果。</p>",
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
                        object_columns = st.columns(min(3, len(result["objects"])))
                        for index, item in enumerate(result["objects"]):
                            object_columns[index % len(object_columns)].markdown(
                                (
                                    "<div class='soft-card'>"
                                    f"<strong>{item['name']}</strong><br>"
                                    f"形态：{item['shape']}<br>"
                                    f"置信度：{item['confidence']}"
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )
                except Exception as exc:
                    st.error(str(exc))


main()
