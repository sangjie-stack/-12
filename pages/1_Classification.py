from datetime import datetime

import streamlit as st

from utils.stage3_streamlit import (
    APP_CSS,
    MAX_HISTORY_ITEMS,
    build_probability_figure,
    build_stage3_model_info,
    classify_uploaded_bytes,
)


st.markdown(APP_CSS, unsafe_allow_html=True)

if "classifier_history" not in st.session_state:
    st.session_state["classifier_history"] = []
if "classifier_results" not in st.session_state:
    st.session_state["classifier_results"] = []


def push_history(entries):
    current = list(st.session_state.get("classifier_history", []))
    st.session_state["classifier_history"] = (entries + current)[:MAX_HISTORY_ITEMS]


def main() -> None:
    model_info = build_stage3_model_info()

    st.markdown(
        """
        <div class="hero-card">
          <p style="margin:0; letter-spacing:0.08em; text-transform:uppercase; opacity:0.84; font-weight:700;">Streamlit / Classification</p>
          <h2 style="margin:0.35rem 0 0.7rem 0;">乐高分类识别</h2>
          <p style="margin:0;">调用第二阶段训练好的 7 类 Brick 分类模型，支持文件上传、移动端拍照入口、Plotly 概率分布图和识别历史。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("页面已经秒开；分类模型会在你第一次点击“开始识别”时自动初始化。")

    info_columns = st.columns(4)
    info_columns[0].metric("当前模型", model_info["checkpoint_name"])
    info_columns[1].metric("测试准确率", f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录")
    info_columns[2].metric("推理设备", model_info["device"])
    info_columns[3].metric("类别数", model_info["class_count"])

    left_col, right_col = st.columns([1.05, 0.95], gap="large")
    with left_col:
        st.subheader("上传或拍照")
        uploaded_files = st.file_uploader(
            "批量选择图片",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            help="可以一次识别多张图片。",
        )
        camera_file = st.camera_input("拍照识别", help="移动端可直接调用摄像头。")
        run_clicked = st.button("开始识别", type="primary", use_container_width=True)

        if run_clicked:
            candidates = []
            if camera_file is not None:
                candidates.append((camera_file.name or "camera_capture.jpg", camera_file.getvalue()))
            for file in uploaded_files or []:
                candidates.append((file.name, file.getvalue()))

            if not candidates:
                st.warning("请先上传图片或拍一张照片。")
            else:
                results = []
                history_entries = []
                with st.spinner("正在初始化模型并执行识别..."):
                    for filename, content in candidates:
                        item = classify_uploaded_bytes(content, filename)
                        results.append(item)
                        history_entries.append(
                            {
                                "filename": item["filename"],
                                "predicted_class": item["predicted_class"],
                                "confidence_pct": item["confidence_pct"],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                st.session_state["classifier_results"] = results
                push_history(history_entries)

    with right_col:
        st.subheader("模型配置")
        st.caption(model_info["checkpoint_path"])
        st.markdown(
            f"""
            - 数据根目录：`{model_info['data_root']}`
            - 推理设备：`{model_info['device']}`
            - 自动裁剪：`{'开启' if model_info['auto_crop'] else '关闭'}`
            - 图像尺寸：`{model_info['image_size']}`
            - 类别列表：`{' / '.join(model_info['class_names'])}`
            """
        )

    st.subheader("识别结果")
    results = st.session_state.get("classifier_results", [])
    if not results:
        st.info("上传图片后，这里会显示预测类别、置信度和 Plotly 概率分布图。")
    else:
        for item in results:
            card_col, chart_col = st.columns([0.9, 1.1], gap="large")
            with card_col:
                st.image(item["image_bytes"], caption=item["filename"], use_container_width=True)
                st.metric("预测类别", item["predicted_class"], delta=f"{item['confidence_pct']}%")
                st.caption(f"输入尺寸：{item['image_size'][0]} x {item['image_size'][1]}")
            with chart_col:
                figure = build_probability_figure(item["top_probabilities"], title=f"{item['filename']} 概率分布")
                st.plotly_chart(figure, use_container_width=True)

    history_header_col, history_action_col = st.columns([0.8, 0.2])
    with history_header_col:
        st.subheader("识别历史")
    with history_action_col:
        if st.session_state["classifier_history"]:
            if st.button("清空历史", use_container_width=True):
                st.session_state["classifier_history"] = []

    if not st.session_state["classifier_history"]:
        st.caption("当前还没有识别历史。")
    else:
        for entry in st.session_state["classifier_history"]:
            st.markdown(
                (
                    "<div class='soft-card'>"
                    f"<strong>{entry['filename']}</strong><br>"
                    f"预测为 {entry['predicted_class']}，置信度 {entry['confidence_pct']}%<br>"
                    f"<span class='muted-note'>{entry['timestamp']}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


main()
