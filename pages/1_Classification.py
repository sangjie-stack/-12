from datetime import datetime

import streamlit as st

from utils.stage3_streamlit import (
    APP_CSS,
    MAX_HISTORY_ITEMS,
    build_badge_row,
    build_empty_state,
    build_key_value_list,
    build_panel_card,
    build_probability_figure,
    build_stage3_model_info,
    build_stat_grid,
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
    st.markdown(
        build_badge_row(
            [
                {"label": "批量上传", "tone": "blue"},
                {"label": "拍照识别", "tone": "success"},
                {"label": "识别历史", "tone": "violet"},
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        build_stat_grid(
            [
                {"label": "当前模型", "value": model_info["checkpoint_name"], "note": "应用默认 checkpoint", "tone": "blue"},
                {
                    "label": "测试准确率",
                    "value": f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录",
                    "note": f"Loss {model_info['loss']}" if model_info["loss"] is not None else "未记录",
                    "tone": "success",
                },
                {"label": "推理设备", "value": model_info["device"], "note": "首次识别时初始化", "tone": "slate"},
                {"label": "类别数", "value": model_info["class_count"], "note": "当前为 7 类 Brick", "tone": "violet"},
            ]
        ),
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.05, 0.95], gap="large")
    with left_col:
        st.markdown(
            build_panel_card(
                "上传或拍照",
                "<p class='muted-note'>支持一次上传多张图片，也支持移动端直接拍照。点击开始识别后，会统一走同一条模型推理链路。</p>",
                eyebrow="Input",
            ),
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "批量选择图片",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            help="可以一次识别多张图片。",
        )
        camera_file = st.camera_input("拍照识别", help="移动端可直接调用摄像头。")
        run_clicked = st.button("开始识别", type="primary", use_container_width=True)

        selected_count = len(uploaded_files or []) + (1 if camera_file is not None else 0)
        if selected_count:
            st.markdown(
                build_badge_row(
                    [
                        {"label": f"待识别 {selected_count} 张", "tone": "success"},
                        {"label": "结果会自动写入历史", "tone": "slate"},
                    ]
                ),
                unsafe_allow_html=True,
            )

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
        st.markdown(
            build_panel_card(
                "模型配置",
                build_key_value_list(
                    [
                        ("数据根目录", model_info["data_root"]),
                        ("推理设备", model_info["device"]),
                        ("自动裁剪", "开启" if model_info["auto_crop"] else "关闭"),
                        ("图像尺寸", model_info["image_size"]),
                        ("类别列表", " / ".join(model_info["class_names"])),
                    ]
                ),
                eyebrow="Inference",
                footer_html=build_badge_row(
                    [
                        {"label": "统一 checkpoint 加载", "tone": "blue"},
                        {"label": "Top-K 概率输出", "tone": "violet"},
                    ]
                ),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_panel_card(
                "识别说明",
                """
                <ul class="feature-list">
                  <li>首次点击开始识别时，页面会自动初始化模型。</li>
                  <li>同一次识别支持图片上传和拍照混合输入。</li>
                  <li>结果区会同时显示原图、Top-1 预测和概率分布图。</li>
                </ul>
                """,
                eyebrow="Workflow",
                tone="slate",
            ),
            unsafe_allow_html=True,
        )

    st.subheader("识别结果")
    results = st.session_state.get("classifier_results", [])
    if not results:
        st.markdown(
            build_empty_state("结果区已就绪", "上传图片或拍照后，这里会显示预测类别、置信度和 Plotly 概率分布图。"),
            unsafe_allow_html=True,
        )
    else:
        max_confidence = max(item["confidence_pct"] for item in results)
        st.markdown(
            build_stat_grid(
                [
                    {"label": "本次识别数量", "value": len(results), "note": "支持批量处理", "tone": "blue"},
                    {"label": "最高置信度", "value": f"{max_confidence:.2f}%", "note": "当前批次最佳", "tone": "success"},
                ]
            ),
            unsafe_allow_html=True,
        )
        for item in results:
            secondary_label = item["top_probabilities"][1]["class_name"] if len(item["top_probabilities"]) > 1 else "无"
            st.markdown(
                build_panel_card(
                    item["filename"],
                    build_key_value_list(
                        [
                            ("Top-1 预测", item["predicted_class"]),
                            ("Top-1 置信度", f"{item['confidence_pct']}%"),
                            ("输入尺寸", f"{item['image_size'][0]} x {item['image_size'][1]}"),
                            ("次高概率类别", secondary_label),
                        ]
                    ),
                    eyebrow="Prediction",
                    footer_html=build_badge_row(
                        [
                            {"label": f"预测 {item['predicted_class']}", "tone": "blue"},
                            {"label": f"置信度 {item['confidence_pct']}%", "tone": "success"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )
            card_col, chart_col = st.columns([0.9, 1.1], gap="large")
            with card_col:
                st.image(item["image_bytes"], caption=item["filename"], use_container_width=True)
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
        st.markdown(build_empty_state("当前还没有识别历史", "完成一次识别后，最近记录会显示在这里。"), unsafe_allow_html=True)
    else:
        history_cards = ["<div class='history-stack'>"]
        for entry in st.session_state["classifier_history"]:
            badge_html = build_badge_row(
                [
                    {"label": f"预测 {entry['predicted_class']}", "tone": "blue"},
                    {"label": f"{entry['confidence_pct']}%", "tone": "success"},
                ]
            )
            history_cards.append(
                (
                    "<div class='history-card'>"
                    f"<strong>{entry['filename']}</strong>"
                    f"<div>{badge_html}</div>"
                    f"<span class='muted-note'>{entry['timestamp']}</span>"
                    "</div>"
                )
            )
        history_cards.append("</div>")
        st.markdown("".join(history_cards), unsafe_allow_html=True)


main()
