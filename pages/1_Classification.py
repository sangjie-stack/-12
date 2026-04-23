from datetime import datetime

import streamlit as st

from utils.stage3_streamlit import (
    APP_CSS,
    MAX_HISTORY_ITEMS,
    build_badge_row,
    build_empty_state,
    build_key_value_list,
    build_mac_window_card,
    build_panel_card,
    build_probability_figure,
    build_section_header,
    build_stage3_model_info,
    build_stat_grid,
    classify_uploaded_batch,
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

    st.markdown(
        build_section_header(
            "识别工作区",
            "把拍照、上传、模式选择和开始识别整理到同一块区域，操作顺序会更清晰。",
            eyebrow="Workspace",
        ),
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    uploaded_files = []
    camera_file = None
    camera_quick_clicked = False
    shadow_tta_enabled = True

    with left_col:
        camera_tab, upload_tab = st.tabs(["拍照识别", "批量上传"])

        with camera_tab:
            st.markdown(
                build_panel_card(
                    "拍照识别",
                    "<p class='muted-note'>使用浏览器本地摄像头直接拍照，适合现场测试单个乐高零件。建议主体居中、边缘完整、背景尽量简洁。</p>",
                    eyebrow="Camera",
                    footer_html=build_badge_row(
                        [
                            {"label": "本地摄像头", "tone": "blue"},
                            {"label": "点击拍照", "tone": "success"},
                            {"label": "现场测试", "tone": "violet"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )
            camera_file = st.camera_input("点击拍照", help="适合本地摄像头或移动端直接拍照。")
            if camera_file is not None:
                st.success("拍照完成，可以直接识别这张照片。")
            else:
                st.info("打开摄像头后拍一张照片，再点击识别按钮。")
            camera_quick_clicked = st.button(
                "只识别这张拍照图片",
                type="primary",
                use_container_width=True,
                disabled=camera_file is None,
            )

        with upload_tab:
            st.markdown(
                build_panel_card(
                    "批量上传",
                    "<p class='muted-note'>支持一次导入多张图片，也可以和拍照图片一起走同一条识别链路。</p>",
                    eyebrow="Upload",
                    footer_html=build_badge_row(
                        [
                            {"label": "JPG / PNG / WEBP", "tone": "blue"},
                            {"label": "支持多张", "tone": "slate"},
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "批量选择图片",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                accept_multiple_files=True,
                help="可以一次识别多张图片。",
            )
            if uploaded_files:
                st.markdown(
                    build_badge_row(
                        [
                            {"label": f"已选择 {len(uploaded_files)} 张", "tone": "success"},
                            {"label": "支持批量处理", "tone": "slate"},
                        ]
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("上传区已就绪，你可以一次导入多张测试图片。")

        selected_count = len(uploaded_files or []) + (1 if camera_file is not None else 0)

        st.markdown(
            build_panel_card(
                "识别控制",
                "<p class='muted-note'>模式选择和总启动按钮集中在这里，避免页面操作顺序过于分散。</p>",
                eyebrow="Control",
                tone="slate",
            ),
            unsafe_allow_html=True,
        )
        shadow_tta_enabled = st.checkbox(
            "开启抗阴影双路融合（推荐）",
            value=True,
            help="同一张图同时执行标准推理和抗阴影预处理推理，再按 50/50 融合概率。",
        )
        if shadow_tta_enabled:
            reference_text = (
                f"参考测试准确率 {model_info['shadow_tta_accuracy_pct']}%"
                if model_info.get("shadow_tta_accuracy_pct") is not None
                else "已启用抗阴影双路融合"
            )
            st.markdown(
                build_badge_row(
                    [
                        {"label": "推荐模式", "tone": "violet"},
                        {"label": "原图 + 抗阴影图 50/50 融合", "tone": "blue"},
                        {"label": reference_text, "tone": "success"},
                    ]
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                build_badge_row(
                    [
                        {"label": "标准模式", "tone": "slate"},
                        {"label": "单路模型推理", "tone": "blue"},
                    ]
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            build_stat_grid(
                [
                    {"label": "当前待识别", "value": selected_count, "note": "拍照图 + 上传图合计", "tone": "blue"},
                    {"label": "历史记录", "value": len(st.session_state["classifier_history"]), "note": "最多保留最近 12 条", "tone": "violet"},
                ]
            ),
            unsafe_allow_html=True,
        )

        run_clicked = st.button(
            "开始识别当前全部输入",
            type="primary",
            use_container_width=True,
            disabled=selected_count == 0,
        )

        if run_clicked or camera_quick_clicked:
            candidates = []
            if camera_file is not None:
                candidates.append((camera_file.name or "camera_capture.jpg", camera_file.getvalue()))
            if not camera_quick_clicked:
                for file in uploaded_files or []:
                    candidates.append((file.name, file.getvalue()))

            if not candidates:
                st.warning("请先上传图片或拍一张照片。")
            else:
                with st.spinner("正在初始化模型并执行识别..."):
                    results = classify_uploaded_batch(candidates, use_shadow_tta=shadow_tta_enabled)
                history_entries = [
                    {
                        "filename": item["filename"],
                        "predicted_class": item["predicted_class"],
                        "confidence_pct": item["confidence_pct"],
                        "inference_mode": item["inference_mode"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    for item in results
                ]
                st.session_state["classifier_results"] = results
                push_history(history_entries)

    with right_col:
        st.markdown(
            build_section_header(
                "模型与说明",
                "把当前模型配置、输入尺寸和使用说明统一放在右侧，阅读会更顺。",
                eyebrow="Overview",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_mac_window_card(
                "classification-model-window",
                "模型配置",
                build_key_value_list(
                    [
                        ("数据根目录", model_info["data_root"]),
                        ("推理设备", model_info["device"]),
                        ("自动裁剪", "开启" if model_info["auto_crop"] else "关闭"),
                        ("抗阴影双路融合", f"可选，参考 {model_info['shadow_tta_accuracy_pct']}%" if model_info.get("shadow_tta_available") else "未启用"),
                        ("图像尺寸", model_info["image_size"]),
                        ("类别列表", " / ".join(model_info["class_names"])),
                    ]
                ),
                eyebrow="Inference",
                subtitle="这是分类页真实使用的模型信息窗口，红点关闭、黄点最小化、绿点放大都已经接入。",
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
            build_mac_window_card(
                "classification-workflow-window",
                "识别说明",
                """
                <ul class="feature-list">
                  <li>首次点击开始识别时，页面会自动初始化模型。</li>
                  <li>同一次识别支持图片上传和拍照混合输入。</li>
                  <li>结果区会同时显示原图、Top-1 预测和概率分布图。</li>
                </ul>
                """,
                eyebrow="Workflow",
                subtitle="不需要时可以先关闭或最小化，页面右侧会更干净。",
                footer_html=build_badge_row(
                    [
                        {"label": "macOS 收起动效", "tone": "slate"},
                        {"label": "Dock 样式恢复", "tone": "blue"},
                    ]
                ),
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        build_section_header(
            "识别结果",
            "按图片逐条展示原图、预测摘要和概率分布图，减少上下跳读。",
            eyebrow="Results",
        ),
        unsafe_allow_html=True,
    )
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
        for index, item in enumerate(results, start=1):
            secondary_label = item["top_probabilities"][1]["class_name"] if len(item["top_probabilities"]) > 1 else "无"
            st.markdown(
                build_section_header(
                    f"结果 {index} · {item['filename']}",
                    f"Top-1 预测为 {item['predicted_class']}，当前置信度 {item['confidence_pct']}%。",
                    eyebrow="Prediction",
                ),
                unsafe_allow_html=True,
            )
            image_col, detail_col = st.columns([0.92, 1.08], gap="large")
            with image_col:
                st.image(item["image_bytes"], caption=item["filename"], use_container_width=True)
            with detail_col:
                st.markdown(
                    build_panel_card(
                        "预测摘要",
                        build_key_value_list(
                            [
                                ("Top-1 预测", item["predicted_class"]),
                                ("Top-1 置信度", f"{item['confidence_pct']}%"),
                                ("推理模式", item.get("inference_mode", "标准单路推理")),
                                ("输入尺寸", f"{item['image_size'][0]} x {item['image_size'][1]}"),
                                ("次高概率类别", secondary_label),
                            ]
                        ),
                        eyebrow="Summary",
                        footer_html=build_badge_row(
                            [
                                {"label": f"预测 {item['predicted_class']}", "tone": "blue"},
                                {"label": f"置信度 {item['confidence_pct']}%", "tone": "success"},
                                {"label": item.get("inference_mode", "标准单路推理"), "tone": "violet"},
                            ]
                        ),
                    ),
                    unsafe_allow_html=True,
                )
                figure = build_probability_figure(item["top_probabilities"], title=f"{item['filename']} 概率分布")
                st.plotly_chart(figure, use_container_width=True)
            if index < len(results):
                st.divider()

    st.markdown(
        build_section_header(
            "识别历史",
            "最近的识别结果会按时间顺序保留，方便课堂演示时快速回看。",
            eyebrow="History",
        ),
        unsafe_allow_html=True,
    )
    history_header_col, history_action_col = st.columns([0.8, 0.2])
    with history_header_col:
        st.caption("最近记录会优先显示在最上面。")
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
                    {"label": entry.get("inference_mode", "标准单路推理"), "tone": "violet"},
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
        st.markdown(
            build_mac_window_card(
                "classification-history-window",
                "识别历史记录",
                "".join(history_cards),
                eyebrow="History",
                subtitle="历史记录也可以像窗口一样关闭、最小化或放大，演示时不会占满页面。",
                footer_html=build_badge_row(
                    [
                        {"label": f"当前 {len(st.session_state['classifier_history'])} 条", "tone": "success"},
                        {"label": "支持一键恢复", "tone": "violet"},
                    ]
                ),
            ),
            unsafe_allow_html=True,
        )


main()
