import streamlit as st

from utils.stage3_streamlit import (
    APP_CSS,
    build_badge_row,
    build_empty_state,
    build_example_cards,
    build_key_value_list,
    build_panel_card,
    build_probability_figure,
    build_section_header,
    build_stage3_model_info,
    build_stat_grid,
)


st.markdown(APP_CSS, unsafe_allow_html=True)


def main() -> None:
    model_info = build_stage3_model_info()
    examples = build_example_cards()
    correct_count = sum(1 for item in examples if item.get("is_correct"))

    st.markdown(
        """
        <div class="hero-card">
          <p style="margin:0; letter-spacing:0.08em; text-transform:uppercase; opacity:0.84; font-weight:700;">Streamlit / Examples</p>
          <h2 style="margin:0.35rem 0 0.7rem 0;">示例体验</h2>
          <p style="margin:0;">自动抽取第二阶段测试集样本，直接展示当前应用模型的预测结果，适合阶段汇报和课堂演示。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        build_badge_row(
            [
                {"label": "测试集样例直出", "tone": "blue"},
                {"label": "适合课堂演示", "tone": "success"},
                {"label": "Top-K 概率可视化", "tone": "violet"},
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        build_stat_grid(
            [
                {"label": "示例数量", "value": len(examples), "note": "默认每类 1 张", "tone": "blue"},
                {"label": "预测正确", "value": correct_count, "note": "以当前示例集统计", "tone": "success"},
                {
                    "label": "当前模型",
                    "value": model_info["checkpoint_name"],
                    "note": f"测试准确率 {model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "测试准确率未记录",
                    "tone": "violet",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    if not examples:
        st.markdown(
            build_empty_state("当前没有可展示的测试样例", "请先确认 data/splits_stage2_raw/test 和 runs/stage3/verification_summary.json 已准备好。"),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        build_section_header(
            "示例结果列表",
            "每个样例统一展示原图、期望类别、模型预测和概率分布，便于直接用于课堂汇报。",
            eyebrow="Examples",
        ),
        unsafe_allow_html=True,
    )

    for index, item in enumerate(examples, start=1):
        is_correct = item["predicted_class"] == item["expected_class"]
        tone = "success" if is_correct else "warn"
        st.markdown(
            build_section_header(
                f"样例 {index} · {item['filename']}",
                f"期望类别 {item['expected_class']}，模型预测 {item['predicted_class']}，Top-1 置信度 {item['confidence_pct']}%。",
                eyebrow="Example Prediction",
            ),
            unsafe_allow_html=True,
        )
        left_col, right_col = st.columns([0.92, 1.08], gap="large")
        with left_col:
            st.image(item["image_bytes"], caption=item["filename"], use_container_width=True)
        with right_col:
            st.markdown(
                build_panel_card(
                    "样例摘要",
                    build_key_value_list(
                        [
                            ("期望类别", item["expected_class"]),
                            ("模型预测", item["predicted_class"]),
                            ("Top-1 置信度", f"{item['confidence_pct']}%"),
                            ("样本路径", item["image_path"]),
                        ]
                    ),
                    eyebrow="Summary",
                    footer_html=build_badge_row(
                        [
                            {"label": "预测正确" if is_correct else "需要关注", "tone": tone},
                            {"label": f"期望 {item['expected_class']}", "tone": "slate"},
                            {"label": f"预测 {item['predicted_class']}", "tone": tone},
                        ]
                    ),
                    tone=tone,
                ),
                unsafe_allow_html=True,
            )
            figure = build_probability_figure(item["top_probabilities"], title=f"{item['filename']} 概率分布")
            st.plotly_chart(figure, use_container_width=True)
        if index < len(examples):
            st.divider()


main()
