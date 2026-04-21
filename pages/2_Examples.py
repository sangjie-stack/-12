import streamlit as st

from utils.stage3_streamlit import APP_CSS, build_example_cards, build_probability_figure, build_stage3_model_info


st.markdown(APP_CSS, unsafe_allow_html=True)


def main() -> None:
    model_info = build_stage3_model_info()
    examples = build_example_cards()

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

    metrics = st.columns(3)
    metrics[0].metric("示例数量", len(examples))
    metrics[1].metric("当前模型", model_info["checkpoint_name"])
    metrics[2].metric("测试准确率", f"{model_info['accuracy_pct']}%" if model_info["accuracy_pct"] is not None else "未记录")

    if not examples:
        st.warning("当前没有可展示的测试集样本，请先确认 `data/splits_stage2_raw/test` 已准备好。")
        return

    for item in examples:
        left_col, right_col = st.columns([0.9, 1.1], gap="large")
        with left_col:
            st.image(item["image_bytes"], caption=item["filename"], use_container_width=True)
            st.metric("模型预测", item["predicted_class"], delta=f"{item['confidence_pct']}%")
            st.caption(f"期望类别：{item['expected_class']}")
            st.caption(item["image_path"])
        with right_col:
            figure = build_probability_figure(item["top_probabilities"], title=f"{item['filename']} 概率分布")
            st.plotly_chart(figure, use_container_width=True)
        st.divider()


main()
