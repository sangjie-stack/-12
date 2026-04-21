import streamlit as st

from utils.stage3_streamlit import APP_CSS, build_about_context


st.markdown(APP_CSS, unsafe_allow_html=True)


def main() -> None:
    context = build_about_context()
    stage1 = context["stage1"]
    stage2 = context["stage2"]
    stage3 = context["stage3"]
    accuracy_text = "未记录" if stage2["accuracy_pct"] is None else f"{stage2['accuracy_pct']}%"

    st.markdown(
        """
        <div class="hero-card">
          <p style="margin:0; letter-spacing:0.08em; text-transform:uppercase; opacity:0.84; font-weight:700;">Streamlit / About</p>
          <h2 style="margin:0.35rem 0 0.7rem 0;">关于阶段完成情况</h2>
          <p style="margin:0;">这里把第一阶段数据准备、第二阶段模型训练和第三阶段 Streamlit 应用接入结果放到同一个页面里，便于直接写入阶段报告和总结报告。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric("阶段 1 类别数", stage1["class_count"])
    metrics[1].metric("阶段 1 总图像", stage1["total_images"])
    metrics[2].metric("阶段 2 当前准确率", f"{stage2['accuracy_pct']}%" if stage2["accuracy_pct"] is not None else "未记录")
    metrics[3].metric("阶段 3 页面数", stage3["page_count"])

    first_col, second_col, third_col = st.columns(3, gap="large")
    with first_col:
        st.subheader("第一阶段：数据准备")
        st.markdown(
            f"""
            - 类别数：`{stage1['class_count']}`
            - 总图片：`{stage1['total_images']}`
            - Train / Val / Test：`{stage1['split_totals']['train']} / {stage1['split_totals']['val']} / {stage1['split_totals']['test']}`
            """
        )
        st.json(stage1["per_class_counts"])

    with second_col:
        st.subheader("第二阶段：模型训练")
        st.markdown(
            f"""
            - 当前 checkpoint：`{stage2['checkpoint_name']}`
            - 推理设备：`{stage2['device']}`
            - 输入尺寸：`{stage2['image_size']}`
            - 自动裁剪：`{'开启' if stage2['auto_crop'] else '关闭'}`
            - 测试准确率：`{accuracy_text}`
            """
        )
        st.json(
            {
                "class_names": stage2["class_names"],
                "split_totals": stage2["split_totals"],
            }
        )

    with third_col:
        st.subheader("第三阶段：Streamlit 应用")
        for feature in stage3["features"]:
            st.markdown(
                (
                    "<div class='soft-card'>"
                    f"<strong>{feature['title']}</strong><br>"
                    f"<span class='muted-note'>{feature['body']}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.subheader("当前目标对照")
    st.markdown(
        """
        1. 已改为 Streamlit 多页面结构。
        2. 已保留并复用深度学习模型完整推理流程。
        3. 已使用 Plotly 展示类别概率分布图。
        4. 已提供关于页面、边界提示和阶段级验证脚本。
        5. 当前默认接入的是 7 类 Brick 模型，后续替换为 Brick+Plate 联合模型时可继续复用同一加载与推理链路。
        """
    )


main()
