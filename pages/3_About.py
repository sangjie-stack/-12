import streamlit as st

from utils.stage3_streamlit import (
    APP_CSS,
    build_about_context,
    build_badge_row,
    build_key_value_list,
    build_panel_card,
    build_section_header,
    build_stat_grid,
)


st.markdown(APP_CSS, unsafe_allow_html=True)


def main() -> None:
    context = build_about_context()
    stage1 = context["stage1"]
    stage2 = context["stage2"]
    stage3 = context["stage3"]
    accuracy_text = "未记录" if stage2["accuracy_pct"] is None else f"{stage2['accuracy_pct']}%"
    class_count = stage2.get("class_count", len(stage2.get("class_names", [])))
    model_scope_text = (
        "当前已自动接入 Brick+Plate 联合模型，类别列表和示例页面会随 checkpoint 一起更新。"
        if class_count > 7
        else "当前默认接入的是 7 类 Brick 主力模型；已预留 Brick+Plate 联合模型的配置、目录和推理切换链路，训练出联合 checkpoint 后会自动切换。"
    )

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

    st.markdown(
        build_badge_row(
            [
                {"label": "阶段结果汇总", "tone": "blue"},
                {"label": "适合展示", "tone": "success"},
                {"label": "多页面应用状态", "tone": "violet"},
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        build_stat_grid(
            [
                {"label": "阶段 1 类别数", "value": stage1["class_count"], "note": "当前数据集类别总数", "tone": "blue"},
                {"label": "阶段 1 总图像", "value": stage1["total_images"], "note": "quality_report.json", "tone": "slate"},
                {
                    "label": "阶段 2 当前准确率",
                    "value": f"{stage2['accuracy_pct']}%" if stage2["accuracy_pct"] is not None else "未记录",
                    "note": f"Loss {stage2['loss']}" if stage2["loss"] is not None else "评估文件未记录",
                    "tone": "success",
                },
                {"label": "阶段 3 页面数", "value": stage3["page_count"], "note": "首页 + 3 个子页面", "tone": "violet"},
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        build_section_header(
            "阶段成果概览",
            "把数据准备和模型训练分成左右两块，信息密度更均衡，也更方便对照报告内容。",
            eyebrow="Stages",
        ),
        unsafe_allow_html=True,
    )

    first_col, second_col = st.columns(2, gap="large")
    with first_col:
        st.markdown(
            build_section_header(
                "第一阶段：数据准备",
                "完成样本整理、类别统计和训练/验证/测试划分。",
                eyebrow="Stage 1",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_panel_card(
                "数据集状态",
                build_key_value_list(
                    [
                        ("类别数", stage1["class_count"]),
                        ("总图片", stage1["total_images"]),
                        ("Train / Val / Test", f"{stage1['split_totals']['train']} / {stage1['split_totals']['val']} / {stage1['split_totals']['test']}"),
                        ("格式分布", stage1["format_distribution"]),
                    ]
                ),
                eyebrow="Dataset",
                tone="blue",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_panel_card(
                "类别分布",
                build_key_value_list([(name, count) for name, count in stage1["per_class_counts"].items()]),
                eyebrow="Per Class",
                tone="slate",
            ),
            unsafe_allow_html=True,
        )

    with second_col:
        st.markdown(
            build_section_header(
                "第二阶段：模型训练",
                "完成 CNN 训练、最优权重保存和测试集评估。",
                eyebrow="Stage 2",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_panel_card(
                "当前模型状态",
                build_key_value_list(
                    [
                        ("当前 checkpoint", stage2["checkpoint_name"]),
                        ("推理设备", stage2["device"]),
                        ("输入尺寸", stage2["image_size"]),
                        ("自动裁剪", "开启" if stage2["auto_crop"] else "关闭"),
                        ("测试准确率", accuracy_text),
                    ]
                ),
                eyebrow="Model",
                tone="success",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_panel_card(
                "训练集与类别配置",
                build_key_value_list(
                    [
                        ("类别列表", " / ".join(stage2["class_names"])),
                        ("Train / Val / Test", f"{stage2['split_totals']['train']} / {stage2['split_totals']['val']} / {stage2['split_totals']['test']}"),
                    ]
                ),
                eyebrow="Training Setup",
                tone="violet",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        build_section_header(
            "第三阶段：Streamlit 应用",
            "页面能力单独成区展示，避免和前两阶段的训练信息挤在同一行。",
            eyebrow="Stage 3",
        ),
        unsafe_allow_html=True,
    )
    feature_cols = st.columns(2, gap="large")
    for index, feature in enumerate(stage3["features"]):
        with feature_cols[index % 2]:
            st.markdown(
                build_panel_card(
                    feature["title"],
                    f"<p class='muted-note'>{feature['body']}</p>",
                    eyebrow="Stage 3",
                    tone="violet",
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        build_section_header(
            "目标对照与运行状态",
            "把完成项和当前应用状态放在最后，便于直接回答“第三阶段是否完成”。",
            eyebrow="Checklist",
        ),
        unsafe_allow_html=True,
    )
    checklist_col, status_col = st.columns([1.15, 0.85], gap="large")
    with checklist_col:
        st.markdown(
            build_panel_card(
                "当前目标对照",
                f"""
                <ol class="feature-list">
                  <li>已改为 Streamlit 多页面结构。</li>
                  <li>已保留并复用深度学习模型完整推理流程。</li>
                  <li>已使用 Plotly 展示类别概率分布图。</li>
                  <li>已提供关于页面、边界提示和阶段级验证脚本。</li>
                  <li>{model_scope_text}</li>
                </ol>
                """,
                eyebrow="Checklist",
                tone="blue",
            ),
            unsafe_allow_html=True,
        )
    with status_col:
        st.markdown(
            build_panel_card(
                "应用状态",
                build_key_value_list(
                    [
                        ("页面结构", "首页 + 3 个子页面"),
                        ("可视化", "Plotly 概率分布图"),
                        ("输入方式", "上传图片 / 摄像头拍照"),
                        ("默认模型", stage2["checkpoint_name"]),
                        ("当前准确率", accuracy_text),
                    ]
                ),
                eyebrow="Runtime",
                tone="success",
            ),
            unsafe_allow_html=True,
        )


main()
