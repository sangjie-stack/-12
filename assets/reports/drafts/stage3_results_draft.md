# 第三阶段应用开发与部署草稿

## 1. 阶段衔接说明

当前项目已经把第一阶段的数据准备结果和第二阶段的分类模型正式接入到第三阶段应用层中。第一阶段最新质量检查结果显示，当前 `data/raw` 中共有 `213` 张有效图片、`7` 个 Brick 类别；对应的 `64x64` 预处理和 `train / val / test` 划分也已重新生成。第二阶段则以 `runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth` 作为当前 7 类应用模型，并在 `data/splits_stage2_raw` 上复核得到测试准确率 `0.7000`。

## 2. 第三阶段已实现内容

项目新增了面向应用接入的模型定义与推理封装，补齐了 `model/model_def.py` 和 `model/inference.py`，实现了 checkpoint 加载、类别解析、自动裁剪预处理和 Top-K 概率输出，从而把训练好的 CNN 模型从“实验脚本可用”推进到了“页面可直接调用”的状态。

在前端应用部分，当前项目已经改为 Streamlit 多页面结构，包含：

- 首页 `app.py`
- 分类识别页 `pages/1_Classification.py`
- 示例体验页 `pages/2_Examples.py`
- 阶段说明页 `pages/3_About.py`

其中分类识别页支持单张上传、批量上传和移动端拍照入口，识别后会展示预测类别、置信度和 Plotly 概率分布图，并保留最近识别历史。示例体验页会自动抽取第二阶段测试集样本，直接展示当前模型预测结果，适合阶段汇报时快速演示。关于页面则统一汇总第一、二、三阶段完成情况，便于直接写入总结报告。

## 3. 结果与分析

从当前应用接入效果来看，第一阶段和第二阶段已经不再是孤立成果，而是被第三阶段页面完整消费起来。数据准备结果为训练脚本和应用推理提供了统一目录结构，第二阶段的 7 类模型通过推理模块稳定输出预测结果，第三阶段页面则完成了上传、展示、Plotly 可视化、示例验证和历史记录等交互闭环。

就阶段二复核结果而言，当前激活模型在 `data/splits_stage2_raw/test` 上的准确率为 `70.00%`，其中 `1x3`、`1x4` 等类别识别较稳，但 `2x2`、`2x3` 仍存在提升空间。这说明第三阶段虽然已经完成了“模型接入应用”的目标，但后续仍可以围绕小样本类别继续做数据补充和结构优化，以提升应用端整体稳定性。

## 4. 可直接使用的验证命令

```bash
.venv\Scripts\python.exe utils/auto_prepare_stage1.py
.venv\Scripts\python.exe -m scripts.stage2.evaluate_model --checkpoint runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth --data-root data/splits_stage2_raw --batch-size 8
.venv\Scripts\python.exe -m scripts.stage3.verify_stage3 --output runs/stage3/verification_summary.json
.venv\Scripts\streamlit.exe run app.py
```
