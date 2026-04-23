# Repository Guidelines

## 项目结构与模块组织
本仓库主要包含乐高检测、数据准备、模型训练与 Flask 演示。新增内容时优先放入现有目录，避免随意增加新的顶层文件夹。

- `app.py`：本地网页演示入口。
- `detectors/`：尺寸、层高、多主体堆叠等检测模块，支持命令行调用。
- `model/`：数据集定义、CNN 模型与训练辅助工具。
- `scripts/stage2/`：第二阶段训练、评估、可视化与预测脚本。
- `utils/`：爬取、预处理、质量检查、数据划分等工具。
- `assets/`、`data/`、`runs/`：演示素材、数据集、实验输出与模型权重。
- `templates/`、`pages/`：网页模板与页面相关模块。

## 构建、测试与开发命令
建议先创建虚拟环境，再安装依赖：

- `pip install -r requirements.txt`：安装运行和训练所需依赖。
- `python app.py`：启动 Flask 演示，访问 `http://127.0.0.1:5000`。
- `python -m detectors.lego_size_detector assets/demo_inputs/OIP-C.webp`：执行单图检测冒烟验证。
- `python utils/auto_prepare_stage1.py`：运行第一阶段数据准备全流程。
- `python -m scripts.stage2.train_baseline --epochs 20 --batch-size 16 --learning-rate 0.001`：训练基线分类模型。
- `python -m scripts.stage2.evaluate_model --checkpoint runs/stage2/baseline/best_model.pth`：评估已有模型权重。

## 代码风格与命名规范
仓库以 Python 为主，保持与现有代码一致：

- 统一使用 4 个空格缩进。
- 文件名、函数名、变量名、CLI 参数使用 `snake_case`。
- 模块名应直接表达用途，例如 `lego_height_detector.py`、`check_dataset_quality.py`。
- 目录命名应贴合流程阶段，如 `data/raw`、`data/processed`、`runs/stage2`。
- 当前未配置统一格式化工具，提交前请自行整理导入顺序并保持风格一致。

## 测试指南
目前仓库没有独立的 `tests/` 目录，建议按改动范围做最小可运行验证：

- 检测模块改动后，运行对应的 `python -m detectors...` 命令检查输出。
- 训练相关改动后，优先使用 `scripts/stage2/visualize_batch.py`、`train_baseline.py`、`evaluate_model.py` 验证流程。
- 生成的评估结果、图表和权重应保存在 `runs/stage2/...`，不要提交无用的大体积临时文件。

## 提交与合并请求规范
现有提交历史采用简短的祈使句风格，例如 `Add stage-1 dataset preparation workflow`。

- 提交信息建议以动词开头，如 `Add`、`Update`、`Fix`、`Refactor`。
- 每次提交只聚焦一个功能点或一个流程环节。
- 提交 PR 时应说明改动目的、影响路径、执行过的命令；涉及页面变更时附上截图。
- 如改动依赖数据集、报告或模型权重，请在说明中写明对应位置与验证方式。
