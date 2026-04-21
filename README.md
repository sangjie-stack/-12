# LEGO Project

这个项目现在同时包含两部分内容：

1. 乐高检测与网页演示系统
2. 为阶段实验报告补齐的数据准备与项目骨架代码

## 安装依赖

```bash
pip install -r requirements.txt
```

## 检测功能

单图检测：

```bash
python -m detectors.lego_size_detector assets/demo_inputs/OIP-C.webp
```

网页演示：

```bash
streamlit run app.py
```

启动后在浏览器打开 Streamlit 提示的本地地址即可使用，默认通常是 `http://localhost:8501`。

支持的模式：

- 单主体尺寸 + 层高检测
- 单主体只检测尺寸
- 单主体只检测层高
- 多主体堆叠检测
- 多页面分类识别、示例体验与阶段总览

## 目录整理

为了更方便查看和移动文件，当前项目中常用内容已经按用途归类：

```text
assets/
  demo_inputs/      手动检测时使用的示例输入图
  demo_outputs/     检测结果图、合成演示图
  logs/             网页运行日志
  *.md / *.json     报告材料、采集报告、阶段摘要
detectors/
  *.py              乐高尺寸、层高、多主体检测算法
scripts/
  stage2/           第二阶段训练、实验、评估入口脚本
data/
  raw/              网站采集的原始图片
  processed/        Resize + 重命名后的图片
  splits/           train / val / test 划分结果
runs/
  stage2/           第二阶段训练输出、图表和实验记录
```

## 第一阶段报告对应代码

项目已补齐报告中常见的标准目录：

```text
model/
utils/
pages/
data/
assets/
```

其中数据目录推荐结构为：

```text
data/raw/1x1
data/raw/1x2
data/raw/1x3
data/raw/1x4
data/raw/2x2
data/raw/2x3
data/raw/2x4
data/processed
data/splits
```

## 数据准备脚本

0. 从 lemuwu 采集基础砖图片

```bash
python utils/crawl_lemuwu_bricks.py data/raw --limit-per-class 20 --delay 0.8 --clean
```

说明：

- 当前推荐优先使用网站采集图
- 当前爬虫脚本默认抓取 6 类基础砖：`1x1`、`1x2`、`1x4`、`2x2`、`2x3`、`2x4`
- 其中 `1x3` 目前已支持通过手动拍摄图片加入 `data/raw/1x3`
- 使用站点页面中的颜色列表，按可用度优先下载
- 默认低频抓取，并生成 `assets/lemuwu_crawl_report.json`

0b. 自动生成基础 Brick 白底样本图

```bash
python utils/generate_brick_samples.py data/raw --samples-per-class 20 --image-size 320 --clean
```

说明：

- 仅在缺少网站采集图或需要补充测试样本时使用

1. 数据质量检查

```bash
python utils/check_dataset_quality.py data/raw --output data/quality_report.json
```

2. 使用 PIL 批量缩放到 `64x64` 并统一命名

```bash
python utils/resize_and_rename.py data/raw data/processed --size 64 --prefix lego
```

3. 按 `70% / 15% / 15%` 划分训练集、验证集和测试集

```bash
python utils/split_dataset.py data/processed data/splits --clean --report data/splits/split_report.json
```

4. 一键完成第一阶段流程

```bash
python utils/auto_prepare_stage1.py
```

Windows 下也可以直接双击：

```text
run_stage1_pipeline.bat
```

该流程会自动：

- 检查并补齐标准目录
- 默认按当前 `data/raw` 中已有原始图片继续处理
- 可选生成基础 Brick 规格白底样本图作为补充
- 检查 `data/raw` 中的图片质量
- 使用 PIL 缩放到 `64x64` 并统一命名
- 生成 `70% / 15% / 15%` 的数据划分
- 输出阶段摘要 `assets/stage1_auto_summary.md`

## 相关文件

- `assets/requirements_analysis.md`：需求分析草稿
- `assets/brick_specs.md`：基础砖块外形特征与采集说明
- `detectors/`：网页演示和命令行检测使用的识别模块
- `utils/crawl_lemuwu_bricks.py`：lemuwu 基础砖图片采集
- `utils/generate_brick_samples.py`：基础砖块样本图生成补充工具
- `utils/check_dataset_quality.py`：图片质量检查
- `utils/resize_and_rename.py`：PIL 缩放与重命名
- `utils/split_dataset.py`：分层划分脚本

## 第二阶段训练代码

当前项目已补齐第二阶段训练骨架，包括：

- `model/lego_dataset.py`：`LegoDataset` 自定义类
- `model/lego_cnn.py`：`LegoCNN` 模型定义
- `model/training_utils.py`：训练、评估、Early Stopping、绘图工具
- `scripts/stage2/visualize_batch.py`：批次可视化
- `scripts/stage2/train_baseline.py`：基线训练
- `scripts/stage2/run_experiments.py`：超参数对比实验与汇总导出
- `scripts/stage2/evaluate_model.py`：测试集评估
- `scripts/stage2/predict_single.py`：单张图片分类预测

常用命令：

```bash
python -m scripts.stage2.visualize_batch
python -m scripts.stage2.train_baseline --epochs 20 --batch-size 8 --learning-rate 0.001 --dropout 0.0 --base-channels 48 --depth 4 --auto-crop --augmentation-policy affine
python -m scripts.stage2.run_experiments --experiment batch_size --data-root data/splits_stage2_raw --image-size 128 --learning-rate 0.0005 --dropout 0.0 --optimizer adam --base-channels 48 --depth 4 --auto-crop
python -m scripts.stage2.run_experiments --experiment augmentation --data-root data/splits_stage2_raw --image-size 128 --batch-size 8 --learning-rate 0.0005 --dropout 0.0 --optimizer adam --base-channels 48 --depth 4 --auto-crop
python -m scripts.stage2.evaluate_model --checkpoint runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth --data-root data/splits_stage2_raw
python -m scripts.stage2.predict_single data/raw/1x4/photo_1x4_0006.jpg --checkpoint runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth
```

面向实拍图的改进建议：

- 可使用 `data/raw` 重新划分第二阶段专用训练集，例如 `data/splits_stage2_raw`
- 建议在训练时启用 `--auto-crop`，先自动裁剪乐高主体再缩放
- `run_experiments.py` 支持先指定一组基础参数，再对单个超参数做对比，输出 `summary.json`、`summary.md` 和两张对比图
- 当前可选增强策略为 `none`、`affine`、`affine_flip`、`affine_color`
- 启用 `--auto-crop` 时，归一化均值和方差会按裁剪后的训练图像重新统计，避免输入分布不一致
- 如果要评估旧版 6 类 checkpoint，可额外传入 `--class-names 1x1 1x2 1x4 2x2 2x3 2x4` 指定类别顺序
- 当前 7 类应用模型默认使用 `runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6`

## 第三阶段应用代码

当前项目已改为 Streamlit 多页面第三阶段应用，包含：

- `model/model_def.py`：应用侧 checkpoint 加载与模型实例构建
- `model/inference.py`：统一推理入口，输出预测类别和 Top-K 概率
- `scripts/stage3/verify_stage3.py`：阶段 1/2/3 联合验证脚本
- `utils/stage3_streamlit.py`：Streamlit 页面共用的数据、检测、概率图辅助函数
- `app.py`：Streamlit 主入口首页
- `pages/1_Classification.py`：分类识别页面
- `pages/2_Examples.py`：示例体验页面
- `pages/3_About.py`：阶段完成情况页面

主要页面入口：

- `app.py` 首页：几何检测与项目概览
- `Classification` 页面：调用第二阶段分类模型做单张/批量/拍照识别
- `Examples` 页面：展示测试集示例预测结果
- `About` 页面：汇总第一、二、三阶段状态

当前第三阶段页面能力：

- 使用 Streamlit 多页面结构组织应用
- 分类识别页接入第二阶段 7 类 Brick 模型
- 使用 Plotly 展示 Top-K 概率分布图
- 保留边界情况处理与识别历史
- 关于页汇总阶段结果，便于写报告

常用命令：

```bash
streamlit run app.py
python -m scripts.stage3.verify_stage3 --output runs/stage3/verification_summary.json
python -m scripts.stage2.batch_predict_photos --checkpoint runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth
```
