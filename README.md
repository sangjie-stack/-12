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
python lego_size_detector.py OIP-C.webp
```

网页演示：

```bash
python app.py
```

打开 `http://127.0.0.1:5000` 即可使用。

支持的模式：

- 单主体尺寸 + 层高检测
- 单主体只检测尺寸
- 单主体只检测层高
- 多主体堆叠检测
- 前端生成乐高模型并检测 `长 x 宽 x 高`

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
data/raw/1x4
data/raw/2x2
data/raw/2x3
data/raw/2x4
data/processed
data/splits
```

## 数据准备脚本

0. 从 lemuwu 采集 6 类基础砖图片

```bash
python utils/crawl_lemuwu_bricks.py data/raw --limit-per-class 20 --delay 0.8 --clean
```

说明：

- 当前推荐优先使用网站采集图
- 仅抓取 6 类基础砖：`1x1`、`1x2`、`1x4`、`2x2`、`2x3`、`2x4`
- 使用站点页面中的颜色列表，按可用度优先下载
- 默认低频抓取，并生成 `assets/lemuwu_crawl_report.json`

0b. 自动生成 6 类 Brick 白底样本图

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
- 可选生成 6 种 Brick 规格白底样本图作为补充
- 检查 `data/raw` 中的图片质量
- 使用 PIL 缩放到 `64x64` 并统一命名
- 生成 `70% / 15% / 15%` 的数据划分
- 输出阶段摘要 `assets/stage1_auto_summary.md`

## 相关文件

- `assets/requirements_analysis.md`：需求分析草稿
- `assets/brick_specs.md`：6 种基础砖块外形特征与采集说明
- `utils/crawl_lemuwu_bricks.py`：lemuwu 基础砖图片采集
- `utils/generate_brick_samples.py`：基础砖块样本图生成补充工具
- `utils/check_dataset_quality.py`：图片质量检查
- `utils/resize_and_rename.py`：PIL 缩放与重命名
- `utils/split_dataset.py`：分层划分脚本
