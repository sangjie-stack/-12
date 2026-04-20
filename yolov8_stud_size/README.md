# YOLOv8 LEGO 尺寸识别骨架

这个项目骨架专门面向“识别乐高顶部是几 x 几”的任务，而不是通用零件分类。

## 任务定义

推荐使用两阶段方案：

1. `top_face_seg`：分割乐高顶部可见面。
2. `stud_det`：在透视矫正后的顶部图中检测每个 stud。
3. 几何后处理：根据 stud 中心点拟合规则网格，输出 `rows x cols`。

这种做法比直接对整图分类更稳，因为它把“顶部面定位”和“尺寸计数”拆开了。

## 目录结构

```text
yolov8_stud_size/
├─ configs/
│  ├─ top_face_seg.yaml
│  └─ stud_det.yaml
├─ datasets/
│  ├─ top_face_seg/
│  │  ├─ images/
│  │  └─ labels/
│  └─ stud_det/
│     ├─ images/
│     └─ labels/
├─ docs/
│  └─ annotation_guide.md
├─ examples/
│  └─ top_face_seg/
│     ├─ OIP-C.example.txt
│     └─ template.example.txt
├─ scripts/
│  ├─ build_stud_dataset.py
│  ├─ visualize_top_face_label.py
│  ├─ train_top_face_seg.py
│  ├─ train_stud_detector.py
│  └─ predict_size.py
├─ src/
│  └─ lego_size_yolo/
│     ├─ __init__.py
│     ├─ geometry.py
│     └─ pipeline.py
└─ requirements.txt
```

## 标注建议

### 1) 顶部面分割 `top_face_seg`

- 图像：原始乐高图片。
- 标注：只标顶部可见面，类别固定为 `top_face`。
- 格式：YOLOv8 segmentation。

建议覆盖：

- 俯视和斜视角
- 多颜色积木
- 白底和复杂背景
- 反光、高光、阴影
- 带侧面孔洞或特殊外形的砖块

### 2) 顶部 stud 检测 `stud_det`

- 图像：从顶部面透视矫正后的 crop。
- 标注：每个顶部 stud 标一个框，类别固定为 `stud`。
- 格式：YOLOv8 detection。

建议让每张图只保留顶部区域，避免侧面圆孔干扰。

## 安装

```bash
cd E:\乐高模型\yolov8_stud_size
pip install -r requirements.txt
```

## 训练

### 顶部面分割

```bash
python scripts/train_top_face_seg.py --data configs/top_face_seg.yaml --model yolov8n-seg.pt --epochs 100 --imgsz 960
```

### Stud 检测

```bash
python scripts/train_stud_detector.py --data configs/stud_det.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

## 推理

```bash
python scripts/predict_size.py --image ..\OIP-C.webp --top-face-weights runs\top_face_seg\weights\best.pt --stud-weights runs\stud_det\weights\best.pt
```

输出内容：

- 原图标注结果
- 顶部透视矫正图
- 矫正图的 stud 检测可视化
- `result.json`，包含尺寸和置信度

说明：当前输出的 `canonical_size` 会按从小到大排序，因此 `2 x 4` 和旋转后的 `4 x 2` 会统一表示为 `2 x 4`。

## 数据集布局

### `top_face_seg`

```text
datasets/top_face_seg/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

### `stud_det`

```text
datasets/stud_det/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

## 推荐工作流

1. 先标 `top_face_seg` 数据。
2. 训练顶部面分割模型。
3. 用顶部面模型批量裁出矫正后的顶部 crop。
4. 在 crop 上标 stud 检测框。
5. 训练 `stud_det`。
6. 用 `predict_size.py` 跑两阶段推理。

## 标注参考

- 标注说明：`docs/annotation_guide.md:1`
- 示例标签：`examples/top_face_seg/OIP-C.example.txt:1`
- 模板标签：`examples/top_face_seg/template.example.txt:1`

检查标注可视化：

```bash
python scripts/visualize_top_face_label.py --image ..\OIP-C.webp --label examples\top_face_seg\OIP-C.example.txt --output runs\label_check\OIP-C_overlay.png
```

## 批量生成顶部 crop

当你已经标好 `top_face_seg` 之后，可以先把原图自动变成俯视化的顶部 crop，再去标 stud。

```bash
python scripts/build_stud_dataset.py --source-root datasets/top_face_seg --output-root datasets/stud_det
```

脚本会做这些事情：

- 读取 `datasets/top_face_seg/images/{split}` 里的图片
- 读取 `datasets/top_face_seg/labels/{split}` 里的 YOLO 分割标签
- 自动透视矫正出顶部区域
- 保存到 `datasets/stud_det/images/{split}`
- 创建空白的 `datasets/stud_det/labels/{split}` 标签文件，方便继续人工标 stud
- 生成清单文件到 `datasets/stud_det/manifests/{split}.json`

如果你的 `top_face_seg` 暂时是框标注而不是分割标注，这个脚本也能兼容，但效果通常不如分割标注稳定。

## 后续建议

- 如果你的图片角度变化很大，优先扩充 `top_face_seg` 数据。
- 如果误把侧面孔洞识别成顶部 stud，优先清洗 `stud_det` 数据。
- 如果后面要做成单模型，也建议先用这个骨架跑通标签体系和推理链路。
