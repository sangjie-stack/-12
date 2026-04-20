# `top_face_seg` 标注指南

这个文档说明如何给第一阶段任务 `top_face_seg` 打标签。

## 目标

- 只标“乐高积木顶部可见面”。
- 不要把侧面、阴影、反光高亮一起框进去。
- 如果顶部面被轻微遮挡，按你认为真实的顶部边界去贴合可见区域。

## 类别定义

- `0: top_face`

对应配置文件：`configs/top_face_seg.yaml`

## 标签格式

推荐使用 YOLOv8 segmentation 格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 ...
```

注意：

- 所有坐标都是归一化坐标，范围在 `0 ~ 1`。
- 坐标顺序建议按轮廓顺时针或逆时针连续书写。
- 至少 3 个点，通常顶部面用 4 个点就够了。

## 这类图怎么标

对于像 `OIP-C.webp` 这种斜视图：

- 标顶部那一块红色平面
- 不标正面和左右侧面
- 不跟着圆形 studs 的边缘走
- 沿顶部面的四边边界走

也就是说，你标的是“顶部平面轮廓”，不是每个 stud 的轮廓。

## 示例

参考文件：

- `examples/top_face_seg/OIP-C.example.txt`
- `examples/top_face_seg/template.example.txt`

## 检查标签

可以用这个脚本把分割标签画回图片里：

```bash
python scripts/visualize_top_face_label.py --image ..\OIP-C.webp --label examples\top_face_seg\OIP-C.example.txt --output runs\label_check\OIP-C_overlay.png
```

## 常见错误

- 把整个积木外轮廓标成 `top_face`
- 把顶部 studs 圆边也纳入轮廓
- 点顺序乱跳，导致多边形自交
- 坐标没有归一化，直接写成像素值
- 一个图片有多个积木时，没有只选当前目标积木的顶部面

## 标注工具建议

- `labelme`
- `CVAT`
- `Roboflow`

如果导出的不是 YOLO segmentation，也可以先导出成多边形格式，我后面可以再帮你写转换脚本。
