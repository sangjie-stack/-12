import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFilter

from dataset_config import BRICK_CLASSES


PALETTE = [
    (220, 58, 52),
    (35, 98, 214),
    (248, 190, 26),
    (29, 156, 83),
    (255, 120, 36),
    (120, 78, 184),
]


def parse_dims(class_name: str) -> Tuple[int, int]:
    left, right = class_name.lower().split("x", 1)
    return int(left), int(right)


def shade(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


def add_point(point: Tuple[float, float], vector: Tuple[float, float], scale: float = 1.0) -> Tuple[float, float]:
    return point[0] + vector[0] * scale, point[1] + vector[1] * scale


def project_iso(
    origin: Tuple[float, float],
    length: int,
    width: int,
    cell: float,
    brick_height: float,
) -> Dict[str, List[Tuple[float, float]]]:
    axis_x = (cell, cell * 0.5)
    axis_y = (-cell, cell * 0.5)
    axis_z = (0.0, brick_height)

    top_a = origin
    top_b = add_point(top_a, axis_x, length)
    top_d = add_point(top_a, axis_y, width)
    top_c = add_point(top_b, axis_y, width)

    bottom_a = add_point(top_a, axis_z)
    bottom_b = add_point(top_b, axis_z)
    bottom_c = add_point(top_c, axis_z)
    bottom_d = add_point(top_d, axis_z)

    return {
        "top": [top_a, top_b, top_c, top_d],
        "front": [top_b, top_c, bottom_c, bottom_b],
        "side": [top_d, top_c, bottom_c, bottom_d],
        "bottom": [bottom_a, bottom_b, bottom_c, bottom_d],
        "stud_origin": top_a,
        "axis_x": axis_x,
        "axis_y": axis_y,
    }


def stud_center(
    origin: Tuple[float, float],
    axis_x: Tuple[float, float],
    axis_y: Tuple[float, float],
    col: int,
    row: int,
) -> Tuple[float, float]:
    point = add_point(origin, axis_x, col + 0.5)
    point = add_point(point, axis_y, row + 0.5)
    return point


def draw_stud(draw: ImageDraw.ImageDraw, center: Tuple[float, float], radius_x: float, radius_y: float, color: Tuple[int, int, int]) -> None:
    cx, cy = center
    bbox = [cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y]
    draw.ellipse(bbox, fill=color, outline=shade(color, 0.78), width=2)
    highlight = [cx - radius_x * 0.55, cy - radius_y * 0.55, cx + radius_x * 0.1, cy + radius_y * 0.05]
    draw.ellipse(highlight, fill=shade(color, 1.18))


def render_single_brick(class_name: str, image_size: int, rng: random.Random) -> Image.Image:
    length, width = parse_dims(class_name)
    base = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 255))
    shadow_layer = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    draw = ImageDraw.Draw(base)

    color = rng.choice(PALETTE)
    cell = rng.uniform(image_size * 0.055, image_size * 0.07)
    brick_height = rng.uniform(image_size * 0.11, image_size * 0.15)
    offset_x = rng.uniform(-image_size * 0.05, image_size * 0.05)
    offset_y = rng.uniform(-image_size * 0.03, image_size * 0.04)

    object_height = (length + width) * cell * 0.5 + brick_height
    origin = (
        image_size * 0.5 - (length - width) * cell * 0.5 + offset_x,
        image_size * 0.5 - object_height * 0.5 + offset_y,
    )
    geometry = project_iso(origin, length, width, cell, brick_height)

    bottom = geometry["bottom"]
    shadow_bbox = [
        min(point[0] for point in bottom) - cell * 0.35,
        min(point[1] for point in bottom) - cell * 0.1,
        max(point[0] for point in bottom) + cell * 0.35,
        max(point[1] for point in bottom) + cell * 0.45,
    ]
    shadow_draw.ellipse(shadow_bbox, fill=(0, 0, 0, 58))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=max(4, int(cell * 0.3))))
    base.alpha_composite(shadow_layer)

    draw.polygon(geometry["side"], fill=shade(color, 0.72), outline=shade(color, 0.55))
    draw.polygon(geometry["front"], fill=shade(color, 0.84), outline=shade(color, 0.62))
    draw.polygon(geometry["top"], fill=shade(color, 1.03), outline=shade(color, 0.7))

    top_highlight = [
        geometry["top"][0],
        (
            geometry["top"][0][0] + (geometry["top"][1][0] - geometry["top"][0][0]) * 0.65,
            geometry["top"][0][1] + (geometry["top"][1][1] - geometry["top"][0][1]) * 0.65,
        ),
        (
            geometry["top"][3][0] + (geometry["top"][2][0] - geometry["top"][3][0]) * 0.4,
            geometry["top"][3][1] + (geometry["top"][2][1] - geometry["top"][3][1]) * 0.4,
        ),
        (
            geometry["top"][0][0] + (geometry["top"][3][0] - geometry["top"][0][0]) * 0.55,
            geometry["top"][0][1] + (geometry["top"][3][1] - geometry["top"][0][1]) * 0.55,
        ),
    ]
    draw.polygon(top_highlight, fill=shade(color, 1.12))

    stud_fill = shade(color, 1.08)
    for row in range(width):
        for col in range(length):
            center = stud_center(geometry["stud_origin"], geometry["axis_x"], geometry["axis_y"], col, row)
            center = (center[0], center[1] - brick_height * 0.08)
            draw_stud(draw, center, cell * 0.23, cell * 0.13, stud_fill)

    angle = rng.uniform(-8.0, 8.0)
    rotated = base.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255, 255))
    return rotated.convert("RGB")


def generate_dataset(output_root: Path, samples_per_class: int, image_size: int, seed: int, clean: bool) -> Dict[str, int]:
    rng = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    for class_name in BRICK_CLASSES:
        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            for path in class_dir.glob("*.png"):
                path.unlink()

        for index in range(1, samples_per_class + 1):
            image = render_single_brick(class_name, image_size, rng)
            image.save(class_dir / f"studio_{class_name}_{index:04d}.png", format="PNG")
        counts[class_name] = samples_per_class
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic LEGO Brick sample images for six base classes.")
    parser.add_argument("output_root", type=Path, help="Output root, for example data/raw")
    parser.add_argument("--samples-per-class", type=int, default=20, help="Generated image count for each class.")
    parser.add_argument("--image-size", type=int, default=320, help="Output image width and height.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clean", action="store_true", help="Remove old PNG files before generation.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("assets/generated_brick_dataset_report.json"),
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    counts = generate_dataset(args.output_root, args.samples_per_class, args.image_size, args.seed, args.clean)
    report = {
        "output_root": str(args.output_root),
        "samples_per_class": args.samples_per_class,
        "image_size": args.image_size,
        "seed": args.seed,
        "classes": counts,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("样本生成完成。")
    for class_name, count in counts.items():
        print(f"{class_name}: {count}")
    print(f"生成目录: {args.output_root}")
    print(f"报告已保存到: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
