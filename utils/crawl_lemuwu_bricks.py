import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dataset_config import BRICK_CLASSES


BASE_URL = "https://www.lemuwu.com"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

PART_CLASS_MAP: Dict[str, str] = {
    "3005": "1x1",
    "3004": "1x2",
    "3010": "1x4",
    "3003": "2x2",
    "3002": "2x3",
    "3001": "2x4",
}


def http_get(url: str, timeout: int = 20) -> bytes:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str) -> str:
    return http_get(url).decode("utf-8", errors="replace")


def fetch_binary(url: str) -> bytes:
    return http_get(url)


def extract_json_block(pattern: str, html: str) -> str:
    match = re.search(pattern, html, flags=re.S)
    if not match:
        raise ValueError(f"Could not find pattern: {pattern}")
    return match.group(1)


def sanitize_filename(text: str) -> str:
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text.strip(), flags=re.U)
    return safe.strip("_") or "unknown"


def parse_part_page(part_id: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    html = fetch_text(f"{BASE_URL}/part/{part_id}")
    item_json = extract_json_block(r"item\s*=\s*(\{.*?\});", html)
    colors_json = extract_json_block(r"var colors\s*=\s*(\[.*?\]);", html)
    item = json.loads(item_json)
    colors = json.loads(colors_json)
    return item, colors


def build_image_url(part_id: str, color: int, size: str = "ori") -> str:
    return f"{BASE_URL}/pic/part/{size}/{quote(part_id)}@{color}.png"


def select_colors(colors: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    enriched = []
    for color_info in colors:
        abundance = int(color_info.get("abundance") or 0)
        minyear = color_info.get("minyear") if color_info.get("minyear") is not None else 0
        enriched.append((abundance, minyear, int(color_info.get("color", 0)), color_info))
    enriched.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[-1] for item in enriched[:limit]]


def ensure_class_directories(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for class_name in BRICK_CLASSES:
        (output_root / class_name).mkdir(parents=True, exist_ok=True)


def remove_existing_pngs(class_dir: Path) -> None:
    for path in class_dir.glob("lemuwu_*.png"):
        path.unlink()


def crawl_dataset(output_root: Path, limit_per_class: int, delay_seconds: float, clean: bool) -> Dict[str, object]:
    ensure_class_directories(output_root)
    summary: Dict[str, object] = {"classes": {}, "errors": []}

    for part_id, class_name in PART_CLASS_MAP.items():
        class_dir = output_root / class_name
        if clean:
            remove_existing_pngs(class_dir)

        try:
            item, colors = parse_part_page(part_id)
        except Exception as exc:
            summary["errors"].append({"part_id": part_id, "class_name": class_name, "error": str(exc)})
            continue

        chosen = select_colors(colors, limit_per_class)
        downloaded: List[Dict[str, object]] = []
        summary["classes"][class_name] = {
            "part_id": part_id,
            "title": item.get("chn", ""),
            "requested": limit_per_class,
            "downloaded": 0,
            "files": downloaded,
        }

        for index, color_info in enumerate(chosen, start=1):
            color_id = int(color_info["color"])
            color_name = str(color_info.get("cchn") or color_id)
            image_url = build_image_url(part_id, color_id, size="ori")
            filename = f"lemuwu_{class_name}_{index:04d}_{part_id}_{color_id}_{sanitize_filename(color_name)}.png"
            save_path = class_dir / filename

            try:
                content = fetch_binary(image_url)
                save_path.write_bytes(content)
                downloaded.append(
                    {
                        "path": str(save_path),
                        "url": image_url,
                        "color_id": color_id,
                        "color_name": color_name,
                        "abundance": int(color_info.get("abundance") or 0),
                    }
                )
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                summary["errors"].append(
                    {
                        "part_id": part_id,
                        "class_name": class_name,
                        "color_id": color_id,
                        "url": image_url,
                        "error": str(exc),
                    }
                )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        summary["classes"][class_name]["downloaded"] = len(downloaded)
        summary["classes"][class_name]["available_colors"] = len(colors)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl six base Brick classes from lemuwu.com into data/raw.")
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output dataset root, for example data/raw",
    )
    parser.add_argument("--limit-per-class", type=int, default=20, help="Maximum images to download for each class.")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay in seconds between downloads.")
    parser.add_argument("--clean", action="store_true", help="Delete existing lemuwu_*.png files before crawling.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("assets/lemuwu_crawl_report.json"),
        help="JSON report output path.",
    )
    args = parser.parse_args()

    summary = crawl_dataset(args.output_root, args.limit_per_class, args.delay, args.clean)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("采集完成。")
    for class_name in BRICK_CLASSES:
        info = summary["classes"].get(class_name)
        if not info:
            print(f"{class_name}: 失败")
            continue
        print(
            f"{class_name}: {info['downloaded']} / {info['requested']} "
            f"(part {info['part_id']}, colors {info['available_colors']})"
        )
    print(f"错误数量: {len(summary['errors'])}")
    print(f"报告已保存到: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
