from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .geometry import box_to_quad, infer_grid_from_centers, mask_to_quad, warp_from_quad


@dataclass
class SizePrediction:
    canonical_size: Tuple[int, int]
    confidence: float
    top_face_confidence: float
    mean_stud_confidence: float
    stud_count: int
    top_face_quad: List[List[float]]
    stud_centers: List[List[float]]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def _extract_top_face(result: Any, shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    height, width = shape[:2]
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError("Top-face model returned no detections.")

    best_index = int(np.argmax(result.boxes.conf.cpu().numpy()))
    confidence = float(result.boxes.conf[best_index].item())

    if result.masks is not None and len(result.masks.data) > best_index:
        mask = result.masks.data[best_index].cpu().numpy()
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        quad = mask_to_quad(mask)
    else:
        box = result.boxes.xyxy[best_index].cpu().numpy().tolist()
        quad = box_to_quad(box)
    return quad, confidence


def _extract_studs(result: Any) -> Tuple[List[List[float]], float]:
    if result.boxes is None or len(result.boxes) == 0:
        return [], 0.0

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    centers: List[List[float]] = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        centers.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    mean_conf = float(np.mean(confs)) if len(confs) else 0.0
    return centers, mean_conf


def _draw_original(image: np.ndarray, quad: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    polygon = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [polygon], True, (0, 220, 255), 3)
    cv2.putText(
        output,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _draw_warped(image: np.ndarray, centers: Sequence[Sequence[float]], label: str) -> np.ndarray:
    output = image.copy()
    for center_x, center_y in centers:
        cv2.circle(output, (int(round(center_x)), int(round(center_y))), 10, (0, 220, 0), 2)
        cv2.circle(output, (int(round(center_x)), int(round(center_y))), 2, (0, 220, 0), -1)
    cv2.putText(
        output,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def predict_size_from_path(
    image_path: Path,
    top_face_weights: Path,
    stud_weights: Path,
    output_dir: Optional[Path] = None,
    imgsz: int = 960,
    top_face_conf: float = 0.25,
    stud_conf: float = 0.25,
) -> SizePrediction:
    image_path = Path(image_path)
    top_face_weights = Path(top_face_weights)
    stud_weights = Path(stud_weights)

    image = _load_image(image_path)
    top_face_model = YOLO(str(top_face_weights))
    stud_model = YOLO(str(stud_weights))

    top_results = top_face_model.predict(source=image, imgsz=imgsz, conf=top_face_conf, verbose=False)
    quad, top_confidence = _extract_top_face(top_results[0], image.shape)
    warped, _ = warp_from_quad(image, quad)

    stud_results = stud_model.predict(source=warped, imgsz=imgsz, conf=stud_conf, verbose=False)
    centers, mean_stud_confidence = _extract_studs(stud_results[0])
    canonical_size, grid_confidence = infer_grid_from_centers(centers)
    overall_confidence = float(np.clip(0.35 * top_confidence + 0.25 * mean_stud_confidence + 0.40 * grid_confidence, 0.0, 1.0))

    prediction = SizePrediction(
        canonical_size=canonical_size,
        confidence=overall_confidence,
        top_face_confidence=top_confidence,
        mean_stud_confidence=mean_stud_confidence,
        stud_count=len(centers),
        top_face_quad=quad.astype(float).tolist(),
        stud_centers=[[float(x), float(y)] for x, y in centers],
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        label = f"size={prediction.canonical_size[0]} x {prediction.canonical_size[1]} conf={prediction.confidence:.2f}"
        original_vis = _draw_original(image, quad, label)
        warped_vis = _draw_warped(warped, centers, label)

        cv2.imwrite(str(output_dir / f"{image_path.stem}_original.png"), original_vis)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_warped.png"), warped)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_warped_detected.png"), warped_vis)
        with open(output_dir / "result.json", "w", encoding="utf-8") as file:
            json.dump(prediction.to_json(), file, ensure_ascii=False, indent=2)

    return prediction
