import base64
import imghdr
from typing import Any, Dict, List

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from lego_generated_detector import detect_generated_objects, draw_generated_result
from lego_multi_stack_detector import detect_multi_stack_objects, draw_result as draw_multi_stack_result
from lego_size_detector import detect_lego_size, draw_result as draw_single_result


app = Flask(__name__)


def image_to_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def read_uploaded_image(file_storage: Any) -> np.ndarray:
    content = file_storage.read()
    if not content:
        raise ValueError("上传文件为空。")

    file_type = imghdr.what(None, h=content)
    if file_type not in {"jpeg", "png", "webp", "bmp"}:
        raise ValueError("仅支持 JPG、PNG、WEBP、BMP 图片。")

    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取上传图片。")
    return image


def read_data_url_image(data_url: str) -> np.ndarray:
    if not data_url or "," not in data_url:
        raise ValueError("生成模型图片无效。")
    _, encoded = data_url.split(",", 1)
    content = base64.b64decode(encoded)
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析生成模型图片。")
    return image


def build_single_payload(image: np.ndarray, mode: str) -> Dict[str, Any]:
    result = detect_lego_size(image)
    annotated = draw_single_result(
        image,
        result,
        height_only=mode == "height_only",
        size_only=mode == "size_only",
    )
    payload: Dict[str, Any] = {
        "annotated_image": image_to_data_url(annotated),
        "mode": mode,
        "summary": [],
    }

    if mode != "height_only":
        if result.dims != (0, 0):
            payload["summary"].append(f"尺寸：{result.dims[0]} x {result.dims[1]}")
            payload["summary"].append(f"尺寸置信度：{result.size_confidence:.2f}")
        else:
            payload["summary"].append("尺寸：未识别")

    if mode != "size_only":
        if result.height > 0:
            payload["summary"].append(f"层高：{result.height}")
            payload["summary"].append(f"层高置信度：{result.height_confidence:.2f}")
        else:
            payload["summary"].append("层高：未识别")

    if result.dims != (0, 0) and result.height > 0 and mode == "single":
        payload["summary"].append(f"三维形态：{result.dims[0]} x {result.dims[1]} x {result.height}")
    return payload


def build_multi_payload(image: np.ndarray) -> Dict[str, Any]:
    objects = detect_multi_stack_objects(image)
    annotated = draw_multi_stack_result(image, objects)
    items: List[Dict[str, Any]] = []
    for object_result in objects:
        items.append(
            {
                "name": f"对象 {object_result.index}",
                "shape": f"1 x 1 x {object_result.layers}",
                "confidence": f"{object_result.confidence:.2f}",
            }
        )

    summary = [f"检测到主体数：{len(objects)}"]
    if not objects:
        summary.append("没有检测到可分离的主体。")

    return {
        "annotated_image": image_to_data_url(annotated),
        "original_image": image_to_data_url(image),
        "mode": "multi_stack",
        "summary": summary,
        "objects": items,
    }


def build_generated_payload(image: np.ndarray) -> Dict[str, Any]:
    objects = detect_generated_objects(image)
    annotated = draw_generated_result(image, objects)
    items: List[Dict[str, Any]] = []
    for object_result in objects:
        items.append(
            {
                "name": f"对象 {object_result.index}",
                "shape": f"{object_result.dims[0]} x {object_result.dims[1]} x {object_result.height}",
                "confidence": f"{object_result.confidence:.2f}",
            }
        )

    summary = [f"检测到主体数：{len(objects)}"]
    return {
        "annotated_image": image_to_data_url(annotated),
        "original_image": image_to_data_url(image),
        "mode": "generated_multi",
        "summary": summary,
        "objects": items,
    }


def build_payload(image: np.ndarray, mode: str) -> Dict[str, Any]:
    if mode == "generated_multi":
        return build_generated_payload(image)
    if mode == "multi_stack":
        return build_multi_payload(image)

    payload = build_single_payload(image, mode)
    payload["original_image"] = image_to_data_url(image)
    return payload


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    context: Dict[str, Any] = {
        "result": None,
        "error": None,
        "selected_mode": "single",
        "original_image": None,
    }

    if request.method == "POST":
        file = request.files.get("image")
        generated_image = request.form.get("generated_image", "")
        mode = request.form.get("mode", "single")
        context["selected_mode"] = mode

        if (file is None or file.filename == "") and not generated_image:
            context["error"] = "请先选择一张图片。"
            return render_template("index.html", **context)

        try:
            if generated_image:
                image = read_data_url_image(generated_image)
            else:
                image = read_uploaded_image(file)
            context["original_image"] = image_to_data_url(image)
            context["result"] = build_payload(image, mode)
        except Exception as exc:
            context["error"] = str(exc)

    return render_template("index.html", **context)


@app.route("/detect-json", methods=["POST"])
def detect_json():
    try:
        generated_image = request.form.get("generated_image", "")
        mode = request.form.get("mode", "multi_stack")
        if not generated_image:
            return jsonify({"ok": False, "error": "缺少生成图片。"}), 400

        image = read_data_url_image(generated_image)
        payload = build_payload(image, mode)
        return jsonify({"ok": True, "result": payload})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
