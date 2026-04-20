# LEGO Size Detector

This small Python tool estimates a LEGO part's top stud layout from an image.

## What it does

- Finds the LEGO object against a simple background.
- Detects circular studs on the visible surfaces.
- Keeps the circles that most likely belong to the top face.
- Uses LEGO grid knowledge to infer the stud count, for example `2 x 4` or `3 x 3`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python lego_size_detector.py OIP-C.webp
```

Optional output path:

```bash
python lego_size_detector.py OIP-C.webp --output result.png
```

## Notes

- The best input is a single LEGO brick or plate with a clear top face.
- A light, clean background improves the result.
- Strong occlusion, heavy reflections, or unusual custom pieces will reduce accuracy.

## Web UI

Run the local Flask page:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

Available modes in the page:

- Single object: size + height
- Single object: size only
- Single object: height only
- Multi-stack: detect multiple separated stack objects as `1 x 1 x layers`
