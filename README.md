# Metal Surface Scratch Detector

A starter project for industrial visual inspection using **Python + OpenCV + Streamlit**.
The repository is designed as a practical first step toward surface defect detection on metallic parts.
The app works with a classical computer-vision pipeline and is structured so it can be upgraded into a deep-learning model.

---

## Features

- Upload a metal-surface image from the browser
- Detect likely scratches with a classical OpenCV pipeline
- Show:
  - original image
  - annotated image
  - binary defect mask
  - confidence score
  - defect / no defect decision
- Download the final annotated result as PNG
- Clean project structure ready for GitHub

---

## Project structure

```text
metal-scratch-detector/
│
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ .streamlit/
│  └─ config.toml
│
├─ src/
│  ├─ __init__.py
│  ├─ preprocess.py
│  ├─ detect_classic.py
│  ├─ infer.py
│  ├─ utils.py
│  └─ visualize.py
│
├─ models/
│  └─ README.md
│
├─ data/
│  ├─ raw/
│  │  └─ .gitkeep
│  ├─ processed/
│  │  └─ .gitkeep
│  └─ samples/
│     └─ .gitkeep
│
└─ outputs/
   └─ predictions/
      └─ .gitkeep
```

---

## Installation

### 1) Create and activate a virtual environment

#### Windows PowerShell
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```

---

## How the current detector works

The first version uses a **classical image-processing pipeline**:

1. resize while keeping aspect ratio
2. convert to grayscale
3. normalize contrast
4. apply CLAHE enhancement
5. use directional black-hat morphology to highlight thin dark defects
6. combine that with gradient energy
7. threshold the response map
8. clean the binary mask with morphology + connected-component filtering
9. score the final candidates and return a confidence value

This makes the project lightweight and easy to understand.

---

## Recommended dataset for the next step

For a more serious benchmark, use **KolektorSDD** or **KolektorSDD2**.

- `KolektorSDD`:
  - 399 images total
  - 52 with visible defects
  - 347 without defects
  - captured in a controlled real industrial environment
- `KolektorSDD2`:
  - 356 positive images
  - 2979 negative images
  - includes several defect types

Create your own train / validation / test split after downloading the dataset, or follow the official published splits if provided.

---

## Future upgrades

### Upgrade path A: better classical CV
- adaptive threshold tuning
- line-based filtering
- texture descriptors
- batch analysis

### Upgrade path B: deep learning
Later, you can add:
- classification model
- segmentation model
- Ultralytics workflow
- ONNX export for faster inference

---

## Tips

- Use consistent lighting if you test with your own photos.
- Start with long, visible scratches.
- The classical method is a strong baseline, but it is not a guaranteed industrial-grade detector on all textures.

---

## Suggested GitHub screenshots

For a strong repository presentation, add:
- one input image
- one annotated output
- one binary mask
- one short GIF of the app

---

## License note

Check the dataset license before commercial use.
The code in this starter project is your editable project scaffold.

---

## Author credit

Built By Alan Masoud.
