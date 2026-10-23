# Vegetable Disease Detection with MobileNetV2

A computer-vision baseline for classifying plant-leaf images into disease categories using transfer learning with MobileNetV2. It includes training, single-image prediction, and TensorFlow Lite export scripts.

**Honest status:** source code only; no dataset or trained model is included. Run the experiments yourself before reporting accuracy or describing results.

## Dataset layout
Use a permitted dataset such as PlantVillage and arrange it as one folder per class:

```text
data/all_classes/
  Tomato___Early_blight/
  Tomato___healthy/
  Pepper__bell___Bacterial_spot/
```

Class-folder names become the labels. Do not commit the dataset unless its license permits redistribution and the repository size is appropriate.

## Setup
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python src/train.py --data-dir data/all_classes --output-dir artifacts --epochs 10 --batch-size 32
```

Outputs include `best_model.keras`, `class_names.json`, and `evaluation.json`.

## Predict
```bash
python src/predict.py --model artifacts/best_model.keras --labels artifacts/class_names.json --image path/to/leaf.jpg
```

## Export to TensorFlow Lite
```bash
python src/export_tflite.py --model artifacts/best_model.keras --output artifacts/plant_disease.tflite
```

## Method
1. Load class-labeled images.
2. Resize to 224×224 and apply augmentation to training images.
3. Use ImageNet-pretrained MobileNetV2 as the backbone.
4. Train a classification head and optionally fine-tune upper backbone layers.
5. Evaluate on a held-out validation split and export the model.

## Limitations
Controlled dataset images may not represent field conditions. A prediction is not a confirmed agricultural diagnosis. Report only metrics from your actual run, and document dataset version, split, and environment.
