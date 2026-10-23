"""Predict a class for a single leaf image."""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    a = ap.parse_args()
    for f in (a.model, a.labels, a.image):
        if not Path(f).is_file():
            raise SystemExit(f"File not found: {f}")
    model = tf.keras.models.load_model(a.model)
    labels = json.loads(Path(a.labels).read_text(encoding="utf-8"))
    image = Image.open(a.image).convert("RGB").resize((a.image_size, a.image_size))
    probs = model.predict(np.asarray(image, dtype=np.float32)[None, ...], verbose=0)[0]
    i = int(np.argmax(probs))
    print(f"Predicted class: {labels[i]}")
    print(f"Confidence: {float(probs[i]):.4f}")
    print("This is a model prediction, not a confirmed agricultural diagnosis.")

if __name__ == "__main__":
    main()
