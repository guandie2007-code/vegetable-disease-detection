"""Convert a trained Keras model to TensorFlow Lite."""
import argparse
from pathlib import Path
import tensorflow as tf

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="artifacts/plant_disease.tflite")
    a = ap.parse_args()
    if not Path(a.model).is_file():
        raise SystemExit(f"Model not found: {a.model}")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(a.model))
    out.write_bytes(converter.convert())
    print(f"TensorFlow Lite model saved to {out}")

if __name__ == "__main__":
    main()
