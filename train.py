"""Train a MobileNetV2 plant-leaf classifier from class-folder images."""
import argparse
import json
from pathlib import Path
import tensorflow as tf

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--validation-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fine-tune", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.output_dir)
    if not data.is_dir():
        raise SystemExit(f"Dataset directory not found: {data}")
    if not 0 < a.validation_split < 1:
        raise SystemExit("--validation-split must be between 0 and 1.")
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(directory=data, validation_split=a.validation_split, seed=a.seed,
                  image_size=(a.image_size, a.image_size), batch_size=a.batch_size,
                  label_mode="int")
    train = tf.keras.utils.image_dataset_from_directory(subset="training", **kwargs)
    val = tf.keras.utils.image_dataset_from_directory(subset="validation", **kwargs)
    classes = train.class_names
    if len(classes) < 2:
        raise SystemExit("At least two class folders are required.")
    train = train.cache().shuffle(1000, seed=a.seed).prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1)])
    inp = tf.keras.Input((a.image_size, a.image_size, 3))
    x = aug(inp)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(a.image_size, a.image_size, 3), include_top=False, weights="imagenet")
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    model = tf.keras.Model(inp, tf.keras.layers.Dense(len(classes), activation="softmax")(x))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    best = out / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    model.fit(train, validation_data=val, epochs=a.epochs, callbacks=callbacks)
    if a.fine_tune:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=max(1, a.epochs // 2), callbacks=callbacks)
    model = tf.keras.models.load_model(best)
    loss, acc = model.evaluate(val, verbose=0)
    (out / "class_names.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (out / "evaluation.json").write_text(json.dumps({
        "validation_loss": float(loss), "validation_accuracy": float(acc),
        "num_classes": len(classes), "image_size": a.image_size,
        "validation_split": a.validation_split, "seed": a.seed}, indent=2), encoding="utf-8")
    print(f"Saved model: {best}")
    print(f"Validation accuracy for this run: {acc:.4f}")

if __name__ == "__main__":
    main()
