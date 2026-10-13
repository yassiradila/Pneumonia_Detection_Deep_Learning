"""Model evaluation script to generate confusion matrix, metrics, and report."""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except (ImportError, ModuleNotFoundError):
    from keras.models import load_model
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except (ImportError, ModuleNotFoundError):
        from keras.src.legacy.preprocessing.image import ImageDataGenerator

from .config import Config


def evaluate_model(model_path: str, test_dir: str, output_dir: str, threshold: float = 0.5):
    """Evaluate trained model on the test dataset and generate visual metrics."""
    print("=" * 70)
    print(" Pneumonia Detection - Model Evaluation")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at '{test_dir}'")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/4] Loading model from '{model_path}'...")
    model = load_model(model_path)

    print(f"\n[2/4] Loading test dataset from '{test_dir}'...")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=Config().target_size,
        batch_size=Config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print(f"\n[3/4] Running inference on {test_gen.samples} test images...")
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_probs > threshold).astype(int).flatten()

    # Classification metrics
    class_names = list(test_gen.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall for Pneumonia
    specificity = tn / (tn + fp)  # Recall for Normal
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    print("\n" + "=" * 45)
    print("           EVALUATION SUMMARY")
    print("=" * 45)
    print(f" Test Accuracy:               {accuracy * 100:.2f}%")
    print(f" Pneumonia Sensitivity (TPR): {sensitivity * 100:.2f}% (Recall)")
    print(f" Normal Specificity (TNR):    {specificity * 100:.2f}%")
    print(f" Pneumonia Precision:         {precision * 100:.2f}%")
    print(f" Pneumonia F1-Score:          {f1:.4f}")
    print(f" False Negatives (Missed):    {fn}")
    print(f" False Positives:             {fp}")
    print("=" * 45 + "\n")

    print("Detailed Classification Report:")
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    print(report_text)

    # 4. Save Confusion Matrix Heatmap
    print("\n[4/4] Generating and saving confusion matrix...")
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14, "fontweight": "bold"},
    )
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=12, fontweight="bold")
    plt.title("Confusion Matrix - InceptionV3 Test Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to: {cm_plot_path}")

    # Save metrics JSON
    metrics_data = {
        "test_samples": int(test_gen.samples),
        "accuracy": float(accuracy),
        "pneumonia_recall_sensitivity": float(sensitivity),
        "normal_specificity": float(specificity),
        "pneumonia_precision": float(precision),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        },
    }
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[INFO] Saved metrics summary to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate InceptionV3 model on Chest X-Ray test set")
    parser.add_argument("--model-path", type=str, default="my_model.keras", help="Path to saved .keras or .h5 model")
    parser.add_argument("--data-dir", type=str, default=None, help="Root dataset directory containing 'test/'")
    parser.add_argument("--test-dir", type=str, default=None, help="Direct path to 'test' folder")
    parser.add_argument("--output-dir", type=str, default=Config.RESULTS_DIR, help="Directory to save figures")
    parser.add_argument("--threshold", type=float, default=Config.DECISION_THRESHOLD, help="Binary decision threshold")

    args = parser.parse_args()

    test_directory = args.test_dir
    if test_directory is None and args.data_dir is not None:
        test_directory = os.path.join(args.data_dir, "test")
    elif test_directory is None:
        test_directory = os.path.join(Config.DEFAULT_DATA_DIR, "test")

    evaluate_model(
        model_path=args.model_path,
        test_dir=test_directory,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
