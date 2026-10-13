"""Single image inference and clinical risk prediction script."""

import argparse
import json
import os
import sys

try:
    from tensorflow.keras.models import load_model
except (ImportError, ModuleNotFoundError):
    from keras.models import load_model

from .config import Config
from .dataset import preprocess_single_image


def predict_image(
    image_path: str,
    model_path: str = "my_model.keras",
    threshold: float = 0.5,
) -> dict:
    """Run pneumonia inference on a single chest X-ray image.

    Args:
        image_path: Path to the image file (JPG, PNG, etc.).
        model_path: Path to the trained Keras model.
        threshold: Classification decision boundary (default: 0.5).

    Returns:
        dict: Prediction results including predicted label, confidence, and risk level.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: '{image_path}'")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Please specify a valid model path using --model."
        )

    # 1. Load model
    model = load_model(model_path)

    # 2. Preprocess image
    img_tensor = preprocess_single_image(image_path, target_size=Config().target_size)

    # 3. Predict probability
    raw_prob = float(model.predict(img_tensor, verbose=0)[0][0])

    is_pneumonia = raw_prob >= threshold
    predicted_label = "PNEUMONIA" if is_pneumonia else "NORMAL"
    confidence = raw_prob if is_pneumonia else (1.0 - raw_prob)

    # Clinical risk interpretation
    if raw_prob >= 0.85:
        risk_level = "High Suspicion (Severe Indicators Detected)"
    elif raw_prob >= 0.50:
        risk_level = "Moderate Suspicion (Borderline Infiltrates)"
    elif raw_prob >= 0.20:
        risk_level = "Low Suspicion (Minor Inconsistencies)"
    else:
        risk_level = "Clear / Healthy Lung Fields"

    return {
        "image_path": os.path.abspath(image_path),
        "prediction": predicted_label,
        "probability_pneumonia": round(raw_prob, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "threshold_used": threshold,
        "clinical_assessment": risk_level,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference tool for Chest X-Ray Pneumonia Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--model", type=str, default="my_model.keras", help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=Config.DECISION_THRESHOLD, help="Classification threshold")
    parser.add_argument("--json", action="store_true", help="Print result in JSON format")

    args = parser.parse_args()

    try:
        result = predict_image(
            image_path=args.image,
            model_path=args.model,
            threshold=args.threshold,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 55)
            print("        CHEST X-RAY DIAGNOSTIC INFERENCE")
            print("=" * 55)
            print(f" Image:          {os.path.basename(result['image_path'])}")
            print(f" Diagnosis:      {result['prediction']}")
            print(f" Confidence:     {result['confidence_percentage']}%")
            print(f" Pneumonia Prob: {result['probability_pneumonia'] * 100:.2f}%")
            print(f" Assessment:     {result['clinical_assessment']}")
            print("=" * 55)
            print(" Note: For clinical decision support only. Not a standalone diagnosis.\n")

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
