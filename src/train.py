"""CLI Training Pipeline for Pneumonia Detection with InceptionV3.

Executes a two-phase transfer learning workflow:
Phase 1: Feature extraction with frozen base model.
Phase 2: Fine-tuning top 100 layers with a lower learning rate.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except (ImportError, ModuleNotFoundError):
    from keras.callbacks import EarlyStopping, ModelCheckpoint

from .config import Config
from .dataset import get_data_generators
from .model import build_inception_v3, unfreeze_for_fine_tuning


def plot_and_save_curves(history_phase1, history_phase2, output_dir: str):
    """Plot and save combined training accuracy and loss curves."""
    os.makedirs(output_dir, exist_ok=True)

    acc = history_phase1.history.get("accuracy", []) + (
        history_phase2.history.get("accuracy", []) if history_phase2 else []
    )
    val_acc = history_phase1.history.get("val_accuracy", []) + (
        history_phase2.history.get("val_accuracy", []) if history_phase2 else []
    )
    loss = history_phase1.history.get("loss", []) + (
        history_phase2.history.get("loss", []) if history_phase2 else []
    )
    val_loss = history_phase1.history.get("val_loss", []) + (
        history_phase2.history.get("val_loss", []) if history_phase2 else []
    )

    epochs_range = range(1, len(acc) + 1)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563EB", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#059669", linewidth=2)
    if history_phase2:
        plt.axvline(x=len(history_phase1.history["accuracy"]), color="gray", linestyle="--", label="Fine-Tuning Start")
    plt.title("Model Accuracy Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved accuracy curve to: {acc_path}")

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Training Loss", color="#DC2626", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#D97706", linewidth=2)
    if history_phase2:
        plt.axvline(x=len(history_phase1.history["loss"]), color="gray", linestyle="--", label="Fine-Tuning Start")
    plt.title("Model Loss Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved loss curve to: {loss_path}")


def train(args):
    """Run full two-stage training workflow."""
    print("=" * 70)
    print(" Pneumonia Detection - InceptionV3 Training Pipeline")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/5] Loading data from '{args.data_dir}'...")
    train_gen, val_gen, test_gen = get_data_generators(
        data_dir=args.data_dir,
        target_size=Config().target_size,
        batch_size=args.batch_size,
    )
    print(f"      Train samples: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

    # 2. Build Model
    print("\n[2/5] Building InceptionV3 architecture (Phase 1)...")
    model, base_model = build_inception_v3(
        input_shape=Config().input_shape,
        dense_units=Config.DENSE_UNITS,
        dropout_rate=Config.DROPOUT_RATE,
        learning_rate=args.lr_phase1,
    )

    # 3. Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=Config.PATIENCE_EARLY_STOP,
        restore_best_weights=True,
    )
    checkpoint = ModelCheckpoint(
        args.checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # 4. Phase 1 Training: Feature Extraction
    print(f"\n[3/5] Starting Phase 1 Training (Max {args.epochs_phase1} epochs)...")
    history_p1 = model.fit(
        train_gen,
        epochs=args.epochs_phase1,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    history_p2 = None
    if args.epochs_phase2 > 0:
        # 5. Phase 2 Training: Fine-Tuning
        print(f"\n[4/5] Starting Phase 2 Fine-Tuning (Unfreezing top {args.unfreeze_layers} layers)...")
        model = unfreeze_for_fine_tuning(
            model=model,
            base_model=base_model,
            unfreeze_layers=args.unfreeze_layers,
            learning_rate=args.lr_phase2,
        )

        history_p2 = model.fit(
            train_gen,
            epochs=args.epochs_phase2,
            validation_data=val_gen,
            callbacks=[early_stop, checkpoint],
            verbose=1,
        )

    # 6. Evaluation on Test Set
    print("\n[5/5] Evaluating final model on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\n>>> Final Test Loss:     {test_loss:.4f}")
    print(f">>> Final Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Save final model
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    model.save(args.save_model)
    print(f"\n[SUCCESS] Final model saved to '{args.save_model}'")

    # Save curves and metrics
    plot_and_save_curves(history_p1, history_p2, args.assets_dir)


def main():
    parser = argparse.ArgumentParser(description="Train InceptionV3 for Pneumonia Detection")
    parser.add_argument("--data-dir", type=str, default=Config.DEFAULT_DATA_DIR, help="Path to chest_xray directory")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--epochs-phase1", type=int, default=Config.EPOCHS_PHASE1, help="Epochs for Phase 1")
    parser.add_argument("--epochs-phase2", type=int, default=Config.EPOCHS_PHASE2, help="Epochs for Phase 2 fine-tuning")
    parser.add_argument("--lr-phase1", type=float, default=Config.LR_PHASE1, help="Learning rate for Phase 1")
    parser.add_argument("--lr-phase2", type=float, default=Config.LR_PHASE2, help="Learning rate for Phase 2")
    parser.add_argument("--unfreeze-layers", type=int, default=Config.UNFREEZE_TOP_LAYERS, help="Top layers to unfreeze")
    parser.add_argument("--checkpoint-path", type=str, default=Config.CHECKPOINT_PATH, help="Checkpoint filepath")
    parser.add_argument("--save-model", type=str, default=Config.FINAL_MODEL_PATH, help="Path to save final model (.keras)")
    parser.add_argument("--assets-dir", type=str, default=Config.RESULTS_DIR, help="Directory to save metric plots")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
