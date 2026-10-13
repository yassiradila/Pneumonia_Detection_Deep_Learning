"""InceptionV3 transfer learning model architecture and training setup."""

from typing import Tuple

try:
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
except (ImportError, ModuleNotFoundError):
    from keras.applications import InceptionV3
    from keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from keras.models import Model
    from keras.optimizers import Adam

from .config import Config


def build_inception_v3(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    dense_units: int = 1024,
    dropout_rate: float = 0.0,
    weights: str = "imagenet",
    learning_rate: float = 1e-3,
) -> Tuple[Model, Model]:
    """Build and compile the InceptionV3 Transfer Learning model.

    Phase 1: The convolutional base is frozen, and only custom top layers
    are trained for feature extraction.

    Args:
        input_shape: Dimensions of the input tensor (H, W, C).
        dense_units: Number of neurons in the intermediate dense layer.
        dropout_rate: Optional dropout rate between dense layers.
        weights: Pre-trained weight initialization (default: 'imagenet').
        learning_rate: Learning rate for Phase 1 compilation.

    Returns:
        Tuple[Model, Model]: (full_model, base_model)
    """
    # 1. Load Pretrained InceptionV3 base without top classification layers
    base_model = InceptionV3(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )

    # 2. Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # 3. Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(dense_units, activation="relu", name="dense_features")(x)

    if dropout_rate > 0.0:
        x = Dropout(dropout_rate, name="dropout_layer")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=predictions, name="Pneumonia_InceptionV3")

    # 4. Compile the model for binary classification
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_for_fine_tuning(
    model: Model,
    base_model: Model,
    unfreeze_layers: int = 100,
    learning_rate: float = 1e-5,
) -> Model:
    """Unfreeze the top layers of the InceptionV3 base for fine-tuning.

    Phase 2: Fine-tune the top layers of the backbone with a lower learning
    rate to adapt high-level feature representations to medical X-rays.

    Args:
        model: The compiled Keras model.
        base_model: The underlying InceptionV3 base model.
        unfreeze_layers: Number of layers from the end of the base to unfreeze.
        learning_rate: Fine-tuning learning rate (typically 1e-5).

    Returns:
        Model: The recompiled model ready for fine-tuning.
    """
    # Unfreeze the last N layers of the base model
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    # Re-compile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
