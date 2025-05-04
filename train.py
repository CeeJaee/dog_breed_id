import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers, callbacks
from data_loader import load_datasets
from utils import plot_training_curves
from config import (IMAGE_SIZE, NUM_CLASSES, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, USE_VGG, VGG_WEIGHTS, FREEZE_VGG)

# helper function for train()
def create_model():

    # VGG

    base_model = VGG16(
        weights=VGG_WEIGHTS,
        include_top = False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    base_model.trainable = False
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    # end of VGG

    # define cnn architecture

    #---basic model architecture---#
    '''
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # image preprocessing
    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # classification, use softmax for multi-class
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    '''
    #---end basic model architecture---#

   
    #---advanced model architecture---#
    '''
    # define cnn architecture
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # augmentation layers
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.1)(x)

    # image preprocessing
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(32, 3, padding="same")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # classification, use softmax for multi-class
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    '''
    #---end advanced model architecture---#
    
    model.compile(
        optimizer = optimizers.Adam(LEARNING_RATE),
        loss = "categorical_crossentropy",
        metrics= ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )
    
    return model

def train():
    # load datasets
    train_dataset, val_dataset, _ = load_datasets()
    model = create_model()

    # TODO: need callbacks
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = MODEL_SAVE_PATH,
            save_best_only = True,
            monitor = "val_accuracy",
            mode = "max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor = "val_accuracy",
            patience = 5,
            restore_best_weights = True
        ),
        # reduces learning rate to get out of a local minimum and into a global min
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = "val_loss",
            factor = 0.2,
            patience = 5,
            min_lr = 1e-6
        )
    ]

    # --- Phase 1: Train only the custom head (frozen VGG) --- #
    '''
    print("\n=== Phase 1: Training Custom Head (Frozen VGG) ===")
    history_phase1 = model.fit(
        train_dataset,
        epochs=10,  # Fewer epochs for initial training
        validation_data=val_dataset,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=MODEL_SAVE_PATH,
                save_best_only=True,
                monitor="val_accuracy"
            ),
            callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    '''
    # --- Phase 2: Unfreeze & Fine-Tune VGG Layers --- #
    print("\n=== Phase 2: Fine-Tuning VGG Layers ===")
    for layer in model._layers[1]._layers[-4:]:  # Unfreeze last 4 layers of VGG
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE / 10),  # Lower LR
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase2 = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=MODEL_SAVE_PATH,
                save_best_only=True,
                monitor="val_accuracy"
            ),
            callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # Combine histories for plotting
    combined_history = {
        "accuracy": history_phase1.history["accuracy"] + history_phase2.history["accuracy"],
        "val_accuracy": history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"],
        # Add other metrics as needed
    }
    plot_training_curves(combined_history)

    return model

if __name__ == "__main__":
    train()