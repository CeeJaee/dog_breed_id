import tensorflow as tf
from tf.keras import layers, models, optimizers, callbacks
from data_loader import load_datasets
from config import (IMAGE_SIZE, NUM_CLASSES, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH)

# helper function for train()
def create_model():
    # define cnn architecture

    #TODO: define layers of the model here

    #---basic model architecture---#
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    #---end basic model architecture---#

    
    model.compile(
        optimizer = optimizers.RMSprop(LEARNING_RATE),
        loss = "categorical_crossentropy",
        metrics= ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )
    
    return model

def train():
    # load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()
    model = create_model()

    # TODO: need callbacks
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath = MODEL_SAVE_PATH,
            save_best_only = True,
            monitor = "val_loss"
        ),
        keras.callbacks.EarlyStopping(
            monitor = "val_accuracy",
            patience = 2
        )
    ]

    # TODO: Train the model
    history = model.fit(
        train_dataset,
        epochs = EPOCHS,
        validation_data = val_dataset,
        callbacks = callbacks_list
    )

    return model

if __name__ == "__main__":
    train()