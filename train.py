import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from data_loader import load_datasets
from utils import plot_training_curves
from config import (IMAGE_SIZE, NUM_CLASSES, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH)

# helper function for train()
def create_model():
    # define cnn architecture

    #---basic model architecture---#
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
    #---end basic model architecture---#

    # TODO: create more optimal architecture here
    #---advanced model architecture---#
    #inputs = 

    # Data augmentation layers
    '''
    note: the textbook creates an array to store all the augmentations.
          For our purposes, we don't need to do that, just create the
          augmentation layers as is.
    example: 
    inputs = ...

    # Data augmentation layers
    x = layers.RandomFlip("horizontal")
    x = ...rest of the augmentation layers

    x = ...rest of model architecture

    '''

    #---end advanced model architecture---#
    
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

    # TODO: Train the model
    history = model.fit(
        train_dataset,
        epochs = EPOCHS,
        validation_data = val_dataset,
        callbacks = callbacks_list
    )

    # Plot training history
    plot_training_curves(history)

    return model

if __name__ == "__main__":
    train()