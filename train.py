import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from data_loader import load_datasets
from config import (IMAGE_SIZE, NUM_CLASSES, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH)

# helper function for train()
def create_model():
    # define cnn architecture

    #TODO: define layers of the model here


    '''
    model.compile(
        optimizer = keras.optimizers.RMSprop(LEARNING_RATE),
        loss = "categorical_crossentropy",
        metrics= ["accuracy"]
    )
    '''
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
        )
    ]

    # TODO: Train the model
    history = model.fit(
        
    )

    return model

if __name__ == "__main__":
    train()