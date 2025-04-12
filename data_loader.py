import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE

def load_datasets():
    # loads training, validation, and test datasets to be able to be fed into model
    # TODO: train set
    train_dataset = image_dataset_from_directory(

    )

    # TODO: validation set


    # TODO: test set

    return train_dataset, val_dataset, test_dataset

def get_class_names():
    # get a sorted list of class names aka dog breeds
    train_dataset = image_dataset_from_directory(TRAIN_DIR)
    return train_dataset.class_names