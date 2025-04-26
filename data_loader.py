import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE

def load_datasets():
    # loads training, validation, and test datasets to be able to be fed into model
    # TODO: train set
    train_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # TODO: validation set
    val_dataset = image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # TODO: test set
    test_dataset = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # help data throughput when training
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def get_class_names():
    # get a sorted list of class names aka dog breeds
    train_dataset = image_dataset_from_directory(TRAIN_DIR)
    return train_dataset.class_names