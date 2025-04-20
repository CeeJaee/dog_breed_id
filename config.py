import os

# define dataset paths
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")

# TODO: Model dimension sizes
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 

# training config (compilation/fitting)
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = os.path.join("models", "dog_breed_classifier.keras")

# if models directory dne
os.makedirs("models", exist_ok=True)