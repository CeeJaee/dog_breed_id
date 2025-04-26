import tensorflow as tf
from data_loader import load_datasets, get_class_names
from config import MODEL_SAVE_PATH
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model():
    # load datasets, only need test set
    _, _, test_dataset = load_datasets()
    class_names = get_class_names()

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # Evaluate on test set: loss, top 1 and top 5 accuracy
    test_loss, test_acc, test_top5_acc = model.evaluate(test_dataset)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"\nTest top-5 accuracy: {test_top5_acc:.4f}")

    # display sample predictions
    display_sample_predictions(model, test_dataset, class_names)

def display_sample_predictions(model, dataset, class_names):
    plt.figure(figsize=(15, 15))

    # get one batch from the dataset
    for images, labels in dataset.take(1):
        predictions = model.predict(images)

        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            # get the true and predicted labels...and prob score
            true_label = class_names[np.argmax(labels[i])]
            pred_label = class_names[np.argmax(predictions[i])]
            pred_prob = np.max(predictions[i])

            # titles for true and predicted labels. Correct=green, incorrect=red
            title = f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})"
            if true_label == pred_label:
                plt.title(title, color='green')
            else:
                plt.title(title, color='red')

            plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()