import matplotlib.pyplot as plt

def plot_training_curves(hisotry):
    # plot the training and validation metrics
    metrics = ["loss", "accuracy", "top5_accuracy"]

    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)

        plt.plot(history.history[metric], label = f"Training {metric}")
        plt.plot(history.history[f"val){metric}"], label = f"Validation {metric}")

        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.tight_layout
    plt.show()