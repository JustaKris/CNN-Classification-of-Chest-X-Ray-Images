import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASSES


CLASS_NAMES = list(CLASSES.values())

# Accuracy and loss training plots per epoch
def plot_accuracy_loss(history, metric="accuracy"):

    fig = plt.figure(figsize=(12,5))

    # Plot accuracy
    plt.subplot(221)
    # plt.plot(history.history[METRIC],'o-', label = "train")
    # plt.plot(history.history['val_' + METRIC], 'o-', label = "val")
    plt.plot(history[metric],'o-', label = "train")
    plt.plot(history['val_' + metric], 'o-', label = "val")
    plt.title("train vs val accuracy")
    plt.ylabel(metric)
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss
    plt.subplot(222)
    # plt.plot(history.history['loss'],'o-', label = "train")
    # plt.plot(history.history['val_loss'], 'o-', label = "val")
    plt.plot(history['loss'],'o-', label = "train")
    plt.plot(history['val_loss'], 'o-', label = "val")
    plt.title("train vs val loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


def print_confusion_matrix(cm, labels):
    column_width = max([len(x) for x in labels] + [5])  # 5 is a buffer for numbers

    # Print header
    print(" " * column_width, end="")
    for label in labels:
        print(f"{label:>{column_width}}", end="")
    print()

    # Print rows
    for i, label in enumerate(labels):
        print(f"{label:>{column_width}}", end="")
        for j in range(len(labels)):
            print(f"{cm[i, j]:>{column_width}}", end="")
        print()


def evaluate_model(model, dataset, class_names=CLASS_NAMES):
    # Generate predictions
    true = []
    pred = []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=False)
        true.extend(labels.numpy())
        pred.extend(np.argmax(predictions, axis=1))

    # Classification report
    print("\n||---------------- Classification Report ----------------||")
    print(classification_report(true, pred, target_names=class_names))

    # Confusion Matrix
    cf_matrix = confusion_matrix(true, pred)

    # Print Confusion Matrix to Console
    print("||------------------------ Confusion Matrix  ------------------------||")
    print_confusion_matrix(cf_matrix, class_names)
    print()
