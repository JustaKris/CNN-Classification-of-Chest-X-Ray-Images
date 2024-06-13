import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.config import BATCH_SIZE, CLASSES


CLASS_NAMES = list(CLASSES.values())
METRIC = 'weighted_sparse_categorical_accuracy'

# Accuracy and loss training plots per epoch
def plot_accuracy_loss(history):

    fig = plt.figure(figsize=(15,9))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history[METRIC],'o-', label = "train")
    plt.plot(history.history['val_' + METRIC], 'o-', label = "val")
    plt.title("train vs val accuracy")
    plt.ylabel(METRIC)
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss
    plt.subplot(222)
    plt.plot(history.history['loss'],'o-', label = "train")
    plt.plot(history.history['val_loss'], 'o-', label = "val")
    plt.title("train vs val loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()
    
# Confusion matrix plot
def plot_confusion_matrix(model, dataset, class_names=CLASS_NAMES):
    """
    Plots a confusion matrix using predictions from the model on the given dataset.
    
    Args:
        model (tf.keras.Model): The trained model.
        dataset (tf.data.Dataset): The dataset to predict.
        class_names (list): List of class names for labeling the confusion matrix.
    """
    true = []
    pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        true.extend(labels.numpy())
        pred.extend(np.argmax(predictions, axis=1))

    cf_matrix = confusion_matrix(true, pred)

    ax = plt.axes()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.heatmap(cf_matrix, 
                annot=True, 
                linewidths=2, 
                cmap='Blues', 
                annot_kws={"size": 15}, 
                xticklabels=class_names, 
                yticklabels=class_names, 
                ax=ax)
    plt.show()

# Classification report function
def show_classification_report(model, dataset, class_names=CLASS_NAMES):
    """
    Prints a classification report using predictions from the model on the given dataset.
    
    Args:
        model (tf.keras.Model): The trained model.
        dataset (tf.data.Dataset): The dataset to predict.
        class_names (list): List of class names for labeling the classification report.
    """
    true = []
    pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        true.extend(labels.numpy())
        pred.extend(np.argmax(predictions, axis=1))
    
    print(classification_report(true, pred, target_names=class_names))