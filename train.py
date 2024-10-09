#!/usr/bin/env python
# coding: utf-8

import os
import random
import json
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from glob import glob
from scipy.io import loadmat
from scipy.stats import ttest_ind
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sklearn.utils.class_weight import compute_class_weight
from torchvision.io import read_image

import tensorflow as tf
from tensorflow import keras
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
from tensorflow.keras.metrics import Metric
from keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Activation

from model import DeeplabV3Plus
from utils.utils import *


with open("config.json", "r") as config_file:
    config = json.load(config_file)

IMAGE_SIZE = config["IMAGE_SIZE"]
NUM_CLASSES = config["NUM_CLASSES"]
DATA_DIR = config["DATA_DIR"]
NUM_TRAIN_IMAGES = config["NUM_TRAIN_IMAGES"]
NUM_VAL_IMAGES = config["NUM_VAL_IMAGES"]
EPOCHS = config["EPOCHS"]
LEARNING_RATE = config["LEARNING_RATE"]


def label_iou(y_true, y_pred):
    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    y_true_label = tf.greater(y_true, 0)
    y_pred_label = tf.greater(y_pred, 0)

    intersection = tf.reduce_sum(tf.cast(y_true_label & y_pred_label, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true_label | y_pred_label, tf.float32))

    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou


def add_sample_weights(image, label):
    weights = config["CLASS_WEIGHTS"]
    class_weights = class_weights / tf.reduce_sum(class_weights)

    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


all_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))
all_masks = [img.replace("images", "masks") for img in all_images]

assert len(all_images) >= NUM_TRAIN_IMAGES + NUM_VAL_IMAGES, "Not enough images for the specified split."

total_images = len(all_images)
all_indices = list(range(total_images))

train_indices = random.sample(all_indices, NUM_TRAIN_IMAGES)
remaining_indices = list(set(all_indices) - set(train_indices))
val_indices = random.sample(remaining_indices, NUM_VAL_IMAGES)

# Select images and masks based on indices
train_images = [all_images[i] for i in train_indices]
train_masks = [all_masks[i] for i in train_indices]
val_images = [all_images[i] for i in val_indices]
val_masks = [all_masks[i] for i in val_indices]


train_dataset = data_generator(train_images, train_masks, augment=False)
val_dataset = data_generator(val_images, val_masks)

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss, metrics=[label_iou, "accuracy"])

model_checkpoint_callback = ModelCheckpoint(filepath="docs/model.h5", save_best_only=True, monitor="val_label_iou", mode="max")

log_dir = "docs/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    train_dataset.map(add_sample_weights), validation_data=val_dataset, epochs=EPOCHS, callbacks=[model_checkpoint_callback, tensorboard_callback]
)

model = keras.models.load_model("docs/model.h5", custom_objects={"label_iou": label_iou})

train_ious = []
val_ious = []
train_accuracy = []
val_accuracy = []
train_label_ious = []
val_label_ious = []

for data in ["train", "val"]:
    if data == "train":
        images = train_images
        masks = train_masks
    else:
        images = [all_images[i] for i in remaining_indices]
        masks = [all_masks[i] for i in remaining_indices]
    for i, img in enumerate(images):
        image_tensor = read_image(img)
        mask = read_image(masks[i], mask=True)
        mask = np.array(mask[:, :, 0])
        actual_blebs = (mask == 2).astype(np.uint8)

        prediction_mask = infer(image_tensor=image_tensor, model=model)
        predicted_blebs = (prediction_mask == 2).astype(np.uint8)

        actual_mask = (mask > 0).astype(np.uint8)
        predicted_mask = (prediction_mask > 0).astype(np.uint8)

        accuracy = np.mean(actual_mask == predicted_mask)

        intersection = np.logical_and(predicted_blebs, actual_blebs).sum()
        union = np.logical_or(predicted_blebs, actual_blebs).sum()
        label_iou = intersection / union if union != 0 else 0

        intersection = np.logical_and(predicted_mask, actual_mask).sum()
        union = np.logical_or(predicted_mask, actual_mask).sum()
        iou = intersection / union if union != 0 else 0

        if data == "train":
            train_label_ious.append(label_iou)
            train_accuracy.append(accuracy)
            train_ious.append(iou)
        else:
            val_label_ious.append(label_iou)
            val_accuracy.append(accuracy)
            val_ious.append(iou)


mean_train_iou = np.mean(train_ious)
mean_val_iou = np.mean(val_ious)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(train_ious, bins=10, kde=True, ax=ax1)
ax1.set_title(f"Training IOU Histogram\nMean IoU: {mean_train_iou:.2f}")
ax1.set_ylabel("")
ax1.set_xlim([0, None])

sns.histplot(val_ious, bins=10, kde=True, ax=ax2)
ax2.set_title(f"Validation IOU Histogram\nMean IoU: {mean_val_iou:.2f}")
ax2.set_ylabel("")
ax2.set_xlim([0, None])

plt.tight_layout()
plt.savefig("docs/ious.png")


mean_train_label_iou = np.mean(train_label_ious)
mean_val_label_iou = np.mean(val_label_ious)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(train_ious, bins=10, kde=True, ax=ax1)
ax1.set_title(f"Training Paxillin IOU Histogram\nMean IoU: {mean_train_label_iou:.2f}")
ax1.set_ylabel("")
ax1.set_xlim([0, None])

sns.histplot(val_ious, bins=10, kde=True, ax=ax2)
ax2.set_title(f"Validation Paxillin IOU Histogram\nMean IoU: {mean_val_label_iou:.2f}")
ax2.set_ylabel("")
ax2.set_xlim([0, None])

plt.tight_layout()
plt.savefig("docs/label_ious.png")

mean_train_accuracy = np.mean(train_accuracy)
mean_val_accuracy = np.mean(val_accuracy)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(train_accuracy, bins=10, kde=True, ax=ax1)
ax1.set_title(f"Training Accuracy Histogram\nMean Accuracy: {mean_train_accuracy:.2f}")
ax1.set_ylabel("")
ax1.set_xlim([0, None])

sns.histplot(val_accuracy, bins=10, kde=True, ax=ax2)
ax2.set_title(f"Validation Accuracy Histogram\nMean Accuracy: {mean_val_accuracy:.2f}")
ax2.set_ylabel("")
ax2.set_xlim([0, None])

plt.tight_layout()
plt.savefig("docs/overall_accuracy.png")

colormap = np.array(config["COLORMAP"])
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

val_images = [all_images[i] for i in remaining_indices]
val_masks = [all_masks[i] for i in remaining_indices]

# Remove PNG files in predictions folder if they exist
predictions_folder = "docs/predictions"
for file in os.listdir(predictions_folder):
    if file.endswith(".png"):
        os.remove(os.path.join(predictions_folder, file))

plot_predictions(val_images, val_masks, colormap, model)
create_pdf([os.path.join("docs/predictions", f) for f in os.listdir("docs/predictions")], "docs/predictions.pdf")

classes = ["cortical_actin", "focal_adhesions", "lamellipodin", "filopodia"]
data = []

for image_path, mask_path in zip(sorted(val_images), sorted(val_masks)):
    image_tensor = read_image(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    prediction_mask = infer(image_tensor=image_tensor, model=model)

    for class_id, class_label in enumerate(classes, start=1):
        gt_fraction = np.sum(mask == class_id) / np.sum(mask != 0)
        pred_fraction = np.sum(prediction_mask == class_id) / np.sum(prediction_mask != 0)
        data.append(["Ground Truth", class_label, gt_fraction])
        data.append(["Predicted", class_label, pred_fraction])

df = pd.DataFrame(data, columns=["Type", "Class", "Fraction"])
df["Class"] = df["Class"].astype("category")

palette = config["PALETTE"]

plt.figure(figsize=(16, 8))
ax = sns.boxplot(x="Class", y="Fraction", hue="Type", data=df, dodge=True, palette=palette)
plt.xlabel("Class")
plt.ylabel("Fraction")
plt.title("Box Plot of Fraction of Each Class for Predicted vs Ground Truth")
plt.legend(title="Class")

# Perform t-tests and annotate p-values
for class_label in classes:
    gt_fractions = df[(df["Type"] == "Ground Truth") & (df["Class"] == class_label)]["Fraction"]
    pred_fractions = df[(df["Type"] == "Predicted") & (df["Class"] == class_label)]["Fraction"]
    t_stat, p_value = ttest_ind(gt_fractions, pred_fractions)

    x1, x2 = classes.index(class_label) - 0.2, classes.index(class_label) + 0.2
    y, h, col = max(max(gt_fractions), max(pred_fractions)) + 0.05, 0.02, "k"
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) / 2, y + h, f"p={p_value:.3f}", ha="center", va="bottom", color=col)

plt.savefig("docs/class_fraction_boxplot_with_significance_val.png")
plt.show()
