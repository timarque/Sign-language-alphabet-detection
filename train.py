import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Activation
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import (
	Dense,
	Conv2D,
	MaxPooling2D,
	Flatten,
	Dropout,
)
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from pathlib import Path
import argparse

SAVE_DIR = "backup"  # Save directory for backup weights during the training


class DogCatClassifier:
	"""
	Image classifier for dog and cat pictures using Deep Learning
	Convolutionnal Neural Network
	"""

	IMG_HEIGHT = 128
	IMG_WIDTH = 128
	BATCH_SIZE = 32

	def __init__(self, data_dir="train", data_csv_dir="train.csv", epochs=50):
		"""
		:param data_dir: directory of the data
		:param epochs: number of epochs for the training
		"""
		self.epochs = epochs
		self.data_dir = data_dir

		# Load data and labels
		self.X = sorted(os.listdir(self.data_dir))  # Files names of the images
		self.y = np.empty(len(self.X), dtype=str)  # Labels
		self.y[np.char.startswith(self.X, "A")] = "A"
		self.y[np.char.startswith(self.X, "B")] = "B"
		self.y[np.char.startswith(self.X, "C")] = "C"
		self.y[np.char.startswith(self.X, "D")] = "D"
		self.y[np.char.startswith(self.X, "E")] = "E"
		self.y[np.char.startswith(self.X, "F")] = "F"
		self.y[np.char.startswith(self.X, "G")] = "G"
		self.y[np.char.startswith(self.X, "H")] = "H"
		self.y[np.char.startswith(self.X, "I")] = "I"
		self.y[np.char.startswith(self.X, "J")] = "J"
		self.y[np.char.startswith(self.X, "K")] = "K"
		self.y[np.char.startswith(self.X, "L")] = "L"
		self.y[np.char.startswith(self.X, "M")] = "M"
		self.y[np.char.startswith(self.X, "N")] = "N"
		self.y[np.char.startswith(self.X, "O")] = "O"
		self.y[np.char.startswith(self.X, "P")] = "P"
		self.y[np.char.startswith(self.X, "Q")] = "Q"
		self.y[np.char.startswith(self.X, "R")] = "R"
		self.y[np.char.startswith(self.X, "S")] = "S"
		self.y[np.char.startswith(self.X, "T")] = "T"
		self.y[np.char.startswith(self.X, "U")] = "U"
		self.y[np.char.startswith(self.X, "V")] = "V"
		self.y[np.char.startswith(self.X, "W")] = "W"
		self.y[np.char.startswith(self.X, "X")] = "X"
		self.y[np.char.startswith(self.X, "Y")] = "Y"
		self.y[np.char.startswith(self.X, "Z")] = "Z"

		self.data = pd.read_csv(data_csv_dir)
		self.X = np.array([cv2.resize(
			cv2.imread(os.path.join(self.data_dir, img_path)),
			(self.IMG_HEIGHT, self.IMG_WIDTH)) for img_path in self.data["file_name"]])
		# normalize the pixels
		self.X = self.X.astype('float32') / 255

		self.y = self.data['letter'].astype('category').cat.codes
		self.y = tf.keras.utils.to_categorical(self.y, 26)

		self.model = DogCatClassifier._load_model()

	def fit(self, folder):
		"""Fit the model using the data in the selected directory"""

		X_train, X_test, y_train, y_test = train_test_split(
			self.X, self.y, random_state=42)
		X_train, X_val, y_train, y_val = train_test_split(
			X_train, y_train, random_state=42)
		print(len(X_train))
		print(len(X_val))
		print(len(self.X) - len(X_train) - len(X_val))

		# callback object to save weights during the training
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(SAVE_DIR, "weights-{epoch:03d}.ckpt"),
			save_weights_only=True,
			verbose=1,
		)

		# Fit the model
		history = self.model.fit(
			X_train, y_train,
			epochs=self.epochs,
			callbacks=[cp_callback],
			validation_data=(X_val, y_val)
		)

		# Show the predictions on the testing set
		result = self.model.evaluate(
			X_test, y_test, batch_size=self.BATCH_SIZE)
		print(
			"Testing set evaluation:",
			dict(zip(self.model.metrics_names, result)),
		)

		# Save model information
		self.model.save(folder)

		# Plot training results
		epochs_range = range(self.epochs)

		# Accuracy in training and validation sets as the training goes
		acc = history.history["accuracy"]
		val_acc = history.history["val_accuracy"]
		plt.figure(figsize=(8, 4))
		plt.subplot(1, 2, 1)
		plt.plot(epochs_range, acc, label="Training Accuracy")
		plt.plot(epochs_range, val_acc, label="Validation Accuracy")
		plt.legend(loc="lower right")
		plt.title("Training and Validation Accuracy")

		# Loss in training and validation sets as the training goes
		loss = history.history["loss"]
		val_loss = history.history["val_loss"]
		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, loss, label="Training Loss")
		plt.plot(epochs_range, val_loss, label="Validation Loss")
		plt.legend(loc="upper right")
		plt.title("Training and Validation Loss")

		plt.savefig(os.path.join(SAVE_DIR, "results.png"))

	@classmethod
	def _load_model(cls):
		"""Build a CNN model for image classification"""
		model = Sequential()

		# 2D Convolutional layer
		model.add(
			Conv2D(
				128,  # Number of filters
				(3, 3),  # Padding size
				input_shape=(
					128,
					128,
					3,
				),  # Shape of the input images
				activation="relu",  # Output function of the neurons
				padding="same",
			)
		)  # Behaviour of the padding region near the borders
		# 2D Pooling layer to reduce image shape
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Transform 2D input shape into 1D shape
		model.add(Flatten())
		# Dense layer of fully connected neurons
		model.add(Dense(128, activation="relu"))
		# Dropout layer to reduce overfitting, the argument is the proportion of random neurons ignored in the training
		model.add(Dropout(0.2))
		# Output layer
		model.add(Dense(26, activation="softmax"))

		model.compile(
			loss="categorical_crossentropy",  # Loss function for binary classification
			optimizer=RMSprop(
				lr=1e-3
			),  # Optimizer function to update weights during the training
			metrics=["accuracy", "AUC"],
		)  # Metrics to monitor during training and testing

		# Print model summary
		model.summary()

		return model


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="CNN Trainer for the Cat or Dog app.")

	parser.add_argument(
		"-f",
		"--folder",
		type=str,
		help="Destination folder to save the model after training ends.",
		default="Custom",
	)
	args = parser.parse_args()

	if Path(f"model_{args.folder}").is_dir():
		print(
			f"Folder model_{args.folder} already exists do you want to overwrite ?")
		y = input('Type "Yes" or "No": ')
		if y != "Yes":
			print("Aborting.")
			sys.exit()

	clf = DogCatClassifier()
	clf.fit(Path(f"model_{args.folder}"))
