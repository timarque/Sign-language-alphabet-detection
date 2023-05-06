import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from pathlib import Path
import argparse
import DenseNet_Architecture

SAVE_DIR = "backup"  # Save directory for backup weights during the training


class SignLanguageClassifier:
	"""
	Image classifier for American sign language pictures using Dense Convolutional
	Network (DenseNet)
	"""

	IMG_HEIGHT = 128
	IMG_WIDTH = 128
	IMG_COLOR = 3
	BATCH_SIZE = 32
	INPUT_SHAPE = (IMG_HEIGHT,IMG_WIDTH,IMG_COLOR)

	def __init__(self, data_csv_dir="train.csv",data_dir="train", epochs=1):
		"""
		:param data_dir: directory of the data
		:param epochs: number of epochs for the training
		"""
		self.epochs = epochs
		self.data_dir = data_dir
		self.data = pd.read_csv(self.data_csv_dir)
		self.num_classes = len(np.unique(self.data['letter']))

		# Load data and labels
		# read and resize the images
		self.X = np.array([cv2.resize(
	  						cv2.imread(os.path.join(self.data_dir,img_path)),
							(self.IMG_HEIGHT,self.IMG_WIDTH)) for img_path in self.data["file_name"]])
		# normalize the pixels
		self.X = self.X.astype('float32') / 255 
		self.y = self.data['letters'].astype('category').cat.codes
		self.y = tf.keras.utils.to_categorical(self.y,self.num_classes)
		

		self.model = SignLanguageClassifier._load_model()

	def fit(self, folder):
		"""Fit the model using the data in the selected directory"""
  
		# Split data into training+validation and testing sets
		X_train, X_test, y_train, y_test = train_test_split(
			self.X, self.y, random_state=42)
		X_train, X_val, y_train, y_val = train_test_split(
			X_train, y_train, random_state=42)

		# callback object to save weights during the training
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(SAVE_DIR, "weights-{epoch:03d}.ckpt"),
			save_weights_only=True,
			verbose=1,
		)

		# Fit the model
		history = self.model.fit(
			X_train,y_train,
			epochs=self.epochs,
			validation_data=(X_val,y_val),
			callbacks=[cp_callback],
		)

		# Show the predictions on the testing set
		result = self.model.evaluate(X_test,y_test, batch_size=self.BATCH_SIZE)
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
		"""Build a DenseNet model for image classification"""
		model = DenseNet_Architecture.DenseNet(cls.INPUT_SHAPE,cls.num_classes)

		model.complile(loss="categorical_crossentropy",
		               optimizer="adam", metrics=['accuracy', 'AUC'])

		  
		# Print model summary
		model.summary()

		return model


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="DenseNet Trainer for American Sign Language app.")

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

	clf = SignLanguageClassifier()
	clf.fit(Path(f"model_{args.folder}"))
