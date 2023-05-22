import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import cv2
import DenseNet_Architecture

SAVE_DIR = "backup_denseNet_2"  # Save directory for backup weights during the training


class SignLanguageClassifier:
	"""
	Image classifier for American sign language letter in
 	form of Picture using Dense Convolutional Network (DenseNet)
	"""

	IMG_HEIGHT = 128
	IMG_WIDTH = 128
	IMG_COLOR = 3
	BATCH_SIZE = 32
	INPUT_SHAPE = (IMG_HEIGHT,IMG_WIDTH,IMG_COLOR)
	NUM_CLASSES = 26

	def __init__(self, data_train_csv="train_data.csv",data_dir="train", epochs=50):
		"""
		:param data_csv_dir: file containing image path and label associated
		:param data_dir: directory of the data
		:param epochs: number of epochs for the training
		"""
		self.epochs = epochs
		self.data_dir = data_dir
		self.data_train = pd.read_csv(data_train_csv)
		self.data_test = pd.read_csv("test_data.csv")
		self.data_valid = pd.read_csv("valid_data.csv")
		self.NUM_CLASSES = len(np.unique(self.data_train['letter']))

		# Load data
		self.X_train = self._read_csv_data("train",self.data_train)
		self.X_test = self._read_csv_data('test', self.data_test)
		self.X_valid = self._read_csv_data('valid', self.data_valid)
  
		# normalize the pixels
		self.X_train = self.X_train.astype('float32') / 255 
		self.X_test = self.X_test.astype('float32') / 255 
		self.X_valid = self.X_valid.astype('float32') / 255
		# one hot encoding of y train
		self.y_train = self.data_train['letter'].astype('category').cat.codes
		self.y_train = tf.keras.utils.to_categorical(self.y_train,self.NUM_CLASSES)
  
		# one hot encoding of y test
		self.y_test = self.data_test['letter'].astype('category').cat.codes
		self.y_test = tf.keras.utils.to_categorical(self.y_test, self.NUM_CLASSES)
  
		# one hot encoding of y valid
		self.y_valid = self.data_valid['letter'].astype('category').cat.codes
		self.y_valid = tf.keras.utils.to_categorical(self.y_valid, self.NUM_CLASSES)
		
		self.model = SignLanguageClassifier._load_model()

	def _read_csv_data(self, dir,data_csv):
		# read and resize the images
		X = np.array([cv2.resize(
				cv2.imread(os.path.join(dir, img_path)),
				(self.IMG_HEIGHT, self.IMG_WIDTH)) for img_path in data_csv["file_name"]])
		return X

	def fit(self, folder):
		"""Fit the model using the data in the selected directory"""
  
		# Split data into training+validation and testing sets
		# X_train, X_test, y_train, y_test = train_test_split(
		# 	self.X_train, self.y_train, random_state=42)
		# X_train, X_val, y_train, y_val = train_test_split(
		# 	X_train, y_train, random_state=42)

		# callback object to save weights during the training
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(SAVE_DIR, "weights-{epoch:03d}.ckpt"),
			save_weights_only=True,
			verbose=1,
		)

		# Fit the model
		history = self.model.fit(
			self.X_train,self.y_train,
			epochs=self.epochs,
			validation_data=(self.X_valid ,self.y_valid),
			callbacks=[cp_callback],
		)

		# Show the predictions on the testing set
		result = self.model.evaluate(self.X_test,self.y_test, batch_size=self.BATCH_SIZE)
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
		plt.title("Training and Validation Accuracy for DenseNet")

		# Loss in training and validation sets as the training goes
		loss = history.history["loss"]
		val_loss = history.history["val_loss"]
		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, loss, label="Training Loss")
		plt.plot(epochs_range, val_loss, label="Validation Loss")
		plt.legend(loc="upper right")
		plt.title("Training and Validation Loss for DenseNet")

		plt.savefig(os.path.join(SAVE_DIR, "results.png"))

	@classmethod
	def _load_model(cls):
		"""Build a DenseNet model for image classification"""
		model = DenseNet_Architecture.DenseNet(cls.INPUT_SHAPE,cls.NUM_CLASSES)

		model.compile(loss="categorical_crossentropy",
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
