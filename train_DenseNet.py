import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
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
    IMG_COLOR = 1
    BATCH_SIZE = 32
    INPUT_SHAPE = (IMG_HEIGHT,IMG_WIDTH,IMG_COLOR)
    NUM_CLASSES = 24+1


    def __init__(self, data_csv_dir="sign_mnist_train.csv",epochs=5):
        """
        :param data_csv_dir: file containing image path and label associated
        :param data_dir: directory of the data
        :param epochs: number of epochs for the training
        """
        self.epochs = epochs
        train_df = pd.read_csv(data_csv_dir)
        self.y_train = train_df['label']
        del train_df['label']
        self.x_train = train_df.values
        self.x_train = self.x_train / 255
        self.x_train = self.x_train.reshape(-1, 28, 28, 1)

        print(len(np.unique(self.y_train)))

        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.NUM_CLASSES)

        test_df = pd.read_csv("sign_mnist_test.csv")
        self.y_test = test_df['label']
        del test_df['label']

        self.x_test = test_df.values
        self.x_test = self.x_test / 255
        self.x_test = self.x_test.reshape(-1, 28, 28, 1)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.NUM_CLASSES)
        
        
        self.model = SignLanguageClassifier._load_model()
  

    
    def fit(self, folder):
        """Fit the model using the data in the selected directory"""
  
        train_set,valid_set = self._gen_data()
        
        # callback object to save weights during the training
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, "weights-{epoch:03d}.ckpt"),
            save_weights_only=True,
            verbose=1,
        )

        # Fit the model
        history = self.model.fit(
            train_set,
            epochs=self.epochs,
            callbacks=[cp_callback],
            validation_data=valid_set,
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

    def _gen_data(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.x_train, self.y_train)
        
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=10,
            zoom_range=0.1,  # Randomly zoom image
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False  # randomly flip images
           )  
        train_generator = datagen.flow(X_train,y_train,batch_size=self.BATCH_SIZE)
        valid_generator = datagen.flow(X_valid,y_valid,batch_size=self.BATCH_SIZE)
        
        return train_generator, valid_generator


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
