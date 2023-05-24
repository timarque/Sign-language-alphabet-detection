import sys
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QApplication,
    QLabel,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QDialog,
)
from PyQt5 import QtCore, QtGui
from skimage import io, transform
from Conv_operation import Transformations
import pandas as pd
from tensorflow import keras
import shutil
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("Sign language recognition")
        self.setFixedSize(600, 500)
        centralwidget = QWidget(self)
        PredictTab(centralwidget)
        self.setCentralWidget(centralwidget)


class PredictTab(QWidget):
    def __init__(self, parent):
        super(PredictTab, self).__init__(parent)
        self.setFixedSize(600, 500)
        self.imgPath = []
        self.imgIndex = 0
        self.predictions = []
        self.cnn = None
        self.kernel_name = None
        self.n_layers = 1
        mainLayout = QVBoxLayout(self)

        self.imgLabel = QLabel()
        self.imgLabel.setStyleSheet(
            "background-color: lightgrey; border: 1px solid gray;"
        )
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.prevButton = QPushButton("<")
        self.prevButton.setMaximumWidth(50)
        self.prevButton.setEnabled(False)
        self.nextButton = QPushButton(">")
        self.nextButton.setMaximumWidth(50)
        self.nextButton.setEnabled(False)
        self.predLabel = QLabel("None")
        self.predLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.predLabel.setFixedWidth(300)
        self.predLabel.setFixedHeight(20)
        hWidget1 = QWidget(self)
        hWidget1.setFixedHeight(20)
        hLayout1 = QHBoxLayout(hWidget1)
        hLayout1.setContentsMargins(0, 0, 0, 0)
        hWidget2 = QWidget(self)
        hWidget2.setFixedHeight(25)
        hLayout2 = QHBoxLayout(hWidget2)
        hLayout2.setContentsMargins(0, 0, 0, 0)
        hWidget3 = QWidget(self)
        hWidget3.setFixedHeight(25)
        hLayout3 = QHBoxLayout(hWidget3)
        hLayout3.setContentsMargins(0, 0, 0, 0)
        hWidget4 = QWidget(self)
        hWidget4.setFixedHeight(25)
        hLayout4 = QHBoxLayout(hWidget4)
        hLayout4.setContentsMargins(0, 0, 0, 0)
        # hWidget.setStyleSheet("border: 1px solid red; padding: 0 0 0 0; margin: 0px;")

        loadButton = QPushButton("Select picture(s)")
        modelButton = QPushButton("Select model (none)")
        predButton = QPushButton("Predict")
        loadButton.clicked.connect(self.loadImg)
        self.prevButton.clicked.connect(self.prevImg)
        self.nextButton.clicked.connect(self.nextImg)
        modelButton.clicked.connect(lambda: self.selectedModel(modelButton))
        predButton.clicked.connect(self.predict)

        mainLayout.addWidget(self.imgLabel)
        hLayout1.addWidget(self.prevButton)
        hLayout1.addWidget(self.predLabel)
        hLayout1.addWidget(self.nextButton)
        hLayout2.addWidget(loadButton)
        hLayout2.addWidget(modelButton)
        hLayout3.addWidget(predButton)
        mainLayout.addWidget(hWidget1)
        mainLayout.addWidget(hWidget2)
        mainLayout.addWidget(hWidget3)
        mainLayout.addWidget(hWidget4)

    def loadImg(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("Select an image")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            self.imgPath = [str(i) for i in dialog.selectedFiles()]
            self.predictions = [None for i in range(len(self.imgPath))]
            self.imgIndex = 0
            print("Selection:")
            for i in self.imgPath:
                print(i)
            self.updatePixmap(self.imgPath[self.imgIndex])
            self.prevButton.setEnabled(False)
            if len(self.imgPath) > 1:
                self.nextButton.setEnabled(True)
            elif len(self.imgPath) == 1:
                self.nextButton.setEnabled(False)
            self.updatePixmap(self.imgPath[self.imgIndex])
            # if self.cnn is not None:
            # self.predict()

    def updatePixmap(self, path, pred=1000):
        outputlist = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J","K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
        self.imgLabel.setPixmap(QtGui.QPixmap(path).scaled(500, 500))
        # self.imgLabel.setScaledContents(True)
        self.predLabel.setText(str(self.predictions[self.imgIndex]))
        if type(pred) == np.ndarray:
            letter = 0
            predicted = 0
            for i in range(len(pred[0])):
                if pred[0][i] > predicted:
                    letter = i
                    predicted = pred[0][i]
            self.predLabel.setText(
                "I think it's the letter '" + outputlist[letter] + "' !"
            )
        else:
            self.predLabel.setText("I don't know yet ")

    def predict(self):
        if len(self.imgPath) > 0 and self.cnn is not None:

            actual_image = Image.open(self.imgPath[self.imgIndex])
            resized = actual_image.resize((28,28))
            image = Image.new('L', (28, 28), color=0)
            padding_left = int((28 - resized.width) / 2)
            padding_top = int((28 - resized.height) / 2)
            image.paste(resized, (padding_left, padding_top))
            array = np.array(image)
            array = array / 255
            array = np.reshape(array, (-1, 28, 28, 1))

            img = transform.resize(
                io.imread(self.imgPath[self.imgIndex]),
                (28, 28),
            )
            img = img.reshape(-1, 28, 28, 1)

            self.predictions[self.imgIndex] = self.cnn.predict(array, 1)

            self.updatePixmap(
                self.imgPath[self.imgIndex], self.predictions[self.imgIndex]
            )

        else:
            QMessageBox(
                QMessageBox.Warning,
                "Error",
                "Please select an image and a neural network model before making prediction",
            ).exec_()

    def nextImg(self):
        self.imgIndex += 1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == len(self.imgPath) - 1:
            self.nextButton.setEnabled(False)
        else:
            self.nextButton.setEnabled(True)
        self.prevButton.setEnabled(True)

    def prevImg(self):
        self.imgIndex -= 1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == 0:
            self.prevButton.setEnabled(False)
        else:
            self.prevButton.setEnabled(True)
        self.nextButton.setEnabled(True)

    def selectedModel(self, btn):
        win = ModelWindow()
        if win.exec_():
            name, self.cnn = win.getModel()
            btn.setText(f"Select model ({name})")




class ModelWindow(QDialog):
    def __init__(self):
        super(ModelWindow, self).__init__()
        self.setWindowTitle("Model selection")
        self.model = None
        self.name = None
        mainLayout = QVBoxLayout(self)
        hWidget = QWidget()
        hLayout = QHBoxLayout(hWidget)
        text = QLabel("Select a neural network model: ")
        list = QListWidget()
        self.select = QPushButton("Select")
        self.select.clicked.connect(
            lambda: self.ok_pressed(list.currentItem().text())
        )
        self.delete = QPushButton("Delete")
        self.delete.clicked.connect(lambda: self.delete_pressed(list))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel_pressed)

        dir = [
            name
            for name in os.listdir(".")
            if os.path.isdir(name) and name.startswith("model_")
        ]
        if len(dir) > 0:
            list.addItems(["_".join(name.split("_")[1:]) for name in dir])
        else:
            self.checkCount(list)

        mainLayout.addWidget(text)
        mainLayout.addWidget(list)
        hLayout.addWidget(self.select)
        hLayout.addWidget(self.delete)
        hLayout.addWidget(cancel)
        hLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(hWidget)
        self.setLayout(mainLayout)

    def checkCount(self, list):
        if list.count() == 0:
            list.addItems(["No models found"])
            list.setEnabled(False)
            self.select.setEnabled(False)
            self.delete.setEnabled(False)

    def getModel(self):
        return (self.name, self.model)

    def ok_pressed(self, selected):
        print(selected, "selected")
        try:
            self.model = keras.models.load_model("model_" + selected)
            self.name = selected
        except:
            print("Cannot load model")
        self.accept()

    def delete_pressed(self, list):
        shutil.rmtree("model_" + list.currentItem().text())
        list.takeItem(list.currentRow())
        self.checkCount(list)

    def cancel_pressed(self):
        self.reject()



def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
