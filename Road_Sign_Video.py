from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
import numpy as np
import cv2
import os

classs = {0: "Speed limit (20km/h)",
          1: "Speed limit (30km/h)",
          2: "Speed limit (50km/h)",
          3: "Speed limit (60km/h)",
          4: "Speed limit (70km/h)",
          5: "Speed limit (80km/h)",
          6: "End of speed limit (80km/h)",
          7: "Speed limit (100km/h)",
          8: "Speed limit (120km/h)",
          9: "No passing",
          10: "No passing veh over 3.5 tons",
          11: "Right-of-way at intersection",
          12: "Priority road",
          13: "Yield",
          14: "Stop",
          15: "No vehicles",
          16: "Veh > 3.5 tons prohibited",
          17: "No entry",
          18: "General caution",
          19: "Dangerous curve left",
          20: "Dangerous curve right",
          21: "Double curve",
          22: "Bumpy road",
          23: "Slippery road",
          24: "Road narrows on the right",
          25: "Road work",
          26: "Traffic signals",
          27: "Pedestrians",
          28: "Children crossing",
          29: "Bicycles crossing",
          30: "Beware of ice/snow",
          31: "Wild animals crossing",
          32: "End speed + passing limits",
          33: "Turn right ahead",
          34: "Turn left ahead",
          35: "Ahead only",
          36: "Go straight or right",
          37: "Go straight or left",
          38: "Keep right",
          39: "Keep left",
          40: "Roundabout mandatory",
          41: "End of no passing",
          42: "End no passing veh > 3.5 tons"}

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(100, 50, 600, 400))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(350, 470, 111, 16))
        self.label.setObjectName("label")
        self.resultLbl = QtWidgets.QLabel(self.centralwidget)
        self.resultLbl.setGeometry(QtCore.QRect(150, 490, 500, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.resultLbl.setFont(font)
        self.resultLbl.setObjectName("resultLbl")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Load the trained model
        self.model = load_model('Road_sign_reco_model.h5')

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)  # Update frame every 60 milliseconds

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Road Sign Detection"))
        self.label.setText(_translate("MainWindow", "Result:"))

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(convert_to_qt_format)
            self.imageLbl.setPixmap(pixmap)
            self.detect_sign(frame)

    def detect_sign(self, frame):
        # Resize frame to match model input size
        resized_frame = cv2.resize(frame, (30, 30))
        # Preprocess the frame for prediction
        x = np.expand_dims(resized_frame, axis=0)
        # Predict class probabilities
        preds = self.model.predict(x)
        # Get predicted class label
        pred_class = np.argmax(preds)
        # Display result
        result_text = classs[pred_class]
        self.resultLbl.setText(result_text)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
