import sys
import os
import csv
import argparse

import cv2
from PySide6 import QtCore
from PySide6.QtCore import QDir, Qt, QRectF, QPoint, QPointF
from PySide6.QtGui import QImage, QKeyEvent, QPainter, QPixmap, QColor, QPen, QFont, QBrush, QTransform
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QWidget
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox
from PySide6.QtWidgets import QFileDialog, QGraphicsTextItem
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QCheckBox

parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

parser.add_argument('--match_dir', required=True, type=str, help="Match directory. Must be named with prefix 'match' following by an index.")
parser.add_argument('--video_path', type=str, default=None, help="Optional path to a video file. If provided, frames are read on demand via OpenCV instead of match_dir/frames.")

args = parser.parse_args()

MATCH_DIR = args.match_dir
FRAMES_DIR = os.path.join(MATCH_DIR, "frames")
CSV_DIR = os.path.join(MATCH_DIR, "Labels.csv")
VIDEO_PATH = args.video_path

if(not os.path.exists(MATCH_DIR)):
    print("\nERROR: The following directory does not exist: " + str(MATCH_DIR))
    exit(1)

def validDir(directory):
    _, suffix = os.path.split(directory)
    if not suffix.startswith('match'):  
        return False  
    number_part = suffix[5:] 
    return number_part.isdigit()

if(not (validDir(MATCH_DIR))):
    print("\nERROR: Specified export folder does not start with 'match' followed by an index.")
    exit(1)

if VIDEO_PATH is not None and not os.path.exists(VIDEO_PATH):
    print("\nERROR: The following video file does not exist: " + str(VIDEO_PATH))
    exit(1)

if(VIDEO_PATH is None and not os.path.exists(FRAMES_DIR)):
    print("\nERROR: The following directory does not exist: " + str(FRAMES_DIR))
    exit(1)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.view = QGraphicsView()
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setMouseTracking(True)
        self._apply_render_hints()
        
        self.view.wheelEvent = self.wheelEvent
        self.view.mousePressEvent = self.getPixelCoordinates
        
        self.images = []
        self.frameLabels = []
        self.video_path = VIDEO_PATH
        self.video_capture = None
        self.video_frame_count = 0
        self.frameIndex = 0
        self.pixelCoordinates = {}
        self.states = {}
        self.annotated = {}
        self.showCrosshair = True
        self.currentCenterPoint = self.view.mapToScene(self.view.viewport().rect().center()) 
        self.loadImages()

        self.pixmap = self.loadFramePixmap(self.frameIndex)
        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.pixmap)
        self.view.setScene(self.scene)

        self.zoomLevel = 1.0

        self.toggleStateButton = QPushButton("Toggle State", self)
        self.toggleStateButton.setFixedSize(200, 75)
        self.toggleStateButton.clicked.connect(self.toggleState)

        self.removePixelButton = QPushButton("Remove Pixel", self)
        self.removePixelButton.setFixedSize(200, 75)
        self.removePixelButton.clicked.connect(self.removePixel)

        self.removeFrameButton = QPushButton("Remove Frame", self)
        self.removeFrameButton.setFixedSize(200, 75)
        self.removeFrameButton.clicked.connect(self.removeFrame)

        self.saveResultsButton = QPushButton("Save Results", self)
        self.saveResultsButton.setFixedSize(200, 75)
        self.saveResultsButton.clicked.connect(self.saveResults)

        self.loadCsvButton = QPushButton("Load CSV", self)
        self.loadCsvButton.setFixedSize(200, 75)
        self.loadCsvButton.clicked.connect(self.loadResults)

        self.crosshairCheckbox = QCheckBox("Show Crosshair", self)
        self.crosshairCheckbox.setChecked(True)
        self.crosshairCheckbox.setFont(QFont("Arial", 20))
        self.crosshairCheckbox.stateChanged.connect(self.toggleCrosshairVisibility)

        self.visbilityText = QLabel()
        self.visbilityText.setStyleSheet("color: red;")
        self.visbilityText.setFont(QFont("Arial", 30))

        self.imageText = QLabel()
        self.imageText.setStyleSheet("color: black;")
        self.imageText.setFont(QFont("Arial", 30))

        self.annotatedText = QLabel()   
        self.annotatedText.setFont(QFont("Arial", 30))
        
        self.topLayout = QHBoxLayout()
        self.topLayout.addWidget(self.visbilityText)
        self.topLayout.addWidget(self.imageText)
        self.topLayout.addWidget(self.annotatedText)

        self.buttonLayout = QVBoxLayout()
        self.buttonLayout.addWidget(self.toggleStateButton)
        self.buttonLayout.addWidget(self.removePixelButton)
        self.buttonLayout.addWidget(self.removeFrameButton)
        self.buttonLayout.addWidget(self.saveResultsButton)
        self.buttonLayout.addWidget(self.loadCsvButton)
        self.buttonLayout.addWidget(self.crosshairCheckbox)
        self.buttonLayout.addStretch()

        self.bottomLayout = QHBoxLayout()
        self.bottomLayout.addWidget(self.view)
        self.bottomLayout.addLayout(self.buttonLayout)
        self.bottomLayout.setAlignment(self.buttonLayout, Qt.AlignRight)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addLayout(self.bottomLayout)

    
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

        self.showImage()

    def _apply_render_hints(self):
        render_hints = QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        render_hints |= QPainter.RenderHint.TextAntialiasing
        self.view.setRenderHints(render_hints)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0: 
            self.zoomLevel = self.zoomLevel * 1.2
            self.view.scale(1.2, 1.2)
        else:
            self.zoomLevel = self.zoomLevel / 1.2
            self.view.scale(1 / 1.2, 1 / 1.2)
        return True
        

    def loadImages(self):
        global FRAMES_DIR

        if self.video_path is not None:
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                print("\nERROR: Failed to open video: " + str(self.video_path))
                exit(1)

            self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.images = list(range(self.video_frame_count))
            self.frameLabels = [f"frame_{index}" for index in self.images]
        else:
            directory = QDir(FRAMES_DIR)
            directory.setNameFilters(["*.png"])
            directory.setSorting(QDir.Name)
            self.images = [directory.filePath(file) for file in directory.entryList()]
            self.images = sorted(self.images, key=lambda x: int(x.split('/')[-1].split(".")[0]))
            self.frameLabels = [path.split('/')[-1] for path in self.images]

        if len(self.images) == 0:
            print("\nERROR: No frames were found for annotation.")
            exit(1)

        for i in range (len(self.images)):
            self.annotated[i] = False

    def loadFramePixmap(self, frame_index):
        if self.video_path is None:
            return QPixmap(self.images[frame_index])

        source_frame_index = self.images[frame_index]
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
        success, frame = self.video_capture.read()
        if not success:
            raise RuntimeError(f"Failed to read frame {source_frame_index} from {self.video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def getFrameDisplayName(self, frame_index):
        return self.frameLabels[frame_index]

    def drawCrosshair(self, x, y):
        if not self.showCrosshair:
            return

        crosshair_pen = QPen(Qt.red)
        crosshair_pen.setWidth(2)
        arm_length = 18
        gap_size = 6

        self.view.scene().addLine(x - gap_size - arm_length, y, x - gap_size, y, crosshair_pen)
        self.view.scene().addLine(x + gap_size, y, x + gap_size + arm_length, y, crosshair_pen)
        self.view.scene().addLine(x, y - gap_size - arm_length, x, y - gap_size, crosshair_pen)
        self.view.scene().addLine(x, y + gap_size, x, y + gap_size + arm_length, crosshair_pen)

    def toggleCrosshairVisibility(self, state):
        self.showCrosshair = state == int(Qt.CheckState.Checked)
        self.showImage()

    def showImage(self):
        self._apply_render_hints()

        self.pixmap = self.loadFramePixmap(self.frameIndex)
        self.view.setScene(QGraphicsScene())
        self.view.scene().addPixmap(self.pixmap)
        self.view.fitInView(self.view.sceneRect(), Qt.KeepAspectRatio)
        self.view.scale(self.zoomLevel, self.zoomLevel)
        self.view.centerOn(self.currentCenterPoint)

        self.imageText.setText(self.getFrameDisplayName(self.frameIndex))

        if not self.frameIndex in self.states:
            self.states[self.frameIndex] = "VISIBLE"

        self.visbilityText.setText(self.states[self.frameIndex])

        if(self.frameIndex in self.pixelCoordinates or self.states[self.frameIndex] == "INVISIBLE"):
            self.annotatedText.setText("Annotated")
            self.annotatedText.setStyleSheet("color: green;")
        else:
            self.annotatedText.setText("Not Annotated")
            self.annotatedText.setStyleSheet("color: red;")


        if(self.frameIndex in self.pixelCoordinates):
            x, y = self.pixelCoordinates[self.frameIndex]
            self.drawCrosshair(x, y)
   
    def toggleState(self):
        if(self.frameIndex in self.states):
            if(self.states[self.frameIndex] == "VISIBLE"):
                self.states[self.frameIndex]  = "INVISIBLE"
            else:
                self.states[self.frameIndex]  = "VISIBLE"

        if(self.states[self.frameIndex]  == "INVISIBLE"):
            self.annotated[self.frameIndex] = True
        elif(self.states[self.frameIndex]  == "VISIBLE" and self.frameIndex not in self.pixelCoordinates):
            self.annotated[self.frameIndex] = False
        
        self.showImage()

    def removePixel(self):
        if self.frameIndex in self.pixelCoordinates:
            del self.pixelCoordinates[self.frameIndex]
            if(self.states[self.frameIndex] == "VISIBLE"):
                self.annotated[self.frameIndex] = False
            self.showImage()

    def removeFrame(self):
        print("Removed image: " + str(self.getFrameDisplayName(self.frameIndex)))

        if(self.frameIndex in self.pixelCoordinates):
            self.pixelCoordinates.pop(self.frameIndex)
            for key in list(self.pixelCoordinates.keys()):
                if key > self.frameIndex:
                    value = self.pixelCoordinates.pop(key)
                    self.pixelCoordinates[key-1] = value

        if(self.frameIndex in self.states):    
            self.states.pop(self.frameIndex)
            for key in list(self.states.keys()):
                if key > self.frameIndex:
                    value = self.states.pop(key)
                    self.states[key-1] = value

        if(self.frameIndex in self.annotated):    
            self.annotated.pop(self.frameIndex)
            for key in list(self.annotated.keys()):
                if key > self.frameIndex:
                    value = self.annotated.pop(key)
                    self.annotated[key-1] = value
        
        if self.video_path is None:
            os.remove(self.images[self.frameIndex])
        
        self.images.pop(self.frameIndex)
        self.frameLabels.pop(self.frameIndex)

        if len(self.images) == 0:
            print("All frames were removed.")
            self.close()
            return

        if(self.frameIndex >= len(self.images)):
            self.frameIndex = len(self.images) - 1

        self.showImage()

    def saveResults(self):
        global FRAMES_DIR

        allAnnotated = True

        for i in range(len(self.images)):
            if(not self.annotated[i]):
                allAnnotated = False
                print(str(self.getFrameDisplayName(i)) + " is not annotated.")

        if(allAnnotated):
            with open(CSV_DIR, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Visibility", "X", "Y"])
    
                data = []   
                for i in range(len(self.images)):
                    if(self.states.get(i, "VISIBLE") == "VISIBLE"):
                        visibility = 1
                        
                    else:
                        visibility = 0
                    
                    if(i in self.pixelCoordinates):
                        x_coord = int(self.pixelCoordinates[i][0])
                        y_coord = int(self.pixelCoordinates[i][1])
                    else:
                        x_coord = 0
                        y_coord = 0
                        
                    source_index = self.images[i] if self.video_path is not None else i
                    data.append([source_index, visibility, x_coord, y_coord])

                writer.writerows(data)
                print("Saved Results!")
            
            if self.video_path is None:
                for i in range(len(self.images)):
                    os.rename(str(self.images[i]), os.path.join(FRAMES_DIR, str(i) + ".png"))

            QApplication.instance().quit()

    def loadResults(self):
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Labels CSV",
            MATCH_DIR,
            "CSV Files (*.csv)"
        )

        if not csv_path:
            return

        frame_to_index = {}
        for index, frame_value in enumerate(self.images):
            source_index = frame_value if self.video_path is not None else index
            frame_to_index[source_index] = index

        loaded_rows = 0
        self.pixelCoordinates.clear()
        self.states.clear()
        self.annotated = {i: False for i in range(len(self.images))}

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    source_index = int(row["Frame"])
                    visibility = int(row["Visibility"])
                    x_coord = int(row["X"])
                    y_coord = int(row["Y"])
                except (KeyError, TypeError, ValueError):
                    continue

                if source_index not in frame_to_index:
                    continue

                frame_index = frame_to_index[source_index]
                self.states[frame_index] = "VISIBLE" if visibility == 1 else "INVISIBLE"

                if x_coord != 0 or y_coord != 0:
                    self.pixelCoordinates[frame_index] = (x_coord, y_coord)

                self.annotated[frame_index] = True
                loaded_rows += 1

        print(f"Loaded {loaded_rows} annotations from {csv_path}")
        self.showImage()

    def stepFrame(self, offset):
        if len(self.images) == 0:
            return

        new_index = max(0, min(len(self.images) - 1, self.frameIndex + offset))
        if new_index != self.frameIndex:
            self.frameIndex = new_index
            self.showImage()
      
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_A:
            self.stepFrame(-1)
        elif event.key() == Qt.Key_D:
            self.stepFrame(1)
        elif event.key() == Qt.Key_W:
            self.stepFrame(15)
        elif event.key() == Qt.Key_S:
            self.stepFrame(-15)

    def getPixelCoordinates(self, event):
        pos = event.pos()
        scene_pos = self.view.mapToScene(pos)

        if event.button() == Qt.MiddleButton:
            self.removePixel()
            return

        if event.button() != Qt.LeftButton:
            return

        self.pixelCoordinates[self.frameIndex] = (scene_pos.x(), scene_pos.y())
        self.annotated[self.frameIndex] = True
        
        self.currentCenterPoint = QPointF(scene_pos.x(), scene_pos.y())
        self.showImage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    #viewer.showFullScreen()
    sys.exit(app.exec())
