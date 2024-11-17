import cv2
import os
import sys
import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QMainWindow

class ImageViewer(QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Resizable, Zoomable Image Viewer")

        # Check if the image is grayscale or color
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels (RGB)
    
        # Convert the OpenCV image (BGR) to QImage (RGB)
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)

        # Set up the scene and view for displaying the image
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)

        # Add the image to the scene
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.scene.addItem(self.pixmap_item)

        # Fit the view to the image size initially
        self.view.setSceneRect(QRectF(self.pixmap_item.pixmap().rect()))
        self.view.setAlignment(Qt.AlignCenter)

        # Enable mouse wheel zooming
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setRenderHint(QPainter.Antialiasing)
        
        self.setCentralWidget(self.view)

    def wheelEvent(self, event):
        # Handle zooming using mouse wheel
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.view.scale(factor, factor)

def show_image(image_path):
    app = QApplication(sys.argv)
    viewer = ImageViewer(image_path)
    viewer.show()
    app.exec_()
    
    
def detect_blur_face(image, sharpness_threshold=100, view=False):
    # Load pre-trained face detection model (Haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pass 1: Without histogram equalization
    sharpness1, is_blurry1 = process_face_detection(gray, face_cascade, sharpness_threshold, view)

    # Pass 2: With histogram equalization
    gray_eq = cv2.equalizeHist(gray)
    sharpness2, is_blurry2 = process_face_detection(gray_eq, face_cascade, sharpness_threshold, view)
    if view:
        show_image(gray_eq)
    # Average sharpness
    avg_sharpness = (sharpness1 + sharpness2) / 2
    avg_blurry = is_blurry1 and is_blurry2  # Image is blurry if both passes consider it blurry

    return avg_sharpness, avg_blurry

def process_face_detection(gray_image, face_cascade, sharpness_threshold, view):
    if view:
        show_image(gray_image)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, classify image as blurry
    if len(faces) == 0:
        return 0, True  # No faces, considered blurry

    face_is_blurry = True
    sharpness_values = []

    for (x, y, w, h) in faces:
        # Crop the face region from the image
        face_roi = gray_image[y:y + h, x:x + w]

        # Measure sharpness using the variance of the Laplacian
        sharpness = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        sharpness_values.append(sharpness)

        # If the sharpness is above the threshold, the face is sharp
        if sharpness > sharpness_threshold:
            face_is_blurry = False

    # Return the maximum sharpness and whether the image is blurry
    max_sharpness = max(sharpness_values) if sharpness_values else 0
    return max_sharpness, face_is_blurry

def process_images_in_folder(folder_path, sharpness_threshold=100):
    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Process only image files (filter based on file extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {filename}")
                continue  # Skip this image if it cannot be loaded
            (orig_height, orig_width) = image.shape[:2]

            # Calculate scaling factors
            #scale = 2000/ orig_height
            
            # Resize image
            #new_width = int(orig_width * scale)
            #new_height = int(orig_height * scale)
            #resize = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # Check if the image is blurry or not
            sharpness, blurry = detect_blur_face(image, sharpness_threshold)
            if blurry:
                print(f"{filename}: The image is blurry. {sharpness}")
            else:
                print(f"{filename}: The image is not blurry. {sharpness}")

# Example usage
folder_path = 'filepath'  # Change this to the folder containing your images
process_images_in_folder(folder_path, sharpness_threshold=100)
