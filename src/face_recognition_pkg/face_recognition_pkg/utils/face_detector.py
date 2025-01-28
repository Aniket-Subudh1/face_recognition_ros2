# face_recognition_pkg/utils/face_detector.py

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os  # Added missing import

class FaceDetectorNN(nn.Module):
    def __init__(self):
        super(FaceDetectorNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 4)  # Output 4 coordinates for bounding box

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FaceDetector:
    def __init__(self, model_path=None):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceDetectorNN().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def detect_faces(self, image):
        """
        Detect faces in the input image
        
        Args:
            image (np.array): Input image in BGR format
            
        Returns:
            list: List of face coordinates (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces

    def preprocess_image(self, image):
        """
        Preprocess image for neural network input
        """
        image = cv2.resize(image, (256, 256))
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0)
        return image.to(self.device)

    def detect_faces_nn(self, image):
        """
        Detect faces using neural network
        """
        with torch.no_grad():
            processed_image = self.preprocess_image(image)
            bbox = self.model(processed_image)
            bbox = bbox.cpu().numpy()[0]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        
        return [(x1, y1, x2-x1, y2-y1)]

    def save_model(self, path):
        """Save the neural network model"""
        torch.save(self.model.state_dict(), path)