# face_recognition_pkg/utils/face_recognizer.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

class FaceRecognizerNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognizerNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 32 * 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class FaceRecognizer:
    def __init__(self, model_path=None, data_dir=None):
        self.data_dir = data_dir or "resource/training_data"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.people = self._load_people()
        self.model = FaceRecognizerNN(len(self.people)).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def _load_people(self):
        """Load list of people from training data directory"""
        if not os.path.exists(self.data_dir):
            return []
        return sorted([d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))])

    def preprocess_face(self, face_img):
        """Preprocess face image for neural network input"""
        face_img = cv2.resize(face_img, (256, 256))
        face_img = torch.from_numpy(face_img.transpose(2, 0, 1)).float()
        face_img = face_img.unsqueeze(0)
        return face_img.to(self.device)

    def recognize_face(self, face_img):
        """
        Recognize a face and return the person's name
        
        Args:
            face_img (np.array): Face image crop
            
        Returns:
            str: Name of the recognized person
        """
        if not self.people:
            return "Unknown"

        with torch.no_grad():
            processed_face = self.preprocess_face(face_img)
            outputs = self.model(processed_face)
            _, predicted = torch.max(outputs, 1)
            
            # Get confidence score
            confidence = F.softmax(outputs, dim=1).max().item()
            
            if confidence > 0.8:  # Confidence threshold
                return self.people[predicted.item()]
            return "Unknown"

    def store_recognition(self, name, face_img):
        """
        Store recognized face with timestamp
        
        Args:
            name (str): Name of the recognized person
            face_img (np.array): Face image
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("resource/recognitions", name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{timestamp}.jpg")
        cv2.imwrite(output_path, face_img)

    def train(self, train_loader, num_epochs=10):
        """
        Train the face recognition model
        
        Args:
            train_loader: PyTorch DataLoader with training data
            num_epochs (int): Number of training epochs
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
        
        self.model.eval()

    def save_model(self, path):
        """Save the neural network model"""
        torch.save(self.model.state_dict(), path)