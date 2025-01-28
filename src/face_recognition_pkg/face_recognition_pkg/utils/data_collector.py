# face_recognition_pkg/utils/data_collector.py

import os
import cv2
import numpy as np
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="resource/training_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def collect_training_data(self, name, face_img):
        """
        Collect training data for a person
        
        Args:
            name (str): Name of the person
            face_img (np.array): Face image
        """
        person_dir = os.path.join(self.data_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(person_dir, f"{timestamp}.jpg")
        
        # Save the face image
        cv2.imwrite(image_path, face_img)

    def get_training_count(self, name):
        """
        Get the number of training images for a person
        
        Args:
            name (str): Name of the person
            
        Returns:
            int: Number of training images
        """
        person_dir = os.path.join(self.data_dir, name)
        if not os.path.exists(person_dir):
            return 0
        return len([f for f in os.listdir(person_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))])

    def prepare_training_data(self):
        """
        Prepare training data for model training
        
        Returns:
            tuple: (images, labels) arrays for training
        """
        images = []
        labels = []
        label_map = {}
        
        for idx, person in enumerate(sorted(os.listdir(self.data_dir))):
            person_dir = os.path.join(self.data_dir, person)
            if not os.path.isdir(person_dir):
                continue
                
            label_map[person] = idx
            
            for image_name in os.listdir(person_dir):
                if not image_name.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                # Preprocess image
                image = cv2.resize(image, (256, 256))
                images.append(image)
                labels.append(idx)
        
        return np.array(images), np.array(labels), label_map