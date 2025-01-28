import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# Change relative imports to absolute
try:
    from .utils.face_detector import FaceDetector
except ImportError as e:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from face_recognition_pkg.utils.face_detector import FaceDetector
try:
    from .utils.face_recognizer import FaceRecognizer
except ImportError as e:
    from face_recognition_pkg.utils.face_recognizer import FaceRecognizer
try:
    from .utils.data_collector import DataCollector
except ImportError as e:
    from face_recognition_pkg.utils.data_collector import DataCollector

class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize camera
        self.get_logger().info('Attempting to open camera...')
        # Try different camera indices
        for camera_index in range(4):
            self.get_logger().info(f'Trying camera index {camera_index}...')
            self.camera = cv2.VideoCapture(camera_index)
            if self.camera.isOpened():
                self.get_logger().info(f'Successfully opened camera {camera_index}')
                break
            else:
                self.camera.release()
        if not self.camera.isOpened():
            self.get_logger().error('Could not open camera')
            return
        self.get_logger().info('Camera opened successfully!')
            
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize components
        self.face_detector = FaceDetector(
            model_path='resource/models/face_detector.pth'
        )
        self.face_recognizer = FaceRecognizer(
            model_path='resource/models/face_recognizer.pth'
        )
        self.data_collector = DataCollector()
        
        # Create subscribers
        self.name_subscription = self.create_subscription(
            String,
            '/face_recognition/register_name',
            self.register_name_callback,
            10
        )
        
        # Create publishers
        self.result_publisher = self.create_publisher(
            String,
            '/face_recognition/result',
            10
        )
        
        self.processed_image_publisher = self.create_publisher(
            Image,
            '/face_recognition/processed_image',
            10
        )
        
        # State variables
        self.training_mode = False
        self.current_person_name = None
        self.training_count = 0
        self.required_training_images = 20
        
        # Create timer for camera capture
        self.create_timer(0.1, self.camera_callback)  # 10 FPS
        
        # Log device information
        self.get_logger().info(f'Using device: {self.face_detector.device}')
        self.get_logger().info('Face Recognition Node initialized')
    
    def camera_callback(self):
        """Process frames from camera"""
        try:
            ret, current_frame = self.camera.read()
            if not ret:
                self.get_logger().warn('Failed to capture frame')
                return
            self.get_logger().info('Frame captured successfully')
                
            # Detect faces in the frame
            faces = self.face_detector.detect_faces(current_frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                face_img = current_frame[y:y+h, x:x+w]
                
                if self.training_mode and self.current_person_name:
                    # Collect training data
                    if self.training_count < self.required_training_images:
                        self.data_collector.collect_training_data(
                            self.current_person_name, 
                            face_img
                        )
                        self.training_count += 1
                        
                        # Draw blue rectangle for training mode
                        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        if self.training_count >= self.required_training_images:
                            self.training_mode = False
                            self.get_logger().info(
                                f'Training completed for {self.current_person_name}'
                            )
                            # Trigger model retraining
                            self.retrain_model()
                else:
                    # Recognition mode
                    name = self.face_recognizer.recognize_face(face_img)
                    
                    # Store recognition with timestamp
                    if name != "Unknown":
                        self.face_recognizer.store_recognition(name, face_img)
                    
                    # Draw green rectangle for recognition mode
                    cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(current_frame, name, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Publish recognition result
                    result_msg = String()
                    result_msg.data = f"Recognized: {name}"
                    self.result_publisher.publish(result_msg)
            
            # Publish processed image
            try:
                processed_msg = self.bridge.cv2_to_imgmsg(current_frame, "bgr8")
                self.processed_image_publisher.publish(processed_msg)
            except Exception as e:
                self.get_logger().warn(f'Error publishing image: {str(e)}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

    def register_name_callback(self, msg):
        """Handle new person registration"""
        self.current_person_name = msg.data
        self.training_mode = True
        self.training_count = 0
        self.get_logger().info(f'Starting training for {self.current_person_name}')

    def retrain_model(self):
        """Retrain the face recognition model with new data"""
        try:
            # Prepare training data
            images, labels, label_map = self.data_collector.prepare_training_data()
            
            if len(images) == 0:
                self.get_logger().warn('No training data available')
                return
            
            # Convert to PyTorch tensors and create data loader
            from torch.utils.data import TensorDataset, DataLoader
            import torch
            
            images = torch.from_numpy(images.transpose(0, 3, 1, 2)).float()
            labels = torch.from_numpy(labels).long()
            dataset = TensorDataset(images, labels)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Train the model
            self.face_recognizer.train(train_loader)
            
            # Save the updated model
            self.face_recognizer.save_model('resource/models/face_recognizer.pth')
            
            self.get_logger().info('Model retraining completed')
            
        except Exception as e:
            self.get_logger().error(f'Error retraining model: {str(e)}')

    def __del__(self):
        """Clean up camera resources"""
        if hasattr(self, 'camera'):
            self.camera.release()

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()