# src/face_recognition_pkg/launch/face_recognition.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Topic for camera input'
        ),
        
        Node(
            package='face_recognition_pkg',
            executable='face_recognition_node',
            name='face_recognition_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'camera_topic': LaunchConfiguration('camera_topic'),
                'recognition_threshold': 0.8,
                'training_images_required': 20
            }]
        )
    ])