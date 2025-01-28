# src/face_recognition_pkg/setup.py

from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'face_recognition_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Face recognition package using ROS2 and neural networks',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'face_recognition_node = face_recognition_pkg.face_recognition_node:main',
        ],
    },
)