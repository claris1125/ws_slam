from setuptools import setup
from glob import glob
import os

package_name = 'semantic_mapper'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='atoz',
    maintainer_email='atoz@example.com',
    description='LiDAR/Depth semantic mapping (fusion + YOLO)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_depth_mapper_node = semantic_mapper.yolo_depth_mapper:main',
        ],
    },
)
