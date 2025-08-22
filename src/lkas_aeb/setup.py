from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lkas_aeb'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config', 'params'), glob('config/params/*.yaml')),
        (os.path.join('share', package_name, 'config', 'rviz'), glob('config/rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.osm'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='redpaladin',
    maintainer_email='parthraj1001@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = lkas_aeb.nodes.perception_node:main',
            'new_perception_node = lkas_aeb.nodes.new_perception_node:main',
            'new_control_node = lkas_aeb.nodes.new_control_node:main',
            'control_node = lkas_aeb.nodes.control_node:main',
            'map_publisher = lkas_aeb.nodes.map_publisher:main',
            'carla_vehicle_marker = lkas_aeb.nodes.carla_vehicle_marker:main',
            'viewer = lkas_aeb.nodes.viewer:main'
        ],
    },
)
