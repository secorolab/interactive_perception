import sys 
from setuptools import find_packages, setup

package_name = 'robot_controller2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['action/MyAction.action']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'networkx', 'mapbox-earcut', 'core-algorithm', 'PyYAML'],
    zip_safe=True,
    maintainer='linux',
    maintainer_email='linux@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Visualizer = robot_controller2.Visualizer:main',
            'GT_Visualizer = robot_controller2.GT_Visualizer:main',
            'Reasoner_refactored = robot_controller2.Reasoner_refactored:main',
            'test_ms = robot_controller2.test_motion_spec:main',
            'Templates = robot_controller2.Templates:main',
            'Util = robot_controller2.Util:main',
            'Evaluation = robot_controller2.Evaluation:main',
        ],
    },
    package_data={
        'robot_controller2': [
            'action/MyAction.action',
            'config/*.yaml',
            'resources/**/*',
        ],
    },
    options={
        'build_scripts': {
            'executable': sys.executable,
        },
    },
)
