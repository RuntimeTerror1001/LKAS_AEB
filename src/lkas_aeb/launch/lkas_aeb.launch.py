import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    # Paths to package resources
    package_path = get_package_share_directory('lkas_aeb')
    carla_ros_bridge_path = get_package_share_directory('carla_ros_bridge')
    carla_spawn_objects_path = get_package_share_directory('carla_spawn_objects')
    waypoint_publisher_path = get_package_share_directory('carla_waypoint_publisher')
    rviz_config = os.path.join(package_path, 'config', 'rviz', 'lkas_aeb.rviz')

    # Launch CARLA ROS bridge
    carla_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(carla_ros_bridge_path
                         ,'carla_ros_bridge.launch.py')
        ),
        launch_arguments={
            'autopilot': 'False',
            'passive': 'False',
            'synchronous_mode': 'True',  
            'fixed_delta_seconds': '0.05',
        }.items()
    )

    ego_vehicle = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(carla_spawn_objects_path, 'carla_spawn_objects.launch.py')
        ),
        launch_arguments={
            'objects_definition_file':os.path.join(
                package_path, 'config', 'vehicle_setup.json'),
        }.items()
    )

    waypoint_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(waypoint_publisher_path, 'carla_waypoint_publisher.launch.py')
        )
    )

    # Launch ADAS nodes
    perception_node = Node(
        package = 'lkas_aeb',
        executable = 'perception_node',
        name = 'perception_node',
        output = 'screen' 
    )

    vehicle_control_node = Node(
        package= 'lkas_aeb',
        executable= 'control_node',
        name= 'control_node',
        output= 'screen'
    )

    new_perception_node = Node(
        package = 'lkas_aeb',
        executable = 'new_perception_node',
        name = 'new_perception_node',
        output = 'screen' 
    )

    new_vehicle_control_node = Node(
        package= 'lkas_aeb',
        executable= 'new_control_node',
        name= 'new_control_node',
        output= 'screen'
    )

    # Visualization Nodes
    map_publisher = Node(
        package= 'lkas_aeb',
        executable= 'map_publisher',
        name= 'map_publisher',
        output= 'screen'
    )

    vehicle_marker = Node(
        package= 'lkas_aeb',
        executable= 'carla_vehicle_marker',
        name= 'carla_vehicle_marker',
        output= 'screen'
    )

    viewer = Node(
        package= 'lkas_aeb',
        executable= 'viewer',
        name= 'viewer',
        output= 'screen'
    )

    # Spawn Traffic
    spawn_traffic = ExecuteProcess(
        cmd=['python3', os.path.join(package_path, 'scripts', 'spawn_traffic.py')],
        output='screen',
        name='spawn_traffic'
    )

    delayed_spawn_traffic = TimerAction(
        period = 20.0, 
        actions=[spawn_traffic]
    )

    # RViz visualization
    rviz_node = Node(
        package = 'rviz2',
        executable = 'rviz2',
        name = 'rviz2',
        arguments = ['-d', rviz_config] 
    )

    return LaunchDescription([
        carla_bridge,
        ego_vehicle,
        waypoint_launch,
        new_perception_node,
        new_vehicle_control_node,
        map_publisher,
        vehicle_marker, 
        delayed_spawn_traffic,
        viewer,
        rviz_node
    ])