import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("robot_description")

    world_file = os.path.join(pkg_share, "world", "depot.sdf")
    urdf_file = os.path.join(pkg_share, "src", "description", "robot.urdf")
    rviz_config = os.path.join(pkg_share, "config", "config.rviz")

    map_yaml = os.path.join(pkg_share, "config", "depot.yaml")
    amcl_yaml = os.path.join(pkg_share, "config", "amcl.yaml")
    gz_bridge_yaml = os.path.join(pkg_share, "config", "gz_bridge.yaml")

    use_sim_time = True

    with open(urdf_file, "r") as f:
        robot_description = f.read()

    gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=":".join(
            [
                os.path.join(pkg_share, "world"),
                str(Path(pkg_share).parent.resolve()),
            ]
        ),
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": ["-r -v 4 ", world_file]}.items(),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[
            {
                "config_file": gz_bridge_yaml,
                "qos_overrides./tf_static.publisher.durability": "transient_local",
            }
        ],
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name",
            "robot",
            "-topic",
            "/robot_description",
            "-x",
            "-4",
            "-y",
            "-2",
            "-z",
            "0.1",
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_description": robot_description},
        ],
    )

    map_to_odom_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"yaml_filename": map_yaml},
        ],
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            amcl_yaml,
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            {"node_names": ["map_server", "amcl"]},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    frame_id_converter = Node(
        package="robot_description",
        executable="frame_id_converter_node",
        name="frame_id_converter_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    cmd_vel = Node(
        package="robot_description",
        executable="cmd_vel",
        name="cmd_vel",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )


    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="True",
                description="Use simulation clock if true",
            ),
            DeclareLaunchArgument(
                "urdf_file",
                default_value=urdf_file,
                description="URDF file path",
            ),
            DeclareLaunchArgument(
                "use_robot_state_pub",
                default_value="True",
                description="Whether to start the robot state publisher",
            ),
            gz_resource_path,
            gz_sim,
            bridge,
            robot_state_publisher,
            spawn_entity,
            rviz,
            frame_id_converter,
            cmd_vel,
            map_server,
            amcl,
            lifecycle_manager,
            map_to_odom_static_tf,
        ]
    )
