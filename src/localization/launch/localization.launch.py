from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time"
    )

    ekf_node = Node(
        package="localization",
        executable="ekf_node",
        name="ekf_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )


    measurement_node = Node(
        package="localization",
        executable="measurement_node",
        name="measurement_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    prediction_node = Node(
        package="localization",
        executable="prediction_node",
        name="prediction_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    initinal_pos = Node(
        package="localization",
        executable="initinal_pos",
        name="initinal_pos",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        declare_sim_time,
        prediction_node,
        measurement_node,
        ekf_node,
        initinal_pos,
    ])
