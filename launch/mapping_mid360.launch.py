import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # 获取包路径
    package_path = get_package_share_directory("fast_lio")

    # 定义配置文件路径
    config_file_lidar = os.path.join(package_path, "config", "mid360.yaml")
    config_file_camera = os.path.join(
        package_path, "config", "camera_pinhole_mid360.yaml"
    )
    rviz_config_file = os.path.join(
        package_path, "rviz", "map_updater.rviz"
    )  # 使用现有的或新建一个

    # 声明启动参数
    rviz_use = LaunchConfiguration("rviz")
    declare_rviz_cmd = DeclareLaunchArgument(
        "rviz", default_value="true", description="Use RViz to monitor results"
    )

    # 1. FAST-LIO 节点
    fast_lio_node = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="laserMapping",
        output="screen",
        parameters=[
            config_file_lidar,  # 加载雷达/IMU/算法参数
            config_file_camera,  # 加载相机参数
            {"use_sim_time": False},
        ],
    )

    # 2. RViz2 节点
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(rviz_use),
        output="log",
    )

    # 3. 图像解压节点 (Image Transport Republish)
    # 对应 ROS1: args="compressed in:=... raw out:=..."
    # ROS2 语法: republish compressed raw --ros-args -r in/compressed:=<topic> -r out:=<topic>
    image_republish_node = Node(
        package="image_transport",
        executable="republish",
        name="image_republish",
        arguments=["compressed", "raw"],
        remappings=[
            ("in/compressed", "/camera/camera/color/image_raw/compressed"),
            ("out", "/camera/camera/color/image_raw"),
        ],
        output="screen",
    )

    # 创建 LaunchDescription
    ld = LaunchDescription()
    ld.add_action(declare_rviz_cmd)
    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)
    ld.add_action(image_republish_node)

    return ld
