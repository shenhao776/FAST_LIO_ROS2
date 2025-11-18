#ifndef LASER_MAPPING_EXTRA_HPP
#define LASER_MAPPING_EXTRA_HPP

// System & Std
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ROS 2
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

// PCL
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/sac_segmentation.h>

// Eigen & Sophus
#include <Eigen/Core>
#include <sophus/se3.hpp>

// Local Types
#include "common_lib.h"

// --- Forward Declarations ---
// Assuming MapUpdater is defined in a separate file ("map_updater.h")
// that you might port later. We declare a placeholder here.
class MapUpdater;

// --- Data Structures ---

// Data structure for Keyframes (inferred from usage)
struct KeyframeData {
  double time;
  Sophus::SE3d pose;  // Using Sophus for SE3 as in the new code
  PointCloudXYZI::Ptr cloud;

  KeyframeData() : time(0), cloud(new PointCloudXYZI()) {}
};

// --- Extended Members Class ---
// This struct holds all the new variables introduced in the modified LIV_MAPPER
// Adapt this into your LaserMappingNode as a member: LaserMappingExtra ext_;
struct LaserMappingExtra {
  // Constructor to initialize pointers
  LaserMappingExtra() {
    original_map_visual_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    deleted_points_visual_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    // map_updater_ should be initialized when needed
  }

  // ============ Map Management ============

  // Voxel Map Index Type (Placeholder for MapUpdater::VoxelMapIndexType)
  // You might need to adjust this type based on map_updater.h
  using VoxelMapIndexType =
      std::unordered_map<VOXEL_LOCATION, int, std::hash<VOXEL_LOCATION>>;
  VoxelMapIndexType voxel_map_index;

  // Keyframes
  std::vector<KeyframeData> original_map_keyframes_;  // Read-only loaded map
  std::vector<KeyframeData>
      new_keyframes_all_;  // All keyframes aligned in this run
  std::vector<KeyframeData> new_keyframes_;  // Newest unaligned keyframes

  // Visuals for Rviz
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr original_map_visual_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr deleted_points_visual_;

  // Map Updater Component
  std::unique_ptr<MapUpdater> map_updater_;

  // ROS 2 Publishers for Map Updates
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_loaded_map;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_original_map_kept_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_newly_added_map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_final_updated_map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_deleted_points_;

  // ============ PCD Saving Logic ============

  V3D last_pcd_save_pos_ = V3D::Zero();
  double pcd_save_distance_thresh_ = 0.2;
  bool is_first_pcd_saved_ = false;

  // ============ Relocalization / Initial Alignment ============

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      sub_initial_pose_;
  std::mutex mtx_initial_pose_;
  bool initial_pose_received_ = false;
  bool initial_align_finished_ = false;
  Sophus::SE3d
      initial_pose_;  // Pose received from /initialpose (2D Pose Estimate)

  int relocalization_min_feature_num = 50;  // Default value

  // ============ Periodic Alignment Parameters ============

  int motion_keyframe_count_ = 0;
  int periodic_align_interval_ = 20;
  double periodic_align_min_dist_ = 0.3;
  Sophus::SE3d last_kf_pose_;

  // ============ Map Update Parameters ============

  std::string map_data_path_;
  bool is_update_mode_ = false;
  double keyframe_search_radius_ = 2.0;    // Example default
  double point_replacement_radius_ = 0.1;  // Example default

  // ============ Automation Mode ============

  bool auto_mode_enabled_ = false;
  double message_timeout_ = 5.0;  // Seconds
  bool auto_save_triggered_ = false;
  rclcpp::Time last_message_ros_time_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_;

  // ============ Services ============
  // ROS 2 Service to save map (replaces the ROS 1 ServiceServer)
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_map_service_;

  // ============ Camera / VIO Extras (from header) ============

  std::vector<double> cameraextrinT;
  std::vector<double> cameraextrinR;
  double IMG_POINT_COV = 10.0;

  // Helper flags
  bool colmap_output_en = false;
};

#endif  // LASER_MAPPING_EXTRA_HPP