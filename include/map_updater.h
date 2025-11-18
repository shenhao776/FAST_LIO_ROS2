/*
 * Author: SHEN HAO
 * Email: shenhao776@gmail.com
 *
 * This code is a ROS 2 adaptation of the map_updater module.
 */

#ifndef MAP_UPDATER_HPP
#define MAP_UPDATER_HPP

// ROS 2 Headers
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL Headers
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

// Third-party & Std Headers
#include <Eigen/StdVector>
#include <filesystem>
#include <functional>
#include <sophus/se3.hpp>
#include <string>
#include <unordered_set>
#include <vector>

// Local Headers
#include "common_lib.h"
#include "voxel_map.h"  // 引用新创建的头文件

// 类型定义
// 注意：PointType 等已在 common_lib.h 中定义，这里做兼容处理
#ifndef COMMON_LIB_H
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
#endif

typedef pcl::PointXYZRGB PointTypeRGB;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;

// 关键帧数据结构
struct KeyframeData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp;
  Sophus::SE3d pose;  // 使用 Sophus::SE3d (double精度) 适配 Eigen
  std::string pcd_path;
  mutable PointCloudXYZI::Ptr cloud;

  KeyframeData() : timestamp(0.0), cloud(new PointCloudXYZI()) {}

  // 如果 cloud 为空则通过 pcd_path 加载点云
  PointCloudXYZI::Ptr getCloud(bool debug = false) const {
    if (cloud && !cloud->empty()) {
      return cloud;
    }

    if (debug) {
      // 使用 common_lib.h 中的 LOG_WARN 宏
      std::cout << "[WARN] Point cloud is empty in memory, trying to load from "
                   "PCD file: "
                << pcd_path << std::endl;
    }

    PointCloudXYZI::Ptr temp_cloud(new PointCloudXYZI());
    if (!pcd_path.empty() && std::filesystem::exists(pcd_path)) {
      if (pcl::io::loadPCDFile<PointType>(pcd_path, *temp_cloud) != -1) {
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*temp_cloud, *temp_cloud, indices);
        cloud = temp_cloud;
        if (debug) {
          std::cout << "[WARN] Loaded point cloud from PCD file: " << pcd_path
                    << ", points num: " << cloud->size() << std::endl;
        }
        return cloud;
      }
    }

    if (debug) {
      std::cout << "[WARN] Failed to load point cloud from PCD file: "
                << pcd_path << std::endl;
    }
    return PointCloudXYZI::Ptr(new PointCloudXYZI());
  }
};

class MapUpdater {
 public:
  // 定义新的 VoxelMapIndexType：从体素位置映射到包含该体素的 PCD 文件路径列表
  using VoxelMapIndexType =
      std::unordered_map<VOXEL_LOCATION, std::vector<std::string>>;

  MapUpdater(const VoxelMapConfig& config, const M3D& Ril, const V3D& Pil);
  ~MapUpdater();

  void saveFinalMap(
      const std::string& output_path,
      const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& voxel_map);

  // ROS 2 发布函数：传入 Publisher 的 SharedPtr
  void publishCloud(
      rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
      const PointCloudXYZI::Ptr& cloud, const std::string& frame_id);

  void publishVoxelMap(
      rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
      const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& voxel_map,
      const std::string& frame_id);

  PointCloudXYZI::Ptr convertVoxelMapToPCL(
      const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& voxel_map);

  // --- 更新/重定位模式相关函数 ---

  /**
   * @brief 构建体素到 PCD 文件的索引地图
   */
  bool buildVoxelMapIndex(const std::string& map_path,
                          VoxelMapIndexType& target_voxel_map_index,
                          std::vector<KeyframeData>& keyframes_db);

  /**
   * @brief 为周期性对齐动态加载目标点云
   */
  PointCloudXYZI::Ptr findTargetPointsForICP(
      const std::vector<KeyframeData>& source_keyframes,
      const VoxelMapIndexType& voxel_map_index,
      const std::vector<KeyframeData>& keyframes_db,
      double periodic_align_min_dist);

  /**
   * @brief 从关键帧数据库生成一个完整的 PCL 点云
   */
  PointCloudXYZRGB::Ptr convertKeyframesToPCL(
      const std::vector<KeyframeData>& keyframes_db, const M3D& Ril,
      const V3D& Pil);

 private:
  VoxelMapConfig m_voxel_config;
  M3D m_ext_r;
  V3D m_ext_t;
};

#endif  // MAP_UPDATER_HPP