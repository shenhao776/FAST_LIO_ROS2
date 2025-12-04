/*
 * Author: SHEN HAO
 * Email: shenhao776@gmail.com
 *
 * This code is a ROS 2 adaptation of the map_updater module.
 */

#include "map_updater.hpp"  // 包含转换后的 .hpp 文件

#include <dirent.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "common_lib.h"

MapUpdater::MapUpdater(const VoxelMapConfig& config, const M3D& Ril,
                       const V3D& Pil)
    : m_voxel_config(config), m_ext_r(Ril), m_ext_t(Pil) {}

MapUpdater::~MapUpdater() {}

PointCloudXYZI::Ptr MapUpdater::convertVoxelMapToPCL(
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& voxel_map) {
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());

  // 递归函数用于从八叉树提取点
  std::function<void(VoxelOctoTree*)> extract_points_recursive =
      [&](VoxelOctoTree* node) {
        if (!node) return;
        if (!node->temp_points_.empty()) {
          for (const auto& pv : node->temp_points_) {
            PointType p;
            p.x = pv.point_w.x();
            p.y = pv.point_w.y();
            p.z = pv.point_w.z();
            p.intensity = pv.intensity;
            cloud->points.push_back(p);
          }
        }
        if (node->octo_state_ == 1) {  // 分支节点
          for (int i = 0; i < 8; ++i) {
            if (node->leaves_[i] != nullptr) {
              extract_points_recursive(node->leaves_[i]);
            }
          }
        }
      };

  for (auto const& [key, val] : voxel_map) {
    extract_points_recursive(val);
  }

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

void MapUpdater::saveFinalMap(
    const std::string& output_path,
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& voxel_map) {
  PointCloudXYZI::Ptr final_cloud = convertVoxelMapToPCL(voxel_map);
  if (final_cloud->empty()) {
    LOG_INFO_F("[WARN] Final map is empty, skipping save.");
    return;
  }
  pcl::io::savePCDFileBinary(output_path, *final_cloud);
  std::cout << "Updated map with " << final_cloud->size() << " points saved to "
            << output_path << std::endl;
}

void MapUpdater::publishCloud(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
    const PointCloudXYZI::Ptr& cloud, const std::string& frame_id) {
  if (cloud && !cloud->empty() && pub) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    // 使用系统时钟，因为 MapUpdater 没有持有 Node 指针
    msg.header.stamp = rclcpp::Clock().now();
    msg.header.frame_id = frame_id;
    pub->publish(msg);
    // ROS 2 中不需要 ros::spinOnce() 来发布消息
  }
}

void MapUpdater::publishVoxelMap(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& voxel_map,
    const std::string& frame_id) {
  PointCloudXYZI::Ptr cloud = convertVoxelMapToPCL(voxel_map);
  publishCloud(pub, cloud, frame_id);
}

// ===================================================================
//                 新增函数的实现
// ===================================================================

bool MapUpdater::buildVoxelMapIndex(const std::string& map_path,
                                    VoxelMapIndexType& target_voxel_map_index,
                                    std::vector<KeyframeData>& keyframes_db) {
  LOG_INFO_F("Starting memory-less index build from: %s", map_path.c_str());
  target_voxel_map_index.clear();
  keyframes_db.clear();

  std::string pose_file = map_path + "/result/localization_output.txt";
  std::string keyframes_dir = map_path + "/result/keyframes/";
  std::ifstream infile(pose_file);
  if (!infile.is_open()) {
    LOG_ERROR_F("Cannot open pose file for index building: %s",
                pose_file.c_str());
    return false;
  }

  float voxel_size = m_voxel_config.max_voxel_size_;

  std::string line;
  while (std::getline(infile, line)) {
    std::stringstream ss(line);
    KeyframeData kf;
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    ss >> kf.timestamp >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >>
        q.w();
    kf.pose = Sophus::SE3d(q, t);  // 显式使用 SE3d

    std::stringstream pcd_filename_ss;
    pcd_filename_ss << std::fixed << std::setprecision(9) << kf.timestamp
                    << ".pcd";
    kf.pcd_path = keyframes_dir + pcd_filename_ss.str();

    if (std::filesystem::exists(kf.pcd_path)) {
      keyframes_db.push_back(kf);
    }
  }
  infile.close();
  if (keyframes_db.empty()) {
    return false;
  }
  LOG_INFO_F("Loaded metadata for %lu keyframes.", keyframes_db.size());

  // 使用临时 map 构建索引
  std::unordered_map<VOXEL_LOCATION, std::unordered_set<std::string>>
      temp_voxel_map;

  for (const auto& kf : keyframes_db) {
    PointCloudXYZI::Ptr kf_cloud = kf.getCloud();
    if (kf_cloud->empty()) continue;

    for (const auto& p_body : kf_cloud->points) {
      V3D p_lidar_body(p_body.x, p_body.y, p_body.z);
      V3D p_imu_body = m_ext_r * p_lidar_body + m_ext_t;
      V3D p_world =
          kf.pose.rotationMatrix() * p_imu_body + kf.pose.translation();

      float loc_xyz[3];
      for (int j = 0; j < 3; j++) {
        loc_xyz[j] = p_world[j] / voxel_size;
        if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
      }
      VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                              (int64_t)loc_xyz[2]);
      temp_voxel_map[position].insert(kf.pcd_path);
    }
  }

  for (const auto& pair : temp_voxel_map) {
    target_voxel_map_index[pair.first] =
        std::vector<std::string>(pair.second.begin(), pair.second.end());
  }

  LOG_INFO_F("Voxel index building process completed. Final voxel count: %lu",
             target_voxel_map_index.size());
  return !target_voxel_map_index.empty();
}

PointCloudXYZI::Ptr MapUpdater::findTargetPointsForICP(
    const std::vector<KeyframeData>& new_keyframes,
    const VoxelMapIndexType& voxel_map_index,
    const std::vector<KeyframeData>& keyframes_db,
    double periodic_align_min_dist) {
  PointCloudXYZI::Ptr target_cloud(new PointCloudXYZI());
  if (new_keyframes.empty()) {
    return target_cloud;
  }

  float voxel_size = m_voxel_config.max_voxel_size_;

  std::unordered_set<VOXEL_LOCATION, std::hash<VOXEL_LOCATION>>
      voxels_to_search;
  std::unordered_set<std::string> pcds_to_load;

  // 1. 确定所有源关键帧覆盖的体素区域
  for (const auto& src_kf : new_keyframes) {
    PointCloudXYZI::Ptr source_cloud_body = src_kf.getCloud();
    if (source_cloud_body->empty()) continue;

    for (const auto& pt : source_cloud_body->points) {
      V3D p_lidar_body(pt.x, pt.y, pt.z);
      V3D p_imu_body = m_ext_r * p_lidar_body + m_ext_t;
      V3D p_world =
          src_kf.pose.rotationMatrix() * p_imu_body + src_kf.pose.translation();

      float loc_xyz[3];
      for (int j = 0; j < 3; j++) {
        loc_xyz[j] = p_world[j] / voxel_size;
        if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
      }
      voxels_to_search.insert(VOXEL_LOCATION(
          (int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]));
    }
  }

  // 2. 从索引中收集所有需要加载的PCD文件路径
  for (const auto& voxel_loc : voxels_to_search) {
    auto it = voxel_map_index.find(voxel_loc);
    if (it != voxel_map_index.end()) {
      for (const auto& pcd_path : it->second) {
        pcds_to_load.insert(pcd_path);
      }
    }
  }

  if (pcds_to_load.empty()) {
    LOG_WARN_F(
        "[WARN] No PCD files found in the vicinity of the current scan.");
    return target_cloud;
  }

  // 3. 加载、变换并聚合点云
  // [优化] 预留空间避免频繁扩容
  target_cloud->points.reserve(pcds_to_load.size() * 8000);

  for (const auto& pcd_path : pcds_to_load) {
    auto kf_it = std::find_if(
        keyframes_db.begin(), keyframes_db.end(),
        [&](const KeyframeData& kf) { return kf.pcd_path == pcd_path; });

    if (kf_it != keyframes_db.end()) {
      PointCloudXYZI::Ptr temp_cloud = kf_it->getCloud();
      if (temp_cloud->empty()) continue;

      // 提取变换矩阵，减少内层循环计算量
      Eigen::Matrix3d R = kf_it->pose.rotationMatrix();
      Eigen::Vector3d t = kf_it->pose.translation();

      for (const auto& p_body : temp_cloud->points) {
        // 坐标变换: Lidar Body -> IMU Body -> World
        V3D p_lidar_body(p_body.x, p_body.y, p_body.z);
        V3D p_imu_body = m_ext_r * p_lidar_body + m_ext_t;
        V3D p_world = R * p_imu_body + t;

        PointType p_w;
        p_w.x = p_world.x();
        p_w.y = p_world.y();
        p_w.z = p_world.z();
        p_w.intensity = p_body.intensity;

        // =========== 【关键修复】 ===========
        // 必须初始化法向量字段为 0，防止内存中的随机 NaN 导致 PCL 算法崩溃
        p_w.normal_x = 0.0f;
        p_w.normal_y = 0.0f;
        p_w.normal_z = 0.0f;
        p_w.curvature = 0.0f;
        // ===================================

        target_cloud->points.push_back(p_w);
      }
    }
  }

  // =========== 【关键修复】 ===========
  // 必须设置点云属性，否则 KDTree 构建会失败
  target_cloud->width = target_cloud->points.size();
  target_cloud->height = 1;
  target_cloud->is_dense = true;
  // ===================================

  std::cout << "[INFO] Loaded " << pcds_to_load.size() << " PCDs, aggregated "
            << target_cloud->size() << " points for ICP target." << std::endl;
  return target_cloud;
}

struct VoxelKey {
  int64_t x, y, z;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& k) const {
    std::size_t hash = std::hash<int64_t>{}(k.x);
    hash = (hash * 16777619) ^ std::hash<int64_t>{}(k.y);
    hash = (hash * 16777619) ^ std::hash<int64_t>{}(k.z);
    return hash;
  }
};

PointCloudXYZRGB::Ptr MapUpdater::convertKeyframesToPCL(
    const std::vector<KeyframeData>& keyframes_db, const M3D& Ril,
    const V3D& Pil) {
  PointCloudXYZRGB::Ptr combined_cloud(new PointCloudXYZRGB());
  if (keyframes_db.empty()) {
    return combined_cloud;
  }

  const float leaf_size = 0.01f;
  const float inv_leaf_size = 1.0f / leaf_size;

  std::unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels;

  size_t total_points_estimate = 0;
  for (const auto& kf : keyframes_db) {
    if (kf.cloud) {
      total_points_estimate += kf.cloud->points.size();
    } else {
      if (!kf.pcd_path.empty() && std::filesystem::exists(kf.pcd_path)) {
        total_points_estimate +=
            std::filesystem::file_size(kf.pcd_path) / sizeof(PointType);
      }
    }
  }

  // 简单预留空间，不一定是精确的
  combined_cloud->reserve(total_points_estimate);
  occupied_voxels.reserve(total_points_estimate);

  for (const auto& kf : keyframes_db) {
    PointCloudXYZI::Ptr kf_cloud = kf.getCloud();
    if (kf_cloud->empty()) continue;

    for (const auto& p_body : kf_cloud->points) {
      V3D p_lidar_body(p_body.x, p_body.y, p_body.z);
      V3D p_imu_body = Ril * p_lidar_body + Pil;
      V3D p_world =
          kf.pose.rotationMatrix() * p_imu_body + kf.pose.translation();

      VoxelKey key;
      key.x = static_cast<int64_t>(std::floor(p_world.x() * inv_leaf_size));
      key.y = static_cast<int64_t>(std::floor(p_world.y() * inv_leaf_size));
      key.z = static_cast<int64_t>(std::floor(p_world.z() * inv_leaf_size));

      if (occupied_voxels.find(key) == occupied_voxels.end()) {
        occupied_voxels.insert(key);

        PointTypeRGB p_w;
        p_w.x = p_world.x();
        p_w.y = p_world.y();
        p_w.z = p_world.z();
        uint8_t intensity = static_cast<uint8_t>(p_body.intensity);
        p_w.r = intensity;
        p_w.g = intensity;
        p_w.b = intensity;
        combined_cloud->points.push_back(p_w);
      }
    }
  }

  combined_cloud->width = combined_cloud->points.size();
  combined_cloud->height = 1;
  combined_cloud->is_dense = true;

  LOG_INFO_F("Finished assembling final map with %lu points.",
             combined_cloud->size());

  return combined_cloud;
}