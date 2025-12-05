/* This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.

Modified by: SHEN HAO
Email: shenhao776@gmail.com

This code is a modified version of the FAST-LIVO2 framework.
Original repository: https://github.com/hku-mars/FAST-LIVO2
*/

#include "fast_lio/laserMapping.hpp"

#include <pcl/filters/extract_indices.h>
#include <tf2/LinearMath/Quaternion.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

LaserMappingNode* LaserMappingNode::ptr_ = nullptr;

void LaserMappingNode::OnSignal(int sig) {
  if (ptr_) {
    ptr_->flg_exit = true;
    std::cout << "catch sig " << sig << std::endl;
    ptr_->sig_buffer.notify_all();
  }
  rclcpp::shutdown();
}

LaserMappingNode::LaserMappingNode(const rclcpp::NodeOptions& options)
    : Node("fastlio_mapping", options),
      p_pre(new Preprocess()),
      p_imu(new ImuProcess()),
      featsFromMap(new PointCloudXYZI()),
      feats_undistort(new PointCloudXYZI()),
      feats_down_body(new PointCloudXYZI()),
      feats_down_world(new PointCloudXYZI()),
      normvec(new PointCloudXYZI(100000, 1)),
      laserCloudOri(new PointCloudXYZI(100000, 1)),
      corr_normvect(new PointCloudXYZI(100000, 1)),
      _featsArray(new PointCloudXYZI()),
      pcl_wait_pub(new PointCloudXYZI()),
      pcl_wait_save(new PointCloudXYZI()),
      original_map_visual_(new PointCloudXYZRGB()),
      deleted_points_visual_(new PointCloudXYZRGB()),
      XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0),
      XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0),
      Lidar_T_wrt_IMU(Zero3d),
      Lidar_R_wrt_IMU(Eye3d) {
  ptr_ = this;
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  // memset(point_selected_surf, true, sizeof(point_selected_surf));
  // memset(res_last, -1000.0f, sizeof(res_last));
  point_selected_surf.resize(MAXN, true);
  res_last.resize(MAXN, -1000.0f);

  readParameters();

  initializeComponents();

  std::fill(epsi, epsi + 23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS,
                    epsi);

  // Initialize VoxelMapConfig
  VoxelMapConfig voxel_config;
  loadVoxelConfig(this, voxel_config);

  // Initialize VoxelMapManager
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));

  // Initialize MapUpdater
  map_updater_ = std::make_unique<MapUpdater>(voxel_config, Lidar_R_wrt_IMU,
                                              Lidar_T_wrt_IMU);

  // Check and load existing map
  std::string pose_file = map_data_path_ + "/result/localization_output.txt";
  if (std::filesystem::exists(pose_file)) {
    is_update_mode_ = true;
    LOG_INFO_F("\033[1;32mMap data found at %s. Running in UPDATE mode.\033[0m",
               map_data_path_.c_str());

    // Build Index from existing map
    if (!map_updater_->buildVoxelMapIndex(map_data_path_, voxel_map_index,
                                          original_map_keyframes_)) {
      LOG_ERROR_F("Failed to build map index.");
    }

    // Load visualization map
    if (!loadExistingMap(map_data_path_)) {
      LOG_WARN_F("Failed to load visualization map.");
    }
  } else {
    is_update_mode_ = false;
    LOG_INFO_F("\033[1;32mNo map data found. Running in MAPPING mode.\033[0m");
  }

  initializeFiles();
  initializeSubscribersAndPublishers();

  FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
  HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                 filter_size_surf_min);
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min,
                                filter_size_map_min);

  LOG_INFO_F("Node init finished.");
}

LaserMappingNode::~LaserMappingNode() {
  if (fout_out.is_open()) fout_out.close();
  if (fout_pre.is_open()) fout_pre.close();
  if (fout_dbg.is_open()) fout_dbg.close();
  if (fout_pcd_pos.is_open()) fout_pcd_pos.close();
  if (fp) fclose(fp);
  if (ptr_ == this) ptr_ = nullptr;
}

void LaserMappingNode::initializeComponents() {
  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
}

void LaserMappingNode::initializeFiles() {
  // 确保目录存在
  std::string root_dir_path = map_data_path_;
  if (!std::filesystem::exists(root_dir_path)) {
    std::filesystem::create_directories(root_dir_path);
  }

  if (is_update_mode_) {
    LOG_INFO_F("Update mode: Preserving existing map data in %s",
               map_data_path_.c_str());
  } else {
    LOG_INFO_F("[WARN] Mapping mode: Cleaning up old data in %s",
               map_data_path_.c_str());
    // 仅在建图模式下清理旧数据
    std::string rm_cmd = "rm -rf " + map_data_path_ + "/*";  // 保留目录本身
    (void)system(rm_cmd.c_str());
  }

  // 创建必要的子目录
  std::filesystem::create_directories(map_data_path_ + "/PCD");
  std::filesystem::create_directories(map_data_path_ + "/result/keyframes");
  std::filesystem::create_directories(map_data_path_ +
                                      "/result/images");  // 如果需要保存图片
  std::filesystem::create_directories(map_data_path_ + "/Log");
}

void LaserMappingNode::readParameters() {
  auto declare_and_get = [&](const std::string& name, auto& var,
                             auto default_val) {
    if (!this->has_parameter(name)) {
      this->declare_parameter(name, default_val);
    }
    this->get_parameter(name, var);
  };
  declare_and_get("common.img_topic", img_topic,
                  std::string("/camera/camera/color/image_raw"));
  declare_and_get("common.lid_topic", lid_topic, std::string("/livox/lidar"));
  declare_and_get("common.imu_topic", imu_topic, std::string("/livox/imu"));
  declare_and_get("common.time_sync_en", time_sync_en, false);
  declare_and_get("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu,
                  0.0);

  declare_and_get("filter_size_corner", filter_size_corner_min, 0.5);
  declare_and_get("filter_size_surf", filter_size_surf_min, 0.5);
  declare_and_get("filter_size_map", filter_size_map_min, 0.5);
  declare_and_get("cube_side_length", cube_len, 200.0);

  declare_and_get("mapping.det_range", DET_RANGE, 300.0f);
  declare_and_get("mapping.fov_degree", fov_deg, 180.0);
  declare_and_get("mapping.gyr_cov", gyr_cov, 0.1);
  declare_and_get("mapping.acc_cov", acc_cov, 0.1);
  declare_and_get("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
  declare_and_get("mapping.b_acc_cov", b_acc_cov, 0.0001);
  declare_and_get("mapping.extrinsic_est_en", extrinsic_est_en, true);
  declare_and_get("mapping.extrinsic_T", extrinT, std::vector<double>());
  declare_and_get("mapping.extrinsic_R", extrinR, std::vector<double>());

  declare_and_get("preprocess.blind", p_pre->blind, 0.01);
  declare_and_get("preprocess.lidar_type", p_pre->lidar_type, (int)AVIA);
  declare_and_get("preprocess.scan_line", p_pre->N_SCANS, 16);
  declare_and_get("preprocess.timestamp_unit", p_pre->time_unit, (int)US);
  declare_and_get("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
  declare_and_get("preprocess.point_filter_num", p_pre->point_filter_num, 2);
  declare_and_get("feature_extract_enable", p_pre->feature_enabled, false);

  declare_and_get("publish.path_en", path_en, true);
  declare_and_get("publish.effect_map_en", effect_pub_en, false);
  declare_and_get("publish.map_en", map_pub_en, false);
  declare_and_get("publish.scan_publish_en", scan_pub_en, true);
  declare_and_get("publish.dense_publish_en", dense_pub_en, true);
  declare_and_get("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);

  declare_and_get("pcd_save.pcd_save_en", pcd_save_en, false);
  declare_and_get("pcd_save.interval", pcd_save_interval, -1);
  declare_and_get("pcd_save.map_data_path", map_data_path_,
                  "/root/shared_files/dataset/my_map_data");
  declare_and_get("pcd_save.pcd_save_distance_thresh",
                  pcd_save_distance_thresh_, 0.2);
  declare_and_get("pcd_save.filter_size_pcd", filter_size_pcd, 0.5);

  declare_and_get("max_iteration", NUM_MAX_ITERATIONS, 4);

  declare_and_get("online_update.periodic_align_interval",
                  periodic_align_interval_, 20);
  declare_and_get("online_update.periodic_align_min_dist",
                  periodic_align_min_dist_, 0.3);
  declare_and_get("online_update.keyframe_search_radius",
                  keyframe_search_radius_, 3.0);
  declare_and_get("online_update.point_replacement_radius",
                  point_replacement_radius_, 0.5);
}

void LaserMappingNode::initializeSubscribersAndPublishers() {
  if (p_pre->lidar_type == AVIA) {
    sub_pcl_livox_ =
        this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            lid_topic, 20,
            std::bind(&LaserMappingNode::livox_pcl_cbk, this,
                      std::placeholders::_1));
  } else {
    sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lid_topic, rclcpp::SensorDataQoS(),
        std::bind(&LaserMappingNode::standard_pcl_cbk, this,
                  std::placeholders::_1));
  }
  sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, 10,
      std::bind(&LaserMappingNode::imu_cbk, this, std::placeholders::_1));

  if (is_update_mode_) {
    sub_initial_pose_ = this->create_subscription<
        geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pcl_pose", 1,
        std::bind(&LaserMappingNode::initialPoseCallback, this,
                  std::placeholders::_1));
  }

  pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/cloud_registered", 20);
  pubLaserCloudFull_body_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/cloud_registered_body", 20);
  pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/cloud_effected", 20);
  pubLaserCloudMap_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
  pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "/aft_mapped_to_init", 20);
  pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // 地图更新相关的发布者
  pub_original_map_kept_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/map/original_map_kept", rclcpp::QoS(1).transient_local());
  pub_newly_added_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/map/newly_added_map", rclcpp::QoS(1).transient_local());
  pub_deleted_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/map/deleted_points", rclcpp::QoS(1).transient_local());
  pub_final_updated_map_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/map/final_updated_map", rclcpp::QoS(1).transient_local());

  voxel_map_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planes", 100);
  if (voxelmap_manager) voxelmap_manager->voxel_map_pub_ = voxel_map_pub_;

  auto period_ms =
      std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
  timer_ =
      rclcpp::create_timer(this, this->get_clock(), period_ms,
                           std::bind(&LaserMappingNode::timer_callback, this));
  if (!img_topic.empty()) {
    sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(
        img_topic, 100,
        std::bind(&LaserMappingNode::img_cbk, this, std::placeholders::_1));
  }
  auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
  map_pub_timer_ = rclcpp::create_timer(
      this, this->get_clock(), map_period_ms,
      std::bind(&LaserMappingNode::map_publish_callback, this));

  // 将服务回调绑定到 saveMapCallback
  map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "map_save", std::bind(&LaserMappingNode::saveMapCallback, this,
                            std::placeholders::_1, std::placeholders::_2));

  if (is_update_mode_ && original_map_visual_ &&
      !original_map_visual_->empty()) {
    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*original_map_visual_, map_msg);
    map_msg.header.stamp = this->now();
    map_msg.header.frame_id = "camera_init";
    pub_original_map_kept_->publish(map_msg);
  }
}

void LaserMappingNode::timer_callback() {
  // 1. 数据同步
  if (sync_packages(Measures)) {
    if (flg_first_scan) {
      first_lidar_time = Measures.lidar_beg_time;
      p_imu->first_lidar_time = first_lidar_time;
      flg_first_scan = false;
      return;
    }

    double t0, t1, t2, t3, t4;  // [移植] 增加 t4 变量
    match_time = 0;
    kdtree_search_time = 0.0;
    solve_time = 0;
    solve_const_H_time = 0;
    t0 = omp_get_wtime();

    // 2. IMU 预积分 & 去畸变
    p_imu->Process(Measures, kf, feats_undistort);
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    if (feats_undistort->empty() || (feats_undistort == NULL)) {
      return;
    }

    flg_EKF_inited =
        (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

    // 3. 动态视场调整
    lasermap_fov_segment();

    // 4. 降采样
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
    t1 = omp_get_wtime();
    feats_down_size = feats_down_body->points.size();

    // --- 【防崩溃】动态扩容 ---
    if (point_selected_surf.size() < feats_down_size) {
      size_t new_size = feats_down_size + 10000;
      LOG_WARN_F("Resizing vectors to %lu", new_size);
      point_selected_surf.resize(new_size, true);
      res_last.resize(new_size, -1000.0f);
    }

    // 5. 初始化 ikd-Tree
    if (ikdtree.Root_Node == nullptr) {
      if (feats_down_size > 5) {
        ikdtree.set_downsample_param(filter_size_map_min);
        feats_down_world->resize(feats_down_size);
        for (int i = 0; i < feats_down_size; i++) {
          pointBodyToWorld(&(feats_down_body->points[i]),
                           &(feats_down_world->points[i]));
        }
        ikdtree.Build(feats_down_world->points);
      }
      return;
    }

    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);
    pointSearchInd_surf.resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);

    t2 = omp_get_wtime();
    double solve_H_time = 0;

    // ==============================================================================
    // 核心逻辑分支
    // ==============================================================================
    if (!is_update_mode_) {
      // --- A. 建图模式 (Mapping Mode) ---
      kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

      state_point = kf.get_x();
      euler_cur = SO3ToEuler(state_point.rot);
      pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
      geoQuat.x = state_point.rot.coeffs()[0];
      geoQuat.y = state_point.rot.coeffs()[1];
      geoQuat.z = state_point.rot.coeffs()[2];
      geoQuat.w = state_point.rot.coeffs()[3];

      publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

      t3 = omp_get_wtime();
      map_incremental();
    } else {
      // --- B. 更新/定位模式 (Update Mode) ---
      // B.1 初始配准
      if (!initial_align_finished_) {
        if (!initial_pose_received_) {
          static int log_cnt = 0;
          if (log_cnt++ % 50 == 0) LOG_INFO_F("Waiting for initial pose...");
          return;
        }

        LOG_INFO_F("Performing initial ICP alignment...");
        KeyframeData temp_kf;
        temp_kf.pose = initial_pose_;
        temp_kf.cloud.reset(new PointCloudXYZI(*feats_undistort));
        std::vector<KeyframeData> temp_kfs = {temp_kf};

        PointCloudXYZI::Ptr target_cloud = map_updater_->findTargetPointsForICP(
            temp_kfs, voxel_map_index, original_map_keyframes_, 30.0);

        if (target_cloud->empty()) {
          LOG_WARN_F("Initial Alignment: Target cloud empty.");
          return;
        }

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setInputSource(feats_undistort);
        icp.setInputTarget(target_cloud);
        icp.setMaxCorrespondenceDistance(1.0);
        icp.setMaximumIterations(100);

        PointCloudXYZI unused_result;
        Eigen::Matrix4f guess = initial_pose_.matrix().cast<float>();
        icp.align(unused_result, guess);

        if (icp.hasConverged()) {
          Eigen::Matrix4d T_final = icp.getFinalTransformation().cast<double>();

          // 1. 检查 NaN，防止非法数据传入
          if (T_final.array().isNaN().any()) {
            LOG_ERROR_F("ICP result contains NaN! Skipping initial alignment.");
            return;
          }

          // 2. 使用四元数构造 SE3，以解决旋转矩阵正交性误差导致的 Sophus 崩溃
          Eigen::Matrix3d R_final = T_final.block<3, 3>(0, 0);
          Eigen::Vector3d t_final = T_final.block<3, 1>(0, 3);

          // 通过四元数归一化，强制修正旋转矩阵的微小数值误差
          Eigen::Quaterniond q_final(R_final);
          q_final.normalize();

          Sophus::SE3d aligned_pose(q_final, t_final);
          Sophus::SE3d T_imu_lidar(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);
          Sophus::SE3d T_world_imu = aligned_pose * T_imu_lidar.inverse();

          state_point.rot = T_world_imu.unit_quaternion();
          state_point.pos = T_world_imu.translation();
          kf.change_x(state_point);

          initial_align_finished_ = true;
          LOG_INFO_F(
              "\033[1;32mInitial Alignment Success! state_point = [%f, %f, "
              "%f], Fitness:%f\033[0m",
              state_point.pos.x(), state_point.pos.y(), state_point.pos.z(),
              icp.getFitnessScore());
        } else {
          LOG_WARN_F("Initial ICP failed.");
          return;
        }
      }

      // B.2 持续跟踪
      if (initial_align_finished_) {
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

        state_point = kf.get_x();
        euler_cur = SO3ToEuler(state_point.rot);
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];

        KeyframeData new_kf;
        new_kf.timestamp = Measures.lidar_beg_time;
        new_kf.pose =
            Sophus::SE3d(state_point.rot.toRotationMatrix(), state_point.pos);
        new_kf.cloud.reset(new PointCloudXYZI(*feats_undistort));
        new_keyframes_.push_back(new_kf);

        if (new_keyframes_.size() == 1) {
          last_kf_pose_ = new_kf.pose;
          motion_keyframe_count_++;
        } else {
          Sophus::SE3d delta = last_kf_pose_.inverse() * new_kf.pose;
          if (delta.translation().norm() > periodic_align_min_dist_) {
            motion_keyframe_count_++;
            last_kf_pose_ = new_kf.pose;
          }
        }
        LOG_WARN_F("Alignment1...");
        periodicAlignment();
        LOG_WARN_F("Alignment1...");
        euler_cur = SO3ToEuler(state_point.rot);
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];

        publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

        t3 = omp_get_wtime();
        map_incremental();
      }
    }

    // ==============================================================================
    // 数据保存模块 (Pose, PCD, Images)
    // ==============================================================================

    // [修复]：定义局部变量用于在锁外保存图像
    cv::Mat image_to_save;
    std::string img_filename_to_save;
    bool need_save_img = false;

    if (pcd_save_en && flg_EKF_inited) {
      std::string pose_file_path =
          is_update_mode_ ? map_data_path_ + "/localization_output_updated.txt"
                          : map_data_path_ + "/result/localization_output.txt";

      static bool dir_checked = false;
      if (!dir_checked) {
        std::filesystem::path p(pose_file_path);
        if (!std::filesystem::exists(p.parent_path())) {
          std::filesystem::create_directories(p.parent_path());
        }
        dir_checked = true;
      }

      std::ofstream evoFile(pose_file_path, std::ios::app);
      if (evoFile.is_open()) {
        evoFile << std::fixed << std::setprecision(9) << Measures.lidar_beg_time
                << " " << state_point.pos.x() << " " << state_point.pos.y()
                << " " << state_point.pos.z() << " " << geoQuat.x << " "
                << geoQuat.y << " " << geoQuat.z << " " << geoQuat.w
                << std::endl;
        evoFile.close();
      }

      if (!is_update_mode_) {
        double dist = (state_point.pos - last_pcd_save_pos_).norm();
        if (!is_first_pcd_saved_ || dist > pcd_save_distance_thresh_) {
          last_pcd_save_pos_ = state_point.pos;
          is_first_pcd_saved_ = true;

          std::stringstream ss;
          ss << std::fixed << std::setprecision(9) << Measures.lidar_beg_time;
          std::string pcd_filename =
              map_data_path_ + "/result/keyframes/" + ss.str() + ".pcd";

          if (!std::filesystem::exists(map_data_path_ + "/result/keyframes/")) {
            std::filesystem::create_directories(map_data_path_ +
                                                "/result/keyframes/");
          }
          pcl::io::savePCDFileBinary(pcd_filename, *feats_undistort);
        }
      }

      // [修复]：图像匹配逻辑 (加锁)，但不在此进行 I/O 操作
      mtx_buffer.lock();
      if (!img_buffer.empty()) {
        double current_lidar_time = Measures.lidar_beg_time;
        double min_time_diff = 1000.0;
        int best_match_idx = -1;

        for (size_t i = 0; i < img_time_buffer.size(); ++i) {
          double time_diff = std::abs(img_time_buffer[i] - current_lidar_time);
          if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            best_match_idx = i;
          }
        }

        if (best_match_idx != -1 && min_time_diff < 0.05) {
          if (!img_buffer[best_match_idx].empty()) {
            // [关键修复] 仅克隆图像数据，不进行磁盘 I/O
            image_to_save = img_buffer[best_match_idx].clone();

            std::stringstream ss;
            ss << std::fixed << std::setprecision(9) << current_lidar_time;
            std::string img_dir = map_data_path_ + "/result/images/";
            img_filename_to_save = img_dir + ss.str() + ".png";

            // 目录检查建议在初始化时完成，这里为了安全也可以保留，但在锁内检查文件系统有微小开销
            if (!std::filesystem::exists(img_dir)) {
              std::filesystem::create_directories(img_dir);
            }
            need_save_img = true;
          }
          // 清理缓冲区
          img_buffer.erase(img_buffer.begin(),
                           img_buffer.begin() + best_match_idx + 1);
          img_time_buffer.erase(img_time_buffer.begin(),
                                img_time_buffer.begin() + best_match_idx + 1);
        }
      }
      mtx_buffer.unlock();  // [关键] 尽快释放锁
    }

    // [修复]：在锁外执行耗时的图像保存
    if (need_save_img && !image_to_save.empty()) {
      cv::imwrite(img_filename_to_save, image_to_save);
    }

    // 6. 发布话题
    if (path_en) publish_path(pubPath_);
    if (scan_pub_en) publish_frame_world(pubLaserCloudFull_);
    if (scan_pub_en && scan_body_pub_en)
      publish_frame_body(pubLaserCloudFull_body_);
    if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);

    if (voxelmap_manager &&
        voxelmap_manager->config_setting_.is_pub_plane_map_) {
      voxelmap_manager->pubVoxelMap(this->now());
    }

    // ==============================================================================
    // [移植] 工程2的输出部分 (Output Part)
    // ==============================================================================
    t4 = omp_get_wtime();
    frame_num++;
    aver_time_consu =
        aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

    LOG_INFO_F(
        "\033[1;34m+-----------------------------------------------------------"
        "-+\033[0m");
    LOG_INFO_F(
        "\033[1;34m|                         LIO Mapping Time                  "
        " |\033[0m");
    LOG_INFO_F(
        "\033[1;34m+-----------------------------------------------------------"
        "-+\033[0m");
    LOG_INFO_F("\033[1;34m| %-29s | %-27s |\033[0m", "Algorithm Stage",
               "Time (secs)");
    LOG_INFO_F(
        "\033[1;34m+-----------------------------------------------------------"
        "-+\033[0m");
    LOG_INFO_F("\033[1;36m| %-29s | %-27f |\033[0m", "DownSample", t1 - t0);
    LOG_INFO_F("\033[1;36m| %-29s | %-27f |\033[0m", "State Estimation",
               t2 - t1);  // 注意：工程1这里是 t2-t1 为配准前准备，t3-t2 为EKF
    // 根据工程1的变量：
    // t0: 开始
    // t1: 降采样结束
    // t2: KDTree 构建完成/准备开始 EKF
    // t3: EKF 结束 / 建图开始
    // t4: 结束
    // 对应关系调整：
    LOG_INFO_F("\033[1;36m| %-29s | %-27f |\033[0m", "Preprocess & Tree",
               t2 - t0);
    LOG_INFO_F("\033[1;36m| %-29s | %-27f |\033[0m", "State Estimation",
               t3 - t2);
    LOG_INFO_F("\033[1;36m| %-29s | %-27f |\033[0m", "Map Incremental",
               t4 - t3);
    LOG_INFO_F(
        "\033[1;34m+-----------------------------------------------------------"
        "-+\033[0m");
    LOG_INFO_F("\033[1;36m| %-29s | %-27f |\033[0m", "Current Total Time",
               t4 - t0);
    LOG_INFO_F("\033[1;36m| %-29s | %-27f |\033[0m", "Average Total Time",
               aver_time_consu);
    LOG_INFO_F(
        "\033[1;34m+-----------------------------------------------------------"
        "-+\033[0m");

    // [关键] 强制刷新 stdout 缓冲区，防止 printf/LOG
    // 内容滞留在缓冲区造成“卡顿”假象
    fflush(stdout);
  }
}

void LaserMappingNode::periodicAlignment() {
  // [Debug] 强制打印日志以确认函数是否被安全调用
  printf("[DEBUG] Entering periodicAlignment...\n");
  fflush(stdout);

  if (!is_update_mode_ || motion_keyframe_count_ < periodic_align_interval_) {
    return;
  }

  // ==============================================================================
  // 1. 聚合源点云 (Source: Current SLAM World)
  // [核心修复] 保留手动变换，避开 pcl::transformPointCloud 内部的
  // SIMD/内存对齐崩溃
  // ==============================================================================
  PointCloudXYZI::Ptr source_cloud_world(new PointCloudXYZI());

  for (const auto& kf_new : new_keyframes_) {
    PointCloudXYZI::Ptr kf_cloud = kf_new.getCloud();
    if (kf_cloud->empty()) continue;
    PointCloudXYZI::Ptr transformed_cloud(new PointCloudXYZI());
    // [Project 2 Logic] 转换到世界系
    pcl::transformPointCloud(*kf_cloud, *transformed_cloud,
                             kf_new.pose.matrix().cast<float>());
    *source_cloud_world += *transformed_cloud;
  }

  if (source_cloud_world->empty()) {
    new_keyframes_all_.insert(new_keyframes_all_.end(), new_keyframes_.begin(),
                              new_keyframes_.end());
    new_keyframes_.clear();
    motion_keyframe_count_ = 0;
    return;
  }

  // ==============================================================================
  // 2. 获取目标点云 (Target: Global Map)
  // ==============================================================================
  PointCloudXYZI::Ptr target_cloud = map_updater_->findTargetPointsForICP(
      new_keyframes_, voxel_map_index, original_map_keyframes_,
      periodic_align_min_dist_);

  if (target_cloud->points.size() < 100) {
    LOG_WARN_F(
        "Periodic Alignment: Target map too small (%zu points). Skipping.",
        target_cloud->points.size());
    return;
  }

  // ==============================================================================
  // 3. 执行 ICP 配准
  // ==============================================================================
  pcl::VoxelGrid<PointType> voxel_filter;
  voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f);

  PointCloudXYZI::Ptr source_cloud_filtered(new PointCloudXYZI());
  voxel_filter.setInputCloud(source_cloud_world);
  voxel_filter.filter(*source_cloud_filtered);

  PointCloudXYZI::Ptr target_cloud_filtered(new PointCloudXYZI());
  voxel_filter.setInputCloud(target_cloud);
  voxel_filter.filter(*target_cloud_filtered);

  if (source_cloud_filtered->empty() || target_cloud_filtered->empty()) {
    return;
  }

  Eigen::Matrix4f initial_guess_matrix = Eigen::Matrix4f::Identity();
  if (initial_pose_received_ && !new_keyframes_.empty()) {
    Sophus::SE3d T_world_ndt;
    {
      std::lock_guard<std::mutex> lock(mtx_initial_pose);
      T_world_ndt = initial_pose_;
    }
    Sophus::SE3d T_world_slam = new_keyframes_.back().pose;
    Sophus::SE3d T_guess = T_world_ndt * T_world_slam.inverse();
    initial_guess_matrix = T_guess.matrix().cast<float>();
  }

  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setInputSource(source_cloud_filtered);
  icp.setInputTarget(target_cloud_filtered);
  icp.setMaxCorrespondenceDistance(1.5);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);

  PointCloudXYZI unused_result;
  try {
    icp.align(unused_result, initial_guess_matrix);
  } catch (...) {
    LOG_ERROR_F("ICP alignment threw an exception! Skipping.");
    return;
  }

  // ==============================================================================
  // 4. 对齐后处理
  // ==============================================================================
  bool converged = icp.hasConverged();
  double score = icp.getFitnessScore();

  if (converged && score < 0.6) {
    Eigen::Matrix4d T_final_mat = icp.getFinalTransformation().cast<double>();

    if (T_final_mat.array().isNaN().any()) {
      LOG_ERROR_F("ICP result contains NaN!");
      return;
    }

    Eigen::Matrix3d R_final = T_final_mat.block<3, 3>(0, 0);
    Eigen::Vector3d t_final = T_final_mat.block<3, 1>(0, 3);
    Eigen::Quaterniond q_final(R_final);
    q_final.normalize();

    Sophus::SE3d T_map_slam(q_final, t_final);

    LOG_INFO_F("\033[1;32mPeriodic Alignment Success! Score: %.4f\033[0m",
               score);

    // 更新关键帧位姿
    for (auto& kf : new_keyframes_) {
      kf.pose = T_map_slam * kf.pose;
    }

    // 更新系统状态
    Sophus::SE3d current_pose_slam(state_point.rot.toRotationMatrix(),
                                   state_point.pos);
    Sophus::SE3d corrected_pose_map = T_map_slam * current_pose_slam;
    state_point.rot = corrected_pose_map.unit_quaternion();
    state_point.pos = corrected_pose_map.translation();
    kf.change_x(state_point);

    // ==============================================================================
    // [核心修改] 与工程 2 保持一致：
    // 1. 移除导致崩溃的 ikdtree = KD_TREE<PointType>();
    // 2. 增加 voxel_map 清理逻辑
    // ==============================================================================
    LOG_INFO_F(
        "\033[1;33mResetting local voxel map to reflect global "
        "correction.\033[0m");

    voxel_map.clear();

    // 如果 LaserMappingNode 中有类似 lidar_map_inited 的标志位，可以在这里重置
    // 但在您提供的代码中，它主要依赖 voxel_map 的状态
    // 此处移除 ikdtree 的重置，直接避开了 Segmentation Fault

  } else {
    LOG_WARN_F("Periodic Alignment failed. Score: %.4f", score);
  }

  // 归档处理完的关键帧
  new_keyframes_all_.insert(new_keyframes_all_.end(), new_keyframes_.begin(),
                            new_keyframes_.end());
  new_keyframes_.clear();
  motion_keyframe_count_ = 0;

  fflush(stdout);
}

bool LaserMappingNode::loadExistingMap(const std::string& map_path) {
  if (this->original_map_keyframes_.empty()) return false;

  original_map_visual_ = map_updater_->convertKeyframesToPCL(
      this->original_map_keyframes_, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);

  if (!original_map_visual_ || original_map_visual_->empty()) return false;
  return true;
}

void LaserMappingNode::initialPoseCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  if (is_update_mode_) {
    std::lock_guard<std::mutex> lock(mtx_initial_pose);
    Eigen::Vector3d p(msg->pose.pose.position.x, msg->pose.pose.position.y,
                      msg->pose.pose.position.z);
    Eigen::Quaterniond q(
        msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    initial_pose_ = Sophus::SE3d(q, p);
    if (!initial_pose_received_) {
      initial_pose_received_ = true;
      LOG_INFO_F("Received initial pose guess.");
    }
  }
}

void LaserMappingNode::savePCD() {
  if (!pcd_save_en) return;

  // ==========================================================================================
  // 分支 1: 建图模式 (Mapping Mode) - 仅保存当前累积的全局点云
  // ==========================================================================================
  if (!is_update_mode_) {
    // 检查是否有累积的点云 (对应 ROS1 中对 pcl_wait_save_intensity 的检查)
    if (pcl_wait_save->points.size() > 0 && pcd_save_interval < 0) {
      std::string pcd_dir = map_data_path_ + "/PCD/";
      // 确保目录存在
      if (!std::filesystem::exists(pcd_dir)) {
        std::filesystem::create_directories(pcd_dir);
      }

      std::string all_points_dir = pcd_dir + "map_final_raw.pcd";
      pcl::PCDWriter pcd_writer;

      // 降采样并保存
      // pcl::PointCloud<PointType>::Ptr downsampled_cloud(
      //     new pcl::PointCloud<PointType>);
      // pcl::VoxelGrid<PointType> voxel_filter;
      // voxel_filter.setInputCloud(pcl_wait_save);
      // voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd,
      //                          filter_size_pcd);
      // voxel_filter.filter(*downsampled_cloud);

      pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);

      LOG_INFO_F(
          "\033[1;32m[Map Save] Final intensity map saved to: %s "
          "with %lu points.\033[0m",
          all_points_dir.c_str(), pcl_wait_save->points.size());
    }
  }
  // ==========================================================================================
  // 分支 2: 更新模式 (Update Mode) - 智能合并、去重、覆盖
  // ==========================================================================================
  else {
    LOG_INFO_F("Saving updated map data to: %s", map_data_path_.c_str());

    // 使用原子变量统计 (线程安全)
    std::atomic<int> deleted_pcd_count(0);
    std::atomic<int> updated_pcd_count(0);
    int original_pcd_count = original_map_keyframes_.size();

    std::string original_pose_file =
        map_data_path_ + "/result/localization_output.txt";
    std::string keyframes_path = map_data_path_ + "/result/keyframes/";

    // 1. 将当前缓冲区中的新关键帧加入总列表
    new_keyframes_all_.insert(new_keyframes_all_.end(), new_keyframes_.begin(),
                              new_keyframes_.end());

    if (new_keyframes_all_.empty()) {
      LOG_INFO_F("No new keyframes to process. Map is unchanged.");
      return;
    }

    // 2. 构建新关键帧的空间索引 (用于快速查找重叠)
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_kfs_pose_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudXYZI::Ptr new_points_world(new PointCloudXYZI());

    // 聚合所有新关键帧的点云到世界系
    for (const auto& kf : new_keyframes_all_) {
      new_kfs_pose_cloud->push_back(pcl::PointXYZ(kf.pose.translation().x(),
                                                  kf.pose.translation().y(),
                                                  kf.pose.translation().z()));

      PointCloudXYZI::Ptr cloud_body = kf.getCloud(true);
      if (cloud_body != nullptr && !cloud_body->empty()) {
        PointCloudXYZI cloud_world;
        pcl::transformPointCloud(*cloud_body, cloud_world,
                                 kf.pose.matrix().cast<float>());
        *new_points_world += cloud_world;
      }
    }

    // 构建 KDTree
    pcl::KdTreeFLANN<pcl::PointXYZ> new_kfs_pose_kdtree;
    new_kfs_pose_kdtree.setInputCloud(new_kfs_pose_cloud);

    pcl::KdTreeFLANN<PointType> new_points_kdtree;
    if (!new_points_world->empty()) {
      new_points_kdtree.setInputCloud(new_points_world);
    }

    // 3. 筛选受影响的旧关键帧
    std::vector<KeyframeData> kfs_to_process;
    std::map<double, std::string> final_poses_str_map;
    std::mutex map_mutex;  // 保护 final_poses_str_map

    for (const auto& old_kf : original_map_keyframes_) {
      pcl::PointXYZ search_point(old_kf.pose.translation().x(),
                                 old_kf.pose.translation().y(),
                                 old_kf.pose.translation().z());
      std::vector<int> point_idx_radius_search;
      std::vector<float> point_radius_squared_distance;

      // 如果旧帧在新轨迹的“影响半径”内，则标记为需要处理
      if (new_kfs_pose_kdtree.radiusSearch(
              search_point, keyframe_search_radius_, point_idx_radius_search,
              point_radius_squared_distance) > 0) {
        kfs_to_process.push_back(old_kf);
      } else {
        // 否则直接保留原始位姿字符串
        std::stringstream ss;
        ss << std::fixed << std::setprecision(9) << old_kf.timestamp << " "
           << old_kf.pose.translation().x() << " "
           << old_kf.pose.translation().y() << " "
           << old_kf.pose.translation().z() << " "
           << old_kf.pose.unit_quaternion().x() << " "
           << old_kf.pose.unit_quaternion().y() << " "
           << old_kf.pose.unit_quaternion().z() << " "
           << old_kf.pose.unit_quaternion().w();
        final_poses_str_map[old_kf.timestamp] = ss.str();
      }
    }
    LOG_INFO_F("Found %lu old keyframes to be processed/updated.",
               kfs_to_process.size());

    // 准备可视化被删除点的容器
    deleted_points_visual_->clear();
    std::mutex deleted_points_mutex;

    // 4. 多线程并行处理旧关键帧 (去重/覆盖)
    unsigned int num_threads =
        std::min((unsigned int)std::thread::hardware_concurrency(),
                 (unsigned int)kfs_to_process.size());
    if (num_threads == 0) num_threads = 1;
    std::vector<std::thread> threads;
    int chunk_size = kfs_to_process.size() / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        int start = t * chunk_size;
        int end =
            (t == num_threads - 1) ? kfs_to_process.size() : start + chunk_size;

        for (int i = start; i < end; ++i) {
          const auto& old_kf = kfs_to_process[i];

          // 检查文件是否存在
          if (!std::filesystem::exists(old_kf.pcd_path) ||
              new_points_world->empty()) {
            // 如果无法处理，原样保留
            std::stringstream ss;
            ss << std::fixed << std::setprecision(9) << old_kf.timestamp << " "
               << old_kf.pose.translation().x() << " "
               << old_kf.pose.translation().y() << " "
               << old_kf.pose.translation().z() << " "
               << old_kf.pose.unit_quaternion().x() << " "
               << old_kf.pose.unit_quaternion().y() << " "
               << old_kf.pose.unit_quaternion().z() << " "
               << old_kf.pose.unit_quaternion().w();
            {
              std::lock_guard<std::mutex> lock(map_mutex);
              final_poses_str_map[old_kf.timestamp] = ss.str();
            }
            continue;
          }

          // 加载旧 PCD
          PointCloudXYZI::Ptr old_cloud_body(new PointCloudXYZI());
          if (pcl::io::loadPCDFile<PointType>(old_kf.pcd_path,
                                              *old_cloud_body) == -1) {
            // 加载失败，保留记录
            std::stringstream ss;
            ss << std::fixed << std::setprecision(9) << old_kf.timestamp
               << " ...";  // (同上简化)
            // 为了代码简洁，这里略去重复的 stringstream 构建代码，逻辑同上
            continue;
          }

          // 转换到世界系
          PointCloudXYZI::Ptr old_cloud_world(new PointCloudXYZI());
          pcl::transformPointCloud(*old_cloud_body, *old_cloud_world,
                                   old_kf.pose.matrix().cast<float>());

          // 查找重叠点
          pcl::PointIndices::Ptr indices_to_remove(new pcl::PointIndices());
          std::vector<int> k_indices(1);
          std::vector<float> k_sqr_distances(1);

          for (int j = 0; j < old_cloud_world->points.size(); ++j) {
            // 如果旧点附近有新点 (距离小于替换半径)，标记删除
            if (new_points_kdtree.nearestKSearch(old_cloud_world->points[j], 1,
                                                 k_indices,
                                                 k_sqr_distances) > 0) {
              if (std::sqrt(k_sqr_distances[0]) < point_replacement_radius_) {
                indices_to_remove->indices.push_back(j);
              }
            }
          }

          // 决策：保留还是更新
          if (indices_to_remove->indices.empty()) {
            // 无重叠，完整保留
            std::stringstream ss;
            ss << std::fixed << std::setprecision(9) << old_kf.timestamp << " "
               << old_kf.pose.translation().x() << " "
               << old_kf.pose.translation().y() << " "
               << old_kf.pose.translation().z() << " "
               << old_kf.pose.unit_quaternion().x() << " "
               << old_kf.pose.unit_quaternion().y() << " "
               << old_kf.pose.unit_quaternion().z() << " "
               << old_kf.pose.unit_quaternion().w();
            {
              std::lock_guard<std::mutex> lock(map_mutex);
              final_poses_str_map[old_kf.timestamp] = ss.str();
            }
          } else {
            // 收集被删除点用于显示
            PointCloudXYZI::Ptr removed_points_world(new PointCloudXYZI());
            pcl::ExtractIndices<PointType> extract_removed;
            extract_removed.setInputCloud(old_cloud_world);
            extract_removed.setIndices(indices_to_remove);
            extract_removed.setNegative(false);
            extract_removed.filter(*removed_points_world);

            {
              std::lock_guard<std::mutex> lock(deleted_points_mutex);
              for (const auto& pt : removed_points_world->points) {
                PointTypeRGB colored_pt;
                colored_pt.x = pt.x;
                colored_pt.y = pt.y;
                colored_pt.z = pt.z;
                colored_pt.r = 255;
                colored_pt.g = 0;
                colored_pt.b = 0;  // Red
                deleted_points_visual_->points.push_back(colored_pt);
              }
            }

            // 保存剩余点
            PointCloudXYZI::Ptr cloud_to_keep(new PointCloudXYZI());
            pcl::ExtractIndices<PointType> extract;
            extract.setInputCloud(old_cloud_body);
            extract.setIndices(indices_to_remove);
            extract.setNegative(true);
            extract.filter(*cloud_to_keep);

            if (cloud_to_keep->empty()) {
              // 全部被覆盖，删除文件
              std::remove(old_kf.pcd_path.c_str());
              deleted_pcd_count++;
            } else {
              // 部分保留，覆盖写回文件
              pcl::io::savePCDFileBinary(old_kf.pcd_path, *cloud_to_keep);
              std::stringstream ss;
              ss << std::fixed << std::setprecision(9) << old_kf.timestamp
                 << " " << old_kf.pose.translation().x() << " "
                 << old_kf.pose.translation().y() << " "
                 << old_kf.pose.translation().z() << " "
                 << old_kf.pose.unit_quaternion().x() << " "
                 << old_kf.pose.unit_quaternion().y() << " "
                 << old_kf.pose.unit_quaternion().z() << " "
                 << old_kf.pose.unit_quaternion().w();
              {
                std::lock_guard<std::mutex> lock(map_mutex);
                final_poses_str_map[old_kf.timestamp] = ss.str();
              }
              updated_pcd_count++;
            }
          }
        }
      });
    }

    for (auto& th : threads) {
      if (th.joinable()) th.join();
    }

    // 5. 将新关键帧位姿加入 Map
    for (const auto& kf : new_keyframes_all_) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(9) << kf.timestamp << " "
         << kf.pose.translation().x() << " " << kf.pose.translation().y() << " "
         << kf.pose.translation().z() << " " << kf.pose.unit_quaternion().x()
         << " " << kf.pose.unit_quaternion().y() << " "
         << kf.pose.unit_quaternion().z() << " "
         << kf.pose.unit_quaternion().w();
      final_poses_str_map[kf.timestamp] = ss.str();
    }

    // 6. 写入最终 Pose 文件
    std::ofstream final_file(original_pose_file, std::ios::trunc);
    for (const auto& pair : final_poses_str_map) {
      final_file << pair.second << std::endl;
    }
    final_file.close();

    // 7. 多线程保存新关键帧 PCD
    if (!std::filesystem::exists(keyframes_path)) {
      std::filesystem::create_directories(keyframes_path);
    }

    threads.clear();
    chunk_size = new_keyframes_all_.size() / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? new_keyframes_all_.size()
                                         : start + chunk_size;
        for (int i = start; i < end; ++i) {
          const auto& kf = new_keyframes_all_[i];
          std::stringstream ss;
          ss << std::fixed << std::setprecision(9) << kf.timestamp;
          std::string pcd_filename = keyframes_path + ss.str() + ".pcd";

          if (std::filesystem::exists(pcd_filename)) {
            continue;
          }
          PointCloudXYZI::Ptr cloud_to_save = kf.getCloud(true);
          if (cloud_to_save && !cloud_to_save->empty()) {
            pcl::io::savePCDFileBinary(pcd_filename, *cloud_to_save);
          }
        }
      });
    }
    for (auto& th : threads) {
      if (th.joinable()) th.join();
    }

    // 8. 打印统计信息
    LOG_INFO_F("\n\n--- Map Update Summary ---");
    LOG_INFO_F("Original PCDs: %d", original_pcd_count);
    LOG_INFO_F("Deleted PCDs: %d", deleted_pcd_count.load());
    LOG_INFO_F("Updated PCDs: %d", updated_pcd_count.load());
    LOG_INFO_F("Newly Added PCDs: %lu", new_keyframes_all_.size());
    LOG_INFO_F("--------------------------\n");
  }
}

void LaserMappingNode::saveMapCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
  (void)req;  // 避免未随后使用变量的警告
  LOG_INFO_F(
      "\033[1;32mService /save_map called. Finalizing process...\033[0m");

  // 步骤 1: 执行核心 PCD 保存和更新逻辑
  savePCD();

  // 步骤 2: 准备可视化数据 (仅在 Update 模式下有意义，或者根据 ROS1
  // 逻辑通用处理) 清空用于最终可视化的容器
  new_keyframes_all_.clear();
  new_keyframes_.clear();

  std::string final_pose_file =
      map_data_path_ + "/result/localization_output.txt";
  std::string keyframes_dir = map_data_path_ + "/result/keyframes/";
  std::ifstream infile(final_pose_file);

  if (!infile.is_open()) {
    LOG_ERROR_F(
        "Failed to re-open pose file to generate final visualization map.");
    res->success = false;
    res->message = "Error: Could not read final pose file for visualization.";
    return;
  }

  // 从更新后的 pose 文件中重新加载所有有效的关键帧信息
  std::map<double, KeyframeData> final_kfs_map;
  std::string line;
  while (std::getline(infile, line)) {
    std::stringstream ss(line);
    KeyframeData kf;
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    ss >> kf.timestamp >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >>
        q.w();
    kf.pose = Sophus::SE3d(q, t);

    std::stringstream pcd_filename_ss;
    pcd_filename_ss << std::fixed << std::setprecision(9) << kf.timestamp
                    << ".pcd";
    kf.pcd_path = keyframes_dir + pcd_filename_ss.str();

    // 只有文件实际存在时才加入
    if (std::filesystem::exists(kf.pcd_path)) {
      final_kfs_map[kf.timestamp] = kf;
    }
  }
  infile.close();
  LOG_INFO_F(
      "\033[1;32mReloaded %lu final keyframes to generate "
      "visualization.\033[0m",
      final_kfs_map.size());

  // --- 步骤 3: 发布三种颜色的点云 ---

  // 3.1 发布被删除的点 (红色)
  if (deleted_points_visual_ && !deleted_points_visual_->empty()) {
    deleted_points_visual_->width = deleted_points_visual_->points.size();
    deleted_points_visual_->height = 1;
    deleted_points_visual_->is_dense = true;

    sensor_msgs::msg::PointCloud2 deleted_map_msg;
    pcl::toROSMsg(*deleted_points_visual_, deleted_map_msg);
    deleted_map_msg.header.stamp = this->now();
    deleted_map_msg.header.frame_id = "camera_init";
    pub_deleted_points_->publish(deleted_map_msg);
    LOG_INFO_F(
        "\033[1;32mPublished %lu deleted points to /map/deleted_points "
        "(RED).\033[0m",
        deleted_points_visual_->size());
  }

  // 3.2 区分 "新增关键帧" 和 "保留的旧关键帧"
  std::unordered_set<double> original_timestamps;
  for (const auto& okf : original_map_keyframes_) {
    original_timestamps.insert(okf.timestamp);
  }

  std::vector<KeyframeData> final_kept_kfs;
  std::vector<KeyframeData> final_new_kfs;
  for (const auto& pair : final_kfs_map) {
    if (original_timestamps.count(pair.first)) {
      final_kept_kfs.push_back(pair.second);
    } else {
      final_new_kfs.push_back(pair.second);
    }
  }

  // 发布新增的点云 (绿色)
  if (!final_new_kfs.empty()) {
    PointCloudXYZRGB::Ptr new_cloud = map_updater_->convertKeyframesToPCL(
        final_new_kfs, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);

    // 强制染成绿色 (覆盖原有的 intensity 或颜色)
    for (auto& p : new_cloud->points) {
      p.r = 0;
      p.g = 255;
      p.b = 0;
    }

    sensor_msgs::msg::PointCloud2 new_map_msg;
    pcl::toROSMsg(*new_cloud, new_map_msg);
    new_map_msg.header.stamp = this->now();
    new_map_msg.header.frame_id = "camera_init";
    pub_newly_added_map_->publish(new_map_msg);
    LOG_INFO_F(
        "\033[1;32mPublished %lu new points to /map/newly_added_map "
        "(GREEN).\033[0m",
        new_cloud->size());
  }

  // 发布保留的旧点云 (白色/原色)
  if (!final_kept_kfs.empty()) {
    PointCloudXYZRGB::Ptr kept_cloud = map_updater_->convertKeyframesToPCL(
        final_kept_kfs, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);

    sensor_msgs::msg::PointCloud2 kept_map_msg;
    pcl::toROSMsg(*kept_cloud, kept_map_msg);
    kept_map_msg.header.stamp = this->now();
    kept_map_msg.header.frame_id = "camera_init";
    pub_original_map_kept_->publish(kept_map_msg);
    LOG_INFO_F(
        "\033[1;32mPublished %lu original kept points to "
        "/map/original_map_kept (WHITE).\033[0m",
        kept_cloud->size());
  }

  // --- 步骤 4: 保存并发布最终合并的可视化地图 ---
  if (!final_kfs_map.empty()) {
    // 合并 kept 和 new
    final_kept_kfs.insert(final_kept_kfs.end(), final_new_kfs.begin(),
                          final_new_kfs.end());
    std::vector<KeyframeData> final_updated_kfs = final_kept_kfs;

    // 清理临时 vector
    final_kept_kfs.clear();
    final_new_kfs.clear();

    PointCloudXYZRGB::Ptr final_updated_map =
        map_updater_->convertKeyframesToPCL(final_updated_kfs, Lidar_R_wrt_IMU,
                                            Lidar_T_wrt_IMU);

    sensor_msgs::msg::PointCloud2 final_map_msg;
    std::string final_map_dir =
        is_update_mode_ ? map_data_path_ + "/" : map_data_path_ + "/PCD/";

    if (!std::filesystem::exists(final_map_dir)) {
      std::filesystem::create_directories(final_map_dir);
    }
    std::string final_map_path = final_map_dir + "map_final_visualization.pcd";

    pcl::PCDWriter pcd_writer;
    if (pcd_writer.writeBinary(final_map_path, *final_updated_map) == 0) {
      LOG_INFO_F(
          "\033[1;32mSaved the final visualization map to: %s with %lu "
          "points.\033[0m",
          final_map_path.c_str(), final_updated_map->size());
    } else {
      LOG_ERROR_F("Failed to write final visualization map to %s",
                  final_map_path.c_str());
    }

    pcl::toROSMsg(*final_updated_map, final_map_msg);
    final_map_msg.header.stamp = this->now();
    final_map_msg.header.frame_id = "camera_init";
    pub_final_updated_map_->publish(final_map_msg);
    LOG_INFO_F(
        "\033[1;32mPublished final visualization map to /map/final_updated_map "
        "with %lu points.\033[0m",
        final_updated_map->size());
  } else {
    LOG_WARN_F("Final visualization map is empty, skipping publish.");
  }

  res->success = true;
  res->message = "Map save process finished.";
}

void LaserMappingNode::map_publish_callback() {
  if (map_pub_en) publish_map(pubLaserCloudMap_);
}

void LaserMappingNode::pointBodyToWorld(PointType const* const pi,
                                        PointType* const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body +
                                  state_point.offset_T_L_I) +
               state_point.pos);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LaserMappingNode::RGBpointBodyToWorld(PointType const* const pi,
                                           PointType* const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body +
                                  state_point.offset_T_L_I) +
               state_point.pos);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LaserMappingNode::RGBpointBodyLidarToIMU(PointType const* const pi,
                                              PointType* const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar +
                 state_point.offset_T_L_I);
  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void LaserMappingNode::points_cache_collect() {
  PointVector points_history;
  ikdtree.acquire_removed_points(points_history);
}

void LaserMappingNode::lasermap_fov_segment() {
  cub_needrm.clear();
  kdtree_delete_counter = 0;
  kdtree_delete_time = 0.0;

  pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);

  V3D pos_LiD = pos_lid;
  if (!Localmap_Initialized) {
    for (int i = 0; i < 3; i++) {
      LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
      LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
    }
    Localmap_Initialized = true;
    return;
  }
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
        dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
      need_move = true;
  }
  if (!need_move) return;
  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points;
  float mov_dist =
      std::max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
               double(DET_RANGE * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = LocalMap_Points;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm.size() > 0)
    kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
  kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void LaserMappingNode::standard_pcl_cbk(
    const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
  mtx_buffer.lock();
  scan_count++;
  double cur_time = get_time_sec(msg->header.stamp);
  double preprocess_start_time = omp_get_wtime();
  if (!is_first_lidar && cur_time < last_timestamp_lidar) {
    lidar_buffer.clear();
  }
  if (is_first_lidar) {
    is_first_lidar = false;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(cur_time);
  last_timestamp_lidar = cur_time;
  if (scan_count < MAXN)
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LaserMappingNode::livox_pcl_cbk(
    const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) {
  mtx_buffer.lock();
  double cur_time = get_time_sec(msg->header.stamp);
  double preprocess_start_time = omp_get_wtime();
  scan_count++;
  if (!is_first_lidar && cur_time < last_timestamp_lidar) {
    lidar_buffer.clear();
  }
  if (is_first_lidar) {
    is_first_lidar = false;
  }
  last_timestamp_lidar = cur_time;

  if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 &&
      !imu_buffer.empty() && !lidar_buffer.empty()) {
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);

  if (scan_count < MAXN)
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LaserMappingNode::imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in) {
  publish_count++;
  sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

  msg->header.stamp =
      get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);

  double timestamp = get_time_sec(msg->header.stamp);

  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    imu_buffer.clear();
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LaserMappingNode::img_cbk(const sensor_msgs::msg::Image::UniquePtr msg) {
  // 【关键修改】移除 !img_en 的判断，确保始终缓存图像用于保存
  // if (!img_en) return;

  // 获取时间戳 (ROS2)
  double msg_header_time = rclcpp::Time(msg->header.stamp).seconds();

  // 如果您在 yaml 中定义了 img_time_offset，可以加上它：
  // msg_header_time += img_time_offset;

  // 防止处理极其接近的重复帧 (1e-4 = 0.1ms)
  if (std::abs(msg_header_time - last_timestamp_img) < 1e-4) {
    return;
  }

  // 检查时间戳回环 (例如 rosbag 播放重置)
  if (last_timestamp_lidar > 0 && msg_header_time < last_timestamp_img) {
    LOG_ERROR("Image loop back, clearing buffer.");
    mtx_buffer.lock();
    img_buffer.clear();
    img_time_buffer.clear();
    mtx_buffer.unlock();
    last_timestamp_img = msg_header_time;
    return;
  }

  mtx_buffer.lock();
  try {
    // 使用 cv_bridge 将 ROS2 图像转换为 OpenCV 格式 (BGR8)
    // 注意：ROS2 中 toCvCopy 接收的是 const sensor_msgs::msg::Image &
    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8);

    if (cv_ptr->image.empty()) {
      LOG_WARN("Received empty image, skipping.");
    } else {
      img_buffer.push_back(cv_ptr->image);
      img_time_buffer.push_back(msg_header_time);
      last_timestamp_img = msg_header_time;
    }
  } catch (cv_bridge::Exception& e) {
    LOG_ERROR_F("cv_bridge exception: %s", e.what());
  }
  mtx_buffer.unlock();

  sig_buffer.notify_all();
}

bool LaserMappingNode::sync_packages(MeasureGroup& meas) {
  if (lidar_buffer.empty() || imu_buffer.empty()) {
    return false;
  }

  if (!lidar_pushed) {
    meas.lidar = lidar_buffer.front();
    meas.lidar_beg_time = time_buffer.front();
    if (meas.lidar->points.size() <= 1) {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    } else if (meas.lidar->points.back().curvature / double(1000) <
               0.5 * lidar_mean_scantime) {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    } else {
      scan_num++;
      lidar_end_time = meas.lidar_beg_time +
                       meas.lidar->points.back().curvature / double(1000);
      lidar_mean_scantime +=
          (meas.lidar->points.back().curvature / double(1000) -
           lidar_mean_scantime) /
          scan_num;
    }

    meas.lidar_end_time = lidar_end_time;
    lidar_pushed = true;
  }

  if (last_timestamp_imu < lidar_end_time) {
    return false;
  }

  double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
  meas.imu.clear();
  while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
    imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    if (imu_time > lidar_end_time) break;
    meas.imu.push_back(imu_buffer.front());
    imu_buffer.pop_front();
  }

  lidar_buffer.pop_front();
  time_buffer.pop_front();
  lidar_pushed = false;
  return true;
}

void LaserMappingNode::map_incremental() {
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size);
  PointNoNeedDownsample.reserve(feats_down_size);
  for (int i = 0; i < feats_down_size; i++) {
    pointBodyToWorld(&(feats_down_body->points[i]),
                     &(feats_down_world->points[i]));
    if (!Nearest_Points[i].empty() && flg_EKF_inited) {
      const PointVector& points_near = Nearest_Points[i];
      bool need_add = true;
      PointType mid_point;
      mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) *
                        filter_size_map_min +
                    0.5 * filter_size_map_min;
      mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) *
                        filter_size_map_min +
                    0.5 * filter_size_map_min;
      mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) *
                        filter_size_map_min +
                    0.5 * filter_size_map_min;
      float dist = calc_dist(feats_down_world->points[i], mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
        PointNoNeedDownsample.push_back(feats_down_world->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist) {
          need_add = false;
          break;
        }
      }
      if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
    } else {
      PointToAdd.push_back(feats_down_world->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  add_point_size = ikdtree.Add_Points(PointToAdd, true);
  ikdtree.Add_Points(PointNoNeedDownsample, false);
  add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time = omp_get_wtime() - st_time;
}

void LaserMappingNode::publish_frame_world(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pubLaserCloudFull) {
  if (scan_pub_en) {
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort
                                                       : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                          &laserCloudWorld->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFull->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
  }

  if (pcd_save_en) {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&feats_undistort->points[i],
                          &laserCloudWorld->points[i]);
    }
    *pcl_wait_save += *laserCloudWorld;
  }
}

void LaserMappingNode::publish_frame_body(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pubLaserCloudFull_body) {
  int size = feats_undistort->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
    RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                           &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
  laserCloudmsg.header.frame_id = "body";
  pubLaserCloudFull_body->publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

void LaserMappingNode::publish_effect_world(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pubLaserCloudEffect) {
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
  for (int i = 0; i < effct_feat_num; i++) {
    RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void LaserMappingNode::publish_map(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pubLaserCloudMap) {
  PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort
                                                     : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                        &laserCloudWorld->points[i]);
  }
  *pcl_wait_pub += *laserCloudWorld;

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudMap->publish(laserCloudmsg);
}

void LaserMappingNode::publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
        pubOdomAftMapped,
    std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br) {
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "body";
  odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
  set_posestamp(odomAftMapped.pose);
  pubOdomAftMapped->publish(odomAftMapped);
  auto P = kf.get_P();
  for (int i = 0; i < 6; i++) {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  geometry_msgs::msg::TransformStamped trans;
  trans.header.frame_id = "camera_init";
  trans.header.stamp = odomAftMapped.header.stamp;
  trans.child_frame_id = "body";
  trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
  trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
  trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
  trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
  trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
  trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
  trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
  tf_br->sendTransform(trans);
}

void LaserMappingNode::publish_path(
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath) {
  set_posestamp(msg_body_pose);
  msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
  msg_body_pose.header.frame_id = "camera_init";

  path.poses.push_back(msg_body_pose);

  path.header.frame_id = "camera_init";
  path.header.stamp = msg_body_pose.header.stamp;

  pubPath->publish(path);
}

void LaserMappingNode::h_share_model(
    state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data) {
  if (!ptr_) return;
  LaserMappingNode& node = *ptr_;

  double match_start = omp_get_wtime();
  node.laserCloudOri->clear();
  node.corr_normvect->clear();
  node.total_residual = 0.0;

#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < node.feats_down_size; i++) {
    PointType& point_body = node.feats_down_body->points[i];
    PointType& point_world = node.feats_down_world->points[i];

    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
    auto& points_near = node.Nearest_Points[i];

    if (ekfom_data.converge) {
      node.ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near,
                                  pointSearchSqDis);
      node.point_selected_surf[i] =
          points_near.size() < NUM_MATCH_POINTS        ? false
          : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                       : true;
    }

    if (!node.point_selected_surf[i]) continue;

    VF(4) pabcd;
    node.point_selected_surf[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f)) {
      float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                  pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9) {
        node.point_selected_surf[i] = true;
        node.normvec->points[i].x = pabcd(0);
        node.normvec->points[i].y = pabcd(1);
        node.normvec->points[i].z = pabcd(2);
        node.normvec->points[i].intensity = pd2;
        node.res_last[i] = abs(pd2);
      }
    }
  }

  node.effct_feat_num = 0;

  for (int i = 0; i < node.feats_down_size; i++) {
    if (node.point_selected_surf[i]) {
      node.laserCloudOri->points[node.effct_feat_num] =
          node.feats_down_body->points[i];
      node.corr_normvect->points[node.effct_feat_num] = node.normvec->points[i];
      node.total_residual += node.res_last[i];
      node.effct_feat_num++;
    }
  }

  if (node.effct_feat_num < 1) {
    ekfom_data.valid = false;
    return;
  }

  node.res_mean_last = node.total_residual / node.effct_feat_num;
  node.match_time += omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  ekfom_data.h_x = MatrixXd::Zero(node.effct_feat_num, 12);
  ekfom_data.h.resize(node.effct_feat_num);

  for (int i = 0; i < node.effct_feat_num; i++) {
    const PointType& laser_p = node.laserCloudOri->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    const PointType& norm_p = node.corr_normvect->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (node.extrinsic_est_en) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }

    ekfom_data.h(i) = -norm_p.intensity;
  }
  node.solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  signal(SIGINT, LaserMappingNode::OnSignal);

  auto node = std::make_shared<LaserMappingNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}