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
  memset(point_selected_surf, true, sizeof(point_selected_surf));
  memset(res_last, -1000.0f, sizeof(res_last));

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
  pubOdomAftMapped_ =
      this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
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
  if (sync_packages(Measures)) {
    if (flg_first_scan) {
      first_lidar_time = Measures.lidar_beg_time;
      p_imu->first_lidar_time = first_lidar_time;
      flg_first_scan = false;
      return;
    }

    double t0, t1, t2, t3, t5;
    match_time = 0;
    kdtree_search_time = 0.0;
    solve_time = 0;
    solve_const_H_time = 0;
    t0 = omp_get_wtime();

    p_imu->Process(Measures, kf, feats_undistort);
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    if (feats_undistort->empty() || (feats_undistort == NULL)) {
      return;
    }

    flg_EKF_inited =
        (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

    lasermap_fov_segment();

    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
    t1 = omp_get_wtime();
    feats_down_size = feats_down_body->points.size();

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

    // --- Update Mode: Initial Alignment Logic ---
    LOG_INFO_F("is_update_mode_: %d, initial_align_finished_: %d",
               is_update_mode_, initial_align_finished_);
    if (is_update_mode_) {
      if (!initial_align_finished_) {
        if (!initial_pose_received_) {
          static int log_cnt = 0;
          if (log_cnt++ % 50 == 0)
            LOG_INFO_F("Waiting for initial pose from /pcl_pose...");
          return;
        }

        LOG_INFO_F("Performing initial ICP alignment using pose guess...");
        PointCloudXYZI::Ptr current_frame_world(new PointCloudXYZI());

        Eigen::Matrix4d T_guess = initial_pose_.matrix();
        pcl::transformPointCloud(*feats_undistort, *current_frame_world,
                                 T_guess);

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
        icp.setInputSource(current_frame_world);
        icp.setInputTarget(target_cloud);
        icp.setMaxCorrespondenceDistance(1.0);
        icp.setMaximumIterations(100);

        PointCloudXYZI unused_result;
        icp.align(unused_result);

        if (icp.hasConverged()) {
          Eigen::Matrix4d T_correction =
              icp.getFinalTransformation().cast<double>();
          Sophus::SE3d refined_pose =
              Sophus::SE3d(T_correction) * initial_pose_;

          state_point.rot = refined_pose.unit_quaternion();
          state_point.pos = refined_pose.translation();
          kf.change_x(state_point);

          initial_align_finished_ = true;
          LOG_INFO_F("\033[1;32mFirst Alignment successful! Fitness:%f\033[0m",
                     icp.getFitnessScore());

          ikdtree = KD_TREE<PointType>();
        } else {
          LOG_WARN_F("Initial ICP failed.");
          return;
        }
      }
    }
    // --------------------------------------------

    kdtree_size_st = ikdtree.size();

    if (feats_down_size < 5) {
      return;
    }

    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);

    V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
    fout_pre << std::setw(20) << Measures.lidar_beg_time - first_lidar_time
             << " " << euler_cur.transpose() << " "
             << state_point.pos.transpose() << " " << ext_euler.transpose()
             << " " << state_point.offset_T_L_I.transpose() << " "
             << state_point.vel.transpose() << " " << state_point.bg.transpose()
             << " " << state_point.ba.transpose() << " " << state_point.grav
             << std::endl;

    pointSearchInd_surf.resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);

    t2 = omp_get_wtime();

    /*** iterated state estimation ***/
    double t_update_start = omp_get_wtime();
    double solve_H_time = 0;

    kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

    state_point = kf.get_x();
    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];

    double t_update_end = omp_get_wtime();

    publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

    t3 = omp_get_wtime();
    map_incremental();
    t5 = omp_get_wtime();

    // --- Update Mode: Periodic Alignment Logic ---
    if (is_update_mode_ && initial_align_finished_) {
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
      periodicAlignment();
    }
    // ----------------------------------------

    if (path_en) publish_path(pubPath_);
    if (scan_pub_en) publish_frame_world(pubLaserCloudFull_);
    if (scan_pub_en && scan_body_pub_en)
      publish_frame_body(pubLaserCloudFull_body_);
    if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);

    if (voxelmap_manager &&
        voxelmap_manager->config_setting_.is_pub_plane_map_) {
      voxelmap_manager->pubVoxelMap(this->now());
    }

    // 建图模式下直接保存 (无复杂逻辑)
    if (pcd_save_en && pcd_save_interval == -1 && !is_update_mode_) {
      double dist = (state_point.pos - last_pcd_save_pos_).norm();
      if (!is_first_pcd_saved_ || dist > pcd_save_distance_thresh_) {
        last_pcd_save_pos_ = state_point.pos;
        is_first_pcd_saved_ = true;

        std::stringstream ss;
        ss << std::fixed << std::setprecision(6) << Measures.lidar_beg_time;
        std::string ts_str = ss.str();
        std::string pcd_filename =
            map_data_path_ + "/result/keyframes/" + ts_str + ".pcd";

        std::filesystem::create_directories(map_data_path_ +
                                            "/result/keyframes/");
        pcl::io::savePCDFileBinary(pcd_filename, *feats_undistort);

        fout_pcd_pos << std::fixed << std::setprecision(9)
                     << Measures.lidar_beg_time << " " << state_point.pos.x()
                     << " " << state_point.pos.y() << " " << state_point.pos.z()
                     << " " << geoQuat.x << " " << geoQuat.y << " " << geoQuat.z
                     << " " << geoQuat.w << std::endl;
      }
    }
  }
}

void LaserMappingNode::periodicAlignment() {
  if (!is_update_mode_ || motion_keyframe_count_ < periodic_align_interval_) {
    return;
  }

  PointCloudXYZI::Ptr source_cloud_world(new PointCloudXYZI());
  for (const auto& kf_new : new_keyframes_) {
    PointCloudXYZI::Ptr kf_cloud = kf_new.getCloud();
    if (kf_cloud->empty()) continue;
    PointCloudXYZI::Ptr transformed_cloud(new PointCloudXYZI());
    Eigen::Matrix4d T = kf_new.pose.matrix();
    pcl::transformPointCloud(*kf_cloud, *transformed_cloud, T);
    *source_cloud_world += *transformed_cloud;
  }

  if (source_cloud_world->empty()) return;

  PointCloudXYZI::Ptr target_cloud = map_updater_->findTargetPointsForICP(
      new_keyframes_, voxel_map_index, original_map_keyframes_,
      periodic_align_min_dist_);

  if (target_cloud->empty()) return;

  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setInputSource(source_cloud_world);
  icp.setInputTarget(target_cloud);
  icp.setMaxCorrespondenceDistance(1.5);
  icp.setMaximumIterations(50);

  PointCloudXYZI unused_result;
  icp.align(unused_result);

  if (icp.hasConverged() && icp.getFitnessScore() < 0.5) {
    Eigen::Matrix4d correction = icp.getFinalTransformation().cast<double>();
    Sophus::SE3d T_map_slam(
        Eigen::Quaterniond(correction.block<3, 3>(0, 0).cast<double>()),
        correction.block<3, 1>(0, 3).cast<double>());

    for (auto& kf : new_keyframes_) {
      kf.pose = T_map_slam * kf.pose;
    }

    Sophus::SE3d current_pose(state_point.rot.toRotationMatrix(),
                              state_point.pos);
    Sophus::SE3d corrected_pose = T_map_slam * current_pose;
    state_point.rot = corrected_pose.unit_quaternion();
    state_point.pos = corrected_pose.translation();

    kf.change_x(state_point);

    ikdtree = KD_TREE<PointType>();

    LOG_INFO_F(
        "\033[1;32mPeriodic Alignment successful! Refinement applied.\033[0m");
  } else {
    LOG_WARN_F("Periodic Alignment failed to converge.");
  }

  new_keyframes_all_.insert(new_keyframes_all_.end(), new_keyframes_.begin(),
                            new_keyframes_.end());
  new_keyframes_.clear();
  motion_keyframe_count_ = 0;
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

  // --- 1. 更新模式：执行智能地图维护 ---
  if (is_update_mode_) {
    LOG_INFO_F("Finalizing map update...");

    // 将缓冲区中的新关键帧加入总列表
    new_keyframes_all_.insert(new_keyframes_all_.end(), new_keyframes_.begin(),
                              new_keyframes_.end());
    if (new_keyframes_all_.empty()) {
      LOG_INFO_F("No new frames to save.");
      return;
    }

    std::atomic<int> deleted_pcd_count(0);
    std::atomic<int> updated_pcd_count(0);

    // 1.1 构建新关键帧的 KDTree，用于快速查找受影响的区域
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_kfs_pose_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudXYZI::Ptr new_points_world(new PointCloudXYZI());

    for (const auto& kf : new_keyframes_all_) {
      new_kfs_pose_cloud->push_back(pcl::PointXYZ(kf.pose.translation().x(),
                                                  kf.pose.translation().y(),
                                                  kf.pose.translation().z()));
      PointCloudXYZI::Ptr body = kf.getCloud(true);
      if (body) {
        PointCloudXYZI world;
        pcl::transformPointCloud(*body, world, kf.pose.matrix());
        *new_points_world += world;
      }
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> new_kfs_kdtree;
    new_kfs_kdtree.setInputCloud(new_kfs_pose_cloud);

    pcl::KdTreeFLANN<PointType> new_points_kdtree;
    if (!new_points_world->empty())
      new_points_kdtree.setInputCloud(new_points_world);

    // 1.2 筛选出需要检查重叠的旧关键帧
    std::vector<KeyframeData> kfs_to_check;
    std::map<double, std::string> final_pose_map;  // 使用 map 保持时间顺序
    std::mutex map_mtx;                            // 保护 final_pose_map

    for (const auto& old_kf : original_map_keyframes_) {
      pcl::PointXYZ pt(old_kf.pose.translation().x(),
                       old_kf.pose.translation().y(),
                       old_kf.pose.translation().z());
      std::vector<int> indices;
      std::vector<float> sqr_dists;

      // 只检查新轨迹半径内的旧帧
      if (new_kfs_kdtree.radiusSearch(pt, keyframe_search_radius_, indices,
                                      sqr_dists) > 0) {
        kfs_to_check.push_back(old_kf);
      } else {
        // 距离较远，原样保留
        std::stringstream ss;
        auto t = old_kf.pose.translation();
        auto q = old_kf.pose.unit_quaternion();
        ss << std::fixed << std::setprecision(9) << old_kf.timestamp << " "
           << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
           << q.y() << " " << q.z() << " " << q.w();
        final_pose_map[old_kf.timestamp] = ss.str();
      }
    }

// 1.3 并行处理旧关键帧：移除被新点云覆盖的点
#pragma omp parallel for
    for (int i = 0; i < kfs_to_check.size(); ++i) {
      const auto& old_kf = kfs_to_check[i];
      PointCloudXYZI::Ptr old_cloud_world(new PointCloudXYZI());
      PointCloudXYZI::Ptr old_cloud_body = old_kf.getCloud();
      if (!old_cloud_body || old_cloud_body->empty()) continue;

      pcl::transformPointCloud(*old_cloud_body, *old_cloud_world,
                               old_kf.pose.matrix());

      pcl::PointIndices::Ptr rm_indices(new pcl::PointIndices);
      std::vector<int> k_idx(1);
      std::vector<float> k_sqr_dist(1);

      // 对旧帧中的每个点，检查是否被新点云“覆盖”
      for (int j = 0; j < old_cloud_world->size(); ++j) {
        if (new_points_kdtree.nearestKSearch(old_cloud_world->points[j], 1,
                                             k_idx, k_sqr_dist) > 0) {
          if (std::sqrt(k_sqr_dist[0]) < point_replacement_radius_) {
            rm_indices->indices.push_back(j);
          }
        }
      }

      if (!rm_indices->indices.empty()) {
        // 收集被删除的点用于可视化 (注意线程安全)
        PointCloudXYZI removed_world;
        pcl::ExtractIndices<PointType> ext;
        ext.setInputCloud(old_cloud_world);
        ext.setIndices(rm_indices);
        ext.setNegative(false);  // 提取被删除的点
        ext.filter(removed_world);

#pragma omp critical
        {
          for (auto& p : removed_world.points) {
            PointTypeRGB pr;
            pr.x = p.x;
            pr.y = p.y;
            pr.z = p.z;
            pr.r = 255;
            pr.g = 0;
            pr.b = 0;  // 红色
            deleted_points_visual_->push_back(pr);
          }
        }

        // 保存剩余点到原文件
        PointCloudXYZI kept_body;
        ext.setInputCloud(old_cloud_body);
        ext.setNegative(true);  // 保留未被删除的点
        ext.filter(kept_body);

        if (kept_body.empty()) {
          std::remove(old_kf.pcd_path.c_str());
          deleted_pcd_count++;
        } else {
          pcl::io::savePCDFileBinary(old_kf.pcd_path, kept_body);
          std::stringstream ss;
          auto t = old_kf.pose.translation();
          auto q = old_kf.pose.unit_quaternion();
          ss << std::fixed << std::setprecision(9) << old_kf.timestamp << " "
             << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
             << q.y() << " " << q.z() << " " << q.w();

#pragma omp critical
          {
            final_pose_map[old_kf.timestamp] = ss.str();
          }
          updated_pcd_count++;
        }
      } else {
        // 无重叠，原样保留
        std::stringstream ss;
        auto t = old_kf.pose.translation();
        auto q = old_kf.pose.unit_quaternion();
        ss << std::fixed << std::setprecision(9) << old_kf.timestamp << " "
           << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
           << q.y() << " " << q.z() << " " << q.w();

#pragma omp critical
        {
          final_pose_map[old_kf.timestamp] = ss.str();
        }
      }
    }

    // 1.4 保存新关键帧
    std::string keyframes_path = map_data_path_ + "/result/keyframes/";
    for (const auto& kf : new_keyframes_all_) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(9) << kf.timestamp;
      std::string name = ss.str();
      pcl::io::savePCDFileBinary(keyframes_path + name + ".pcd",
                                 *kf.getCloud(true));

      std::stringstream ss_pose;
      auto t = kf.pose.translation();
      auto q = kf.pose.unit_quaternion();
      ss_pose << std::fixed << std::setprecision(9) << kf.timestamp << " "
              << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
              << q.y() << " " << q.z() << " " << q.w();
      final_pose_map[kf.timestamp] = ss_pose.str();
    }

    // 1.5 写入最终的 pose 文件
    std::ofstream of(map_data_path_ + "/result/localization_output.txt");
    for (auto const& [ts, line] : final_pose_map) {
      of << line << std::endl;
    }
    of.close();

    LOG_INFO_F(
        "Map Update Complete. Deleted PCDs: %d, Updated PCDs: %d, Added PCDs: "
        "%lu",
        deleted_pcd_count.load(), updated_pcd_count.load(),
        new_keyframes_all_.size());

  } else {
    // --- 2. 建图模式：简单保存 ---
    std::string raw_points_dir = map_data_path_ + "/PCD/all_raw_points.pcd";
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save);
    LOG_INFO_F("Saved all raw points to %s", raw_points_dir.c_str());
  }
}

void LaserMappingNode::saveMapCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
  (void)req;
  LOG_INFO_F("Service /save_map called. Finalizing map update process...");

  // 步骤 1: 执行核心更新逻辑 (删除重叠、保存新帧)
  savePCD();

  // 步骤 2: 可视化更新结果 (仅在更新模式下)
  if (is_update_mode_) {
    // 2.1 发布被删除的点 (红色)
    if (deleted_points_visual_ && !deleted_points_visual_->empty()) {
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(*deleted_points_visual_, msg);
      msg.header.stamp = this->now();
      msg.header.frame_id = "camera_init";
      pub_deleted_points_->publish(msg);
      LOG_INFO_F("Published deleted points visualization (RED).");
    }

    // 2.2 发布新增加的地图部分 (绿色)
    PointCloudXYZRGB::Ptr new_cloud = map_updater_->convertKeyframesToPCL(
        new_keyframes_all_, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);
    for (auto& p : new_cloud->points) {
      p.r = 0;
      p.g = 255;
      p.b = 0;  // 绿色
    }
    sensor_msgs::msg::PointCloud2 msg_new;
    pcl::toROSMsg(*new_cloud, msg_new);
    msg_new.header.stamp = this->now();
    msg_new.header.frame_id = "camera_init";
    pub_newly_added_map_->publish(msg_new);
    LOG_INFO_F("Published newly added map visualization (GREEN).");

    // 2.3 重新加载并发布最终的完整地图 (白色/原色)
    // 为了保证绝对一致性，我们从磁盘重新读取 localization_output.txt
    new_keyframes_all_.clear();
    new_keyframes_.clear();

    std::string final_pose_file =
        map_data_path_ + "/result/localization_output.txt";
    std::string keyframes_dir = map_data_path_ + "/result/keyframes/";
    std::ifstream infile(final_pose_file);
    std::map<double, KeyframeData> final_kfs_map;
    std::string line;

    if (infile.is_open()) {
      while (std::getline(infile, line)) {
        std::stringstream ss(line);
        KeyframeData kf;
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        ss >> kf.timestamp >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >>
            q.z() >> q.w();
        kf.pose = Sophus::SE3d(q, t);

        std::stringstream pcd_filename_ss;
        pcd_filename_ss << std::fixed << std::setprecision(9) << kf.timestamp
                        << ".pcd";
        kf.pcd_path = keyframes_dir + pcd_filename_ss.str();
        if (std::filesystem::exists(kf.pcd_path))
          final_kfs_map[kf.timestamp] = kf;
      }
      infile.close();
    } else {
      LOG_ERROR_F("Failed to reload final pose file for visualization.");
    }

    // 分离 "保留的旧帧" 和 "新帧" 以便于分别可视化 (可选)
    std::vector<KeyframeData> final_kept_kfs;
    std::unordered_set<double> original_timestamps;
    for (const auto& okf : original_map_keyframes_)
      original_timestamps.insert(okf.timestamp);

    for (const auto& pair : final_kfs_map) {
      if (original_timestamps.count(pair.first))
        final_kept_kfs.push_back(pair.second);
    }

    // 发布保留的旧地图 (白色/Intensity)
    if (!final_kept_kfs.empty()) {
      PointCloudXYZRGB::Ptr kept_cloud = map_updater_->convertKeyframesToPCL(
          final_kept_kfs, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);
      sensor_msgs::msg::PointCloud2 kept_msg;
      pcl::toROSMsg(*kept_cloud, kept_msg);
      kept_msg.header.stamp = this->now();
      kept_msg.header.frame_id = "camera_init";
      pub_original_map_kept_->publish(kept_msg);
      LOG_INFO_F("Published preserved original map (WHITE).");
    }

    // 2.4 保存并发布合并后的最终可视化大 PCD
    std::vector<KeyframeData> all_final_kfs;
    for (const auto& pair : final_kfs_map) all_final_kfs.push_back(pair.second);

    if (!all_final_kfs.empty()) {
      PointCloudXYZRGB::Ptr final_map_cloud =
          map_updater_->convertKeyframesToPCL(all_final_kfs, Lidar_R_wrt_IMU,
                                              Lidar_T_wrt_IMU);

      std::string final_viz_path =
          map_data_path_ + "/map_final_visualization.pcd";
      pcl::io::savePCDFileBinary(final_viz_path, *final_map_cloud);
      LOG_INFO_F("Saved merged visualization map to %s",
                 final_viz_path.c_str());

      sensor_msgs::msg::PointCloud2 final_msg;
      pcl::toROSMsg(*final_map_cloud, final_msg);
      final_msg.header.stamp = this->now();
      final_msg.header.frame_id = "camera_init";
      pub_final_updated_map_->publish(final_msg);
    }
  }

  res->success = true;
  res->message = "Map update complete.";
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

  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) {
    path.poses.push_back(msg_body_pose);
    pubPath->publish(path);
  }
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