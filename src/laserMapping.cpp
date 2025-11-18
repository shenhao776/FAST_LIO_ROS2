// src/laserMapping.cpp
#include "fast_lio/laserMapping.hpp"

#include <tf2/LinearMath/Quaternion.h>

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

  // [新增] 初始化组件
  initializeComponents();

  std::fill(epsi, epsi + 23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS,
                    epsi);

  // 初始化 VoxelMapConfig 用于 MapUpdater 和 VoxelMapManager
  VoxelMapConfig voxel_config;
  loadVoxelConfig(this, voxel_config);  // 使用外部的 helper 或成员函数加载

  // 初始化 VoxelMapManager (需要传递 voxel_map 引用)
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));

  // 初始化 MapUpdater
  map_updater_ = std::make_unique<MapUpdater>(voxel_config, Lidar_R_wrt_IMU,
                                              Lidar_T_wrt_IMU);

  initializeFiles();  // 初始化文件路径

  // [逻辑] 检查并加载现有地图
  std::string pose_file = map_data_path_ + "/result/localization_output.txt";
  if (std::filesystem::exists(pose_file)) {
    is_update_mode_ = true;
    RCLCPP_INFO(this->get_logger(),
                "\033[1;32mMap data found. Running in UPDATE mode.\033[0m");

    // 构建索引
    if (!map_updater_->buildVoxelMapIndex(map_data_path_, voxel_map_index,
                                          original_map_keyframes_)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to build map index.");
    }
    // 加载用于 Rviz 显示的地图
    loadExistingMap(map_data_path_);
  } else {
    is_update_mode_ = false;
    RCLCPP_INFO(this->get_logger(),
                "\033[1;32mNo map data found. Running in MAPPING mode.\033[0m");
  }

  // 文件操作
  std::string pos_log_dir = root_dir + "/Log/pos_log.txt";
  fp = fopen(pos_log_dir.c_str(), "w");
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
  fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), std::ios::out);
  if (pcd_save_interval > 0)
    fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json",
                      std::ios::out);

  initializeSubscribersAndPublishers();

  FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
  HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                 filter_size_surf_min);
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min,
                                filter_size_map_min);

  RCLCPP_INFO(this->get_logger(), "Node init finished.");
}

LaserMappingNode::~LaserMappingNode() {
  if (fout_out.is_open()) fout_out.close();
  if (fout_pre.is_open()) fout_pre.close();
  if (fout_dbg.is_open()) fout_dbg.close();
  if (fout_pcd_pos.is_open()) fout_pcd_pos.close();
  if (fp) fclose(fp);
  if (ptr_ == this) ptr_ = nullptr;
}

// [修复] 初始化组件函数
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
  if (is_update_mode_) {
    RCLCPP_INFO(this->get_logger(),
                "Update mode: Preserving existing map data in %s",
                map_data_path_.c_str());
    if (!std::filesystem::exists(map_data_path_)) {
      std::filesystem::create_directories(map_data_path_);
    }
  } else {
    RCLCPP_INFO(this->get_logger(),
                "[WARN] Mapping mode: Cleaning up old data in %s",
                map_data_path_.c_str());
    std::string rm_cmd = "rm -rf " + map_data_path_;
    // [修复] 忽略 system 返回值警告
    (void)system(rm_cmd.c_str());

    std::filesystem::create_directories(map_data_path_);

    if (pcd_save_en) {
      std::string pcd_dir = map_data_path_ + "/PCD/";
      std::filesystem::create_directories(pcd_dir);
    }

    std::string result_dir = map_data_path_ + "/result/";
    std::filesystem::create_directories(result_dir);
    std::string keyframes_dir = result_dir + "keyframes/";
    std::filesystem::create_directories(keyframes_dir);
  }
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
                  std::string(ROOT_DIR) + "Log");
  declare_and_get("pcd_save.pcd_save_distance_thresh",
                  pcd_save_distance_thresh_, 0.2);
  declare_and_get("pcd_save.filter_size_pcd", filter_size_pcd, 0.5);

  declare_and_get("max_iteration", NUM_MAX_ITERATIONS, 4);
  declare_and_get("runtime_pos_log_enable", runtime_pos_log, false);

  declare_and_get("online_update.periodic_align_interval",
                  periodic_align_interval_, 20);
  declare_and_get("online_update.periodic_align_min_dist",
                  periodic_align_min_dist_, 0.3);
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

  // 新增地图相关的发布者
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

  // VoxelMap 可视化
  voxel_map_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planes", 100);
  // 将 publisher 传递给 manager (因为 manager 没有 node handle)
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

  map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "map_save", std::bind(&LaserMappingNode::map_save_callback, this,
                            std::placeholders::_1, std::placeholders::_2));

  // 如果有初始地图，发布一次
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

    if (is_update_mode_) {
      if (!initial_align_finished_) {
        if (!initial_pose_received_) return;
        state_point.rot = initial_pose_.so3().unit_quaternion();
        state_point.pos = initial_pose_.translation();
        kf.change_x(state_point);
        initial_align_finished_ = true;
        RCLCPP_INFO(this->get_logger(), "Initial alignment finished.");
      }

      if (initial_align_finished_) {
        KeyframeData new_kf;
        new_kf.timestamp = Measures.lidar_beg_time;
        new_kf.pose =
            Sophus::SE3d(state_point.rot.toRotationMatrix(), state_point.pos);
        new_kf.cloud.reset(
            new PointCloudXYZI(*feats_undistort));  // 使用去畸变后的点云
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
    }

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

    if (path_en) publish_path(pubPath_);
    if (scan_pub_en) publish_frame_world(pubLaserCloudFull_);
    if (scan_pub_en && scan_body_pub_en)
      publish_frame_body(pubLaserCloudFull_body_);
    if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);

    if (voxelmap_manager &&
        voxelmap_manager->config_setting_.is_pub_plane_map_) {
      voxelmap_manager->pubVoxelMap(this->now());
    }

    if (pcd_save_en && pcd_save_interval == -1) {
      double dist = (state_point.pos - last_pcd_save_pos_).norm();
      if (!is_first_pcd_saved_ || dist > pcd_save_distance_thresh_) {
        last_pcd_save_pos_ = state_point.pos;
        is_first_pcd_saved_ = true;

        if (!is_update_mode_) {
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
                       << " " << state_point.pos.y() << " "
                       << state_point.pos.z() << " " << geoQuat.x << " "
                       << geoQuat.y << " " << geoQuat.z << " " << geoQuat.w
                       << std::endl;
        }
      }
    }

    if (runtime_pos_log) {
      if (time_log_counter >= MAXN) time_log_counter = 0;
      frame_num++;

      T1[time_log_counter] = Measures.lidar_beg_time;
      s_plot[time_log_counter] = t5 - t0;
      s_plot2[time_log_counter] = feats_undistort->points.size();
      s_plot3[time_log_counter] = kdtree_incremental_time;
      s_plot4[time_log_counter] = kdtree_search_time;
      s_plot5[time_log_counter] = kdtree_delete_counter;
      s_plot6[time_log_counter] = kdtree_delete_time;
      s_plot7[time_log_counter] = kdtree_size_st;
      s_plot8[time_log_counter] = kdtree_size_end;
      s_plot9[time_log_counter] = aver_time_consu;
      s_plot10[time_log_counter] = add_point_size;
      time_log_counter++;

      dump_lio_state_to_log(fp);
    }
  }
}

// [修复] 补充 dump_lio_state_to_log 函数实现
void LaserMappingNode::dump_lio_state_to_log(FILE* fp) {
  V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
  fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));  // Angle
  fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1),
          state_point.pos(2));                 // Pos
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // omega
  fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1),
          state_point.vel(2));                 // Vel
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // Acc
  fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1),
          state_point.bg(2));  // Bias_g
  fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1),
          state_point.ba(2));  // Bias_a
  fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1],
          state_point.grav[2]);  // Bias_a
  fprintf(fp, "\r\n");
  fflush(fp);
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
  icp.setMaxCorrespondenceDistance(1.0);
  icp.setMaximumIterations(30);

  PointCloudXYZI unused_result;
  icp.align(unused_result);

  if (icp.hasConverged() && icp.getFitnessScore() < 0.5) {
    Eigen::Matrix4f correction = icp.getFinalTransformation();
    Sophus::SE3d T_correction(correction.block<3, 3>(0, 0).cast<double>(),
                              correction.block<3, 1>(0, 3).cast<double>());

    Sophus::SE3d current_pose(state_point.rot.toRotationMatrix(),
                              state_point.pos);
    Sophus::SE3d corrected_pose = T_correction * current_pose;
    state_point.rot = corrected_pose.unit_quaternion();
    state_point.pos = corrected_pose.translation();

    kf.change_x(state_point);

    for (auto& kf : new_keyframes_) {
      kf.pose = T_correction * kf.pose;
    }

    ikdtree = KD_TREE<PointType>();

    RCLCPP_INFO(
        this->get_logger(),
        "Periodic alignment success. Correction applied and map reset.");
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
      RCLCPP_INFO(this->get_logger(), "Received initial pose guess.");
    }
  }
}

void LaserMappingNode::savePCD() {
  if (!pcd_save_en) return;

  if (is_update_mode_) {
    std::vector<KeyframeData> final_kfs = original_map_keyframes_;
    final_kfs.insert(final_kfs.end(), new_keyframes_all_.begin(),
                     new_keyframes_all_.end());

    auto final_cloud = map_updater_->convertKeyframesToPCL(
        final_kfs, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);

    std::string save_path = map_data_path_ + "/PCD/final_map.pcd";
    std::filesystem::create_directories(map_data_path_ + "/PCD/");
    pcl::io::savePCDFileBinary(save_path, *final_cloud);
    RCLCPP_INFO(this->get_logger(), "Saved merged map to %s",
                save_path.c_str());
  } else {
    std::string raw_points_dir = map_data_path_ + "/PCD/all_raw_points.pcd";
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save);
  }
}

void LaserMappingNode::map_save_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
  (void)req;
  RCLCPP_INFO(this->get_logger(), "Service /save_map called.");
  savePCD();
  res->success = true;
  res->message = "Map saved.";
}

void LaserMappingNode::map_publish_callback() {
  if (map_pub_en) publish_map(pubLaserCloudMap_);
}

// ------------------ 下面是原有的辅助函数实现，保持不变 ------------------

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