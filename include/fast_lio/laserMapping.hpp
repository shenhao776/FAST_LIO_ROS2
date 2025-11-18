#ifndef LASER_MAPPING_HPP
#define LASER_MAPPING_HPP

// std & system
#include <math.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ROS 2
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <image_transport/image_transport.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Eigen & Math
#include <omp.h>
#include <so3_math.h>

#include <Eigen/Core>

// Local Libraries
#include <ikd-Tree/ikd_Tree.h>

#include "IMU_Processing.hpp"
#include "map_updater.hpp"
#include "preprocess.h"
#include "use-ikfom.hpp"
#include "voxel_map.hpp"

// 宏定义
#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

class LaserMappingNode : public rclcpp::Node {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit LaserMappingNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~LaserMappingNode();

  static LaserMappingNode* ptr_;
  static void OnSignal(int sig);

 private:
  // --- 初始化函数 (此前缺失声明) ---
  void readParameters();
  void initializeSubscribersAndPublishers();
  void initializeFiles();
  void initializeComponents();  // 如果cpp中有用到

  // --- 核心流程 ---
  void timer_callback();

  // --- 回调函数 ---
  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg);
  void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg);
  void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in);

  void initialPoseCallback(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  // [修复] 补充声明服务回调和发布回调
  void map_save_callback(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void map_publish_callback();

  // --- 算法辅助 ---
  bool sync_packages(MeasureGroup& meas);
  void lasermap_fov_segment();
  void map_incremental();
  void points_cache_collect();  // [修复] 补充声明

  // --- 坐标变换 ---
  template <typename T>
  void pointBodyToWorld(const Eigen::Matrix<T, 3, 1>& pi,
                        Eigen::Matrix<T, 3, 1>& po);
  void pointBodyToWorld(PointType const* const pi, PointType* const po);
  void RGBpointBodyToWorld(PointType const* const pi, PointType* const po);
  void RGBpointBodyLidarToIMU(PointType const* const pi, PointType* const po);

  // --- 发布与显示 ---
  void publish_frame_world(
      rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);
  void publish_frame_body(
      rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);
  void publish_effect_world(
      rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);
  void publish_map(
      rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);
  void publish_odometry(
      const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom,
      std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br);
  void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath);

  // --- 地图更新与对齐 ---
  bool loadExistingMap(const std::string& map_path);
  void periodicAlignment();
  void savePCD();

  // [修复] 确保此函数在 cpp 中被实现
  void dump_lio_state_to_log(FILE* fp);

  template <typename T>
  void set_posestamp(T& out) {
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
  }

  static void h_share_model(state_ikfom& s,
                            esekfom::dyn_share_datastruct<double>& ekfom_data);

 private:
  // ROS 通信
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserCloudFull_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserCloudFull_body_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pubLaserCloudEffect_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_original_map_kept_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_newly_added_map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_deleted_points_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pub_final_updated_map_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      voxel_map_pub_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr
      sub_pcl_livox_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      sub_initial_pose_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr map_pub_timer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

  std::shared_ptr<Preprocess> p_pre;
  std::shared_ptr<ImuProcess> p_imu;
  esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
  KD_TREE<PointType> ikdtree;

  std::unique_ptr<MapUpdater> map_updater_;
  std::shared_ptr<VoxelMapManager> voxelmap_manager;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*> voxel_map;

  state_ikfom state_point;
  MeasureGroup Measures;
  vect3 pos_lid;
  nav_msgs::msg::Path path;
  nav_msgs::msg::Odometry odomAftMapped;
  geometry_msgs::msg::Quaternion geoQuat;
  geometry_msgs::msg::PoseStamped msg_body_pose;

  PointCloudXYZI::Ptr featsFromMap;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr normvec;
  PointCloudXYZI::Ptr laserCloudOri;
  PointCloudXYZI::Ptr corr_normvect;
  PointCloudXYZI::Ptr _featsArray;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_save;

  PointCloudXYZRGB::Ptr original_map_visual_;
  PointCloudXYZRGB::Ptr deleted_points_visual_;

  std::deque<double> time_buffer;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
  std::mutex mtx_buffer;
  std::condition_variable sig_buffer;
  std::mutex mtx_initial_pose;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterMap;

  std::string root_dir = ROOT_DIR;
  std::string map_file_path, lid_topic, imu_topic;
  bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false,
       extrinsic_est_en = true, path_en = true;
  bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
  bool effect_pub_en = false, map_pub_en = false;
  double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
  double filter_size_corner_min = 0, filter_size_surf_min = 0,
         filter_size_map_min = 0, fov_deg = 0;
  double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0,
         lidar_end_time = 0, first_lidar_time = 0.0;
  double time_diff_lidar_to_imu = 0.0;
  float DET_RANGE = 300.0f;
  const float MOV_THRESHOLD = 1.5f;
  int pcd_save_interval = -1, pcd_index = 0;
  int NUM_MAX_ITERATIONS = 0;

  bool is_update_mode_ = false;
  std::string map_data_path_;
  bool initial_pose_received_ = false;
  bool initial_align_finished_ = false;
  Sophus::SE3d initial_pose_;
  Sophus::SE3d last_kf_pose_;

  std::vector<KeyframeData> original_map_keyframes_;
  std::vector<KeyframeData> new_keyframes_all_;
  std::vector<KeyframeData> new_keyframes_;
  MapUpdater::VoxelMapIndexType voxel_map_index;

  int motion_keyframe_count_ = 0;
  int periodic_align_interval_ = 20;
  double periodic_align_min_dist_ = 0.3;
  double keyframe_search_radius_ = 2.0;
  double point_replacement_radius_ = 0.1;
  double pcd_save_distance_thresh_ = 0.2;
  V3D last_pcd_save_pos_ = V3D::Zero();
  bool is_first_pcd_saved_ = false;
  bool colmap_output_en = false;
  double filter_size_pcd = 0.2;

  double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0,
         kdtree_delete_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;
  double res_mean_last = 0.05, total_residual = 0.0;
  double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
  double epsi[23] = {0.001};

  float res_last[100000] = {0.0};
  bool point_selected_surf[100000] = {0};
  double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN],
      s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN],
      s_plot10[MAXN], s_plot11[MAXN];

  int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0,
      kdtree_delete_counter = 0;
  int effct_feat_num = 0, time_log_counter = 0, scan_count = 0,
      publish_count = 0;
  int feats_down_size = 0;
  int frame_num = 0;

  bool lidar_pushed = false, flg_first_scan = true, flg_exit = false,
       flg_EKF_inited = false;
  bool is_first_lidar = true;
  double lidar_mean_scantime = 0.0;
  int scan_num = 0;

  std::vector<std::vector<int>> pointSearchInd_surf;
  std::vector<BoxPointType> cub_needrm;
  std::vector<PointVector> Nearest_Points;
  std::vector<double> extrinT;
  std::vector<double> extrinR;

  V3F XAxisPoint_body;
  V3F XAxisPoint_world;
  V3D euler_cur;
  V3D Lidar_T_wrt_IMU;
  M3D Lidar_R_wrt_IMU;
  BoxPointType LocalMap_Points;
  bool Localmap_Initialized = false;

  std::ofstream fout_pre, fout_out, fout_dbg, fout_pcd_pos;
  FILE* fp = nullptr;

  double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0,
         aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
};

template <typename T>
void LaserMappingNode::pointBodyToWorld(const Eigen::Matrix<T, 3, 1>& pi,
                                        Eigen::Matrix<T, 3, 1>& po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body +
                                  state_point.offset_T_L_I) +
               state_point.pos);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

#endif  // LASER_MAPPING_HPP