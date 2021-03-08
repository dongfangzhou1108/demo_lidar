#include <deque>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include "geometry_msgs/TwistStamped.h"
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tf/transform_listener.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include "tic_toc.h"
#include "depth_association/parameters.h"

bool sync_data(std::deque<sensor_msgs::PointCloudConstPtr>& feature_data,
								std::deque<sensor_msgs::PointCloud2ConstPtr>& velodyne_data,
								std::deque<geometry_msgs::TwistStampedConstPtr>& velocity_data);

void sync_velocity(double sync_time, 
										geometry_msgs::TwistStampedConstPtr& velocity_0,
										geometry_msgs::TwistStampedConstPtr& velocity_1,
										Eigen::Matrix<double, 6, 1>& velocity_sync);

bool TransformToMatrix(const tf::StampedTransform& transform, Eigen::Matrix4f& transform_matrix);

bool LookupData(tf::TransformListener& listener_, Eigen::Matrix4f& transform_matrix,
									std::string& target_frame, std::string& source_frame); //get sensor calib data

void VelocityDataTransformCoordinate(Eigen::Matrix<double, 6, 1>& sync_velo, Eigen::Matrix4f transform_matrix);

sensor_msgs::PointCloudPtr depthAssociation(sensor_msgs::PointCloudConstPtr& feature, pcl::PointCloud<pcl::PointXYZ>::Ptr& velodyne_point_cloud);

void project2D(Eigen::Vector3d& point, Eigen::Vector2d& pixel);

bool whetherToSelect(Eigen::Vector2d& pixel, Eigen::Vector3d& p_0, Eigen::Vector3d& p_1, Eigen::Vector3d& p_2);