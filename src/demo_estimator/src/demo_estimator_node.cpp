#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <deque>
#include <map>
#include <vector>
#include "demo_estimator/parameters.h"
#include "demo_estimator/feature_manager.h"
#include "demo_estimator/estimator.h"
#include <nav_msgs/Path.h>

using namespace std;

std::mutex data_buf; 
bool get_data;
std::deque<sensor_msgs::PointCloudConstPtr> feature_buf;
Estimator estimator;
ros::Publisher pub_camera_pose;

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
    feature_buf.push_back(feature_msg);
}

nav_msgs::Path getCameraPose(nav_msgs::Path& odometry, const Estimator &estimator, const std_msgs::Header &header) {
	Eigen::Quaterniond back_Q0(estimator.back_R0);
	odometry.header = header;
	geometry_msgs::PoseStamped this_pose_stamped;
    this_pose_stamped.header = header;
    this_pose_stamped.header.frame_id = "map";
    this_pose_stamped.pose.position.x = estimator.back_T0.x();
    this_pose_stamped.pose.position.y = estimator.back_T0.y();
    this_pose_stamped.pose.position.z = estimator.back_T0.z();
    this_pose_stamped.pose.orientation.x = back_Q0.x();
    this_pose_stamped.pose.orientation.y = back_Q0.y();
    this_pose_stamped.pose.orientation.z = back_Q0.z();
    this_pose_stamped.pose.orientation.w = back_Q0.w();
	odometry.poses.push_back(this_pose_stamped);
	return odometry;
}

int main (int argc, char** argv) {
	ros::init(argc, argv, "demo_estimator");
    ros::NodeHandle n("~");
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	readParameters(n);

	ros::Subscriber sub_feature_with_depth = n.subscribe("/depth_association/feature_with_depth", 100, feature_callback);
	estimator.setParameter();
	pub_camera_pose = n.advertise<nav_msgs::Path>("camera_pose", 1000);
	nav_msgs::Path odometry;

	ros::Rate rate(100);
	while(ros::ok()) {
		ros::spinOnce();

		get_data = false;
		sensor_msgs::PointCloudConstPtr feature(new sensor_msgs::PointCloud);
		if (!feature_buf.empty()) {
			feature = feature_buf.front();
			feature_buf.pop_front();
			get_data = true;
		}

		if (get_data) {
			// transform data into a map with key is feature_id, value is pair of track_cnt and xyz_uv_depth
			std::map<int, vector<pair<int, Eigen::Matrix<double, 6, 1>>>> image;
			for (unsigned int i = 0; i < feature->points.size(); i++) {
				int feature_id = round(feature->channels[0].values[i]);
				int track_cnt = round(feature->channels[1].values[i]);
				double x = feature->points[i].x;
				double y = feature->points[i].y;
				double z = feature->points[i].z;
				double p_u = feature->channels[2].values[i];
				double p_v = feature->channels[3].values[i];
				double depth = feature->channels[4].values[i];
				ROS_ASSERT(z == 1);
				Eigen::Matrix<double, 6, 1> xyz_uv_depth;
				xyz_uv_depth << x, y, z, p_u, p_v, depth;
				image[feature_id].emplace_back(track_cnt,  xyz_uv_depth);
			}
			estimator.processImage(image, feature->header);
			if (estimator.frame_count == WINDOW_SIZE) {
				nav_msgs::Path camPose = getCameraPose(odometry, estimator, feature->header);
				pub_camera_pose.publish(camPose);
			}
		}

		rate.sleep();
	}

	return 0;
}