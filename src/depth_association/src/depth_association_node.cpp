#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include "geometry_msgs/TwistStamped.h"
#include <tf/transform_listener.h>
#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include "depth_association/parameters.h"
#include "depth_association/sync_data.h"
std::mutex deque_mutex;
std::deque<sensor_msgs::PointCloudConstPtr> feature_buf;
std::deque<sensor_msgs::PointCloud2ConstPtr> velodyne_buf;
std::deque<geometry_msgs::TwistStampedConstPtr> velocity_buf;
int sub_data_num = 0;
double last_time = 0.0;
double cur_time = 0.0;
bool get_T_velo_imu = false;
Eigen::Matrix4f T_velo_imu; //transform from imu to velo
bool get_T_cam_velo = false;
Eigen::Matrix4f T_cam_velo;  //transform from velo to cam
ros::Publisher pub_feature_with_depth;

void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr &velodyne_msg) {
    velodyne_buf.push_back(velodyne_msg);
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
    feature_buf.push_back(feature_msg);
}

void velocity_callback(const geometry_msgs::TwistStampedConstPtr& twist_msg_ptr) {
	velocity_buf.push_back(twist_msg_ptr);
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "depth_association");
    ros::NodeHandle n("~");
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	readParameters(n);

	ros::Subscriber sub_point_cloud = n.subscribe(POINT_CLOUD_TOPIC, 100, velodyne_callback);
	ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 100, feature_callback);
	ros::Subscriber sub_velocity = n.subscribe(VELOCITY_TOPIC, 100, velocity_callback);

	pub_feature_with_depth = n.advertise<sensor_msgs::PointCloud>("feature_with_depth", 1000);

	tf::TransformListener listener_;
	ros::Duration(0.1).sleep();  //for tf::TransformListener

	ros::Rate rate(100);
	while(ros::ok()) {
		ros::spinOnce();

		bool sync = sync_data(feature_buf, velodyne_buf, velocity_buf);
		if (sync) {
			//parse data
			last_time = cur_time;
			sub_data_num++;
			sensor_msgs::PointCloudConstPtr feature = feature_buf.front();
			double sync_time = feature->header.stamp.toSec();
			cur_time = sync_time;
			feature_buf.pop_front();
			sensor_msgs::PointCloud2ConstPtr velodyne = velodyne_buf.front();
			velodyne_buf.pop_front();
			geometry_msgs::TwistStampedConstPtr velocity_0 = velocity_buf.front();
			geometry_msgs::TwistStampedConstPtr velocity_1 = velocity_buf.at(1);

			//debug for sync data
			ROS_DEBUG("Sync %d data success, feature time is %f, velodyne time is %f",
										sub_data_num, feature->header.stamp.toSec(), velodyne->header.stamp.toSec());
			ROS_DEBUG("velocity_0 time is %f, velocity_1_time is %f", velocity_0->header.stamp.toSec(), velocity_1->header.stamp.toSec());
			if (abs(feature->header.stamp.toSec() - velodyne->header.stamp.toSec()) > 0.02 ||
					velocity_0->header.stamp.toSec() > feature->header.stamp.toSec() ||
					velocity_1->header.stamp.toSec() < feature->header.stamp.toSec())
				ROS_WARN("Data un sync!");
			if(sub_data_num != 1 && (cur_time - last_time) > 0.15)
				ROS_WARN("Data lost!");

			// process velocity
			Eigen::Matrix<double, 6, 1> sync_velo;
			sync_velocity(sync_time, velocity_0, velocity_1, sync_velo); //linear + angular
			std::string target_frame, source_frame;
			target_frame = "/camera_gray_left";
			source_frame = "/velo_link";
			if (!get_T_cam_velo) {
				if(LookupData(listener_, T_cam_velo, target_frame, source_frame))
					get_T_cam_velo = true;
				if (get_T_cam_velo)
					ROS_INFO("success to get T_cam_velo!, T_cam_velo is %f, %f, %f", T_cam_velo(0, 3), T_cam_velo(1, 3), T_cam_velo(2, 3));
			}
			// VelocityDataTransformCoordinate(sync_velo, T_cam_velo);

			//The Velodyne data from the KITTI datasets were processed to eliminate the effect of motion-distortion
			pcl::PointCloud<pcl::PointXYZ>::Ptr velodyne_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr velodyne_point_cloud_(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::fromROSMsg(*velodyne, *velodyne_point_cloud); 
			pcl::transformPointCloud(*velodyne_point_cloud, *velodyne_point_cloud_, T_cam_velo);
			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(*velodyne_point_cloud_, *velodyne_point_cloud_, indices);
			sensor_msgs::PointCloudPtr feature_wiht_depth = depthAssociation(feature, velodyne_point_cloud_);
			feature_wiht_depth->header = feature->header;
			pub_feature_with_depth.publish(feature_wiht_depth); //channels: 0 ids, 1 track_cnt, 2 pixel_u, 3 pixel_v, 4 depth
		}

		rate.sleep();
	}

	return 0;
}