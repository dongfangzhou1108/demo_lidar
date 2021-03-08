#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <cv_bridge/cv_bridge.h>
#include "feature_tracker/parameters.h"
#include "feature_tracker/feature_tracker.h"

bool first_image_flag = true;
FeatureTracker trackerData;
ros::Publisher pub_img, pub_match;

void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) {
	double mx_d, my_d;
	mx_d = 1.0 / m_fx * p(0) - m_cx / m_fx;
    my_d = 1.0 / m_fy * p(1) - m_cy / m_fy;
	P << mx_d, my_d, 1.0;
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg) {
	cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else{
		ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
	}

	cv::Mat show_img = ptr->image;
	trackerData.readImage(ptr->image.rowRange(0, ROW), img_msg->header.stamp.toSec());
	
	if (first_image_flag == true) {
		first_image_flag = false;
		return;
	}
	else {
		sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
		feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";
		sensor_msgs::ChannelFloat32 pixel_u;
		sensor_msgs::ChannelFloat32 pixel_v;
        sensor_msgs::ChannelFloat32 id_of_point;
		sensor_msgs::ChannelFloat32 track_cnt_of_point;

		ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
		cv::Mat stereo_img = ptr->image;
		cv::Mat tmp_img = stereo_img.rowRange(0, ROW);
        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

		for (unsigned int i = 0; i < trackerData.ids.size(); i++) {
			Eigen::Vector2d a(trackerData.cur_pts[i].x, trackerData.cur_pts[i].y);
			Eigen::Vector3d b;
			liftProjective(a, b);
            geometry_msgs::Point32 p;
            p.x = b(0) / b(2); p.y = b(1) / b(2); p.z = 1;
			feature_points->points.push_back(p);
			id_of_point.values.push_back(trackerData.ids[i]);
			track_cnt_of_point.values.push_back(trackerData.track_cnt[i]);
			pixel_u.values.push_back(a.x());
			pixel_v.values.push_back(a.y());

			double len = std::min(1.0, 1.0 * trackerData.track_cnt[i] / WINDOW_SIZE);
            cv::circle(tmp_img, trackerData.cur_pts[i], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
		}
		feature_points->channels.push_back(id_of_point);
		feature_points->channels.push_back(track_cnt_of_point);
		feature_points->channels.push_back(pixel_u);
		feature_points->channels.push_back(pixel_v);
		pub_img.publish(feature_points);
		pub_match.publish(ptr->toImageMsg());
	}
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	readParameters(n);

	ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

	pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
	pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);

	ros::spin();
	return 0;
}