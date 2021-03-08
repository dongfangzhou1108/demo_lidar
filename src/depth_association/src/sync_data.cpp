#include "depth_association/sync_data.h"

bool sync_data(std::deque<sensor_msgs::PointCloudConstPtr>& feature_data,
								std::deque<sensor_msgs::PointCloud2ConstPtr>& velodyne_data,
								std::deque<geometry_msgs::TwistStampedConstPtr>& velocity_data) {
	if (feature_data.empty())
		return false;

	double sync_time = feature_data.front()->header.stamp.toSec();
	while(feature_data.size() >= 1 && velodyne_data.size() >= 1 && velocity_data.size() >=2) {
		if (velocity_data.front()->header.stamp.toSec() > sync_time) {
			feature_data.pop_front();
			return false;
		}
		if (velocity_data.at(1)->header.stamp.toSec() < sync_time) {
            velocity_data.pop_front();
            continue;
        }
        if (sync_time - velocity_data.front()->header.stamp.toSec()  > 0.2) {
            velocity_data.pop_front();
            return false;
        }
        if (velocity_data.front()->header.stamp.toSec() - sync_time  > 0.2) {
			feature_data.pop_front();
            return false;
        }
		break;
	}

	if (velocity_data.size() < 2)
		return false;

	while(velodyne_data.size() >= 1) {
		if (abs(velodyne_data.front()->header.stamp.toSec() - sync_time) < 0.01)
			return true;
		else {
			if (sync_time > velodyne_data.front()->header.stamp.toSec()) {
				velodyne_data.pop_front();
				continue;
			}
			else {
				feature_data.pop_front();
				break;
			}
		}
		break;
	}

	if (velodyne_data.size() < 1)
		return false;

	return true;
}

void sync_velocity(double sync_time, 
										geometry_msgs::TwistStampedConstPtr& velocity_0,
										geometry_msgs::TwistStampedConstPtr& velocity_1,
										Eigen::Matrix<double, 6, 1>& velocity_sync) {
	double front_time = velocity_0->header.stamp.toSec();
	double back_time = velocity_1->header.stamp.toSec();
	double front_scale = (back_time - sync_time) / (back_time - front_time);
	double back_scale = (sync_time - front_time) / (back_time - front_time);
	velocity_sync << front_scale * velocity_0->twist.linear.x + back_scale * velocity_1->twist.linear.x,
										front_scale * velocity_0->twist.linear.y + back_scale * velocity_1->twist.linear.y,
										front_scale * velocity_0->twist.linear.z + back_scale * velocity_1->twist.linear.z,
										front_scale * velocity_0->twist.angular.x + back_scale * velocity_1->twist.angular.x,
										front_scale * velocity_0->twist.angular.y + back_scale * velocity_1->twist.angular.y,
										front_scale * velocity_0->twist.angular.z + back_scale * velocity_1->twist.angular.z;
}

bool TransformToMatrix(const tf::StampedTransform& transform, Eigen::Matrix4f& transform_matrix) {
    Eigen::Translation3f tl_btol(transform.getOrigin().getX(), transform.getOrigin().getY(), transform.getOrigin().getZ());
    
    double roll, pitch, yaw;
    tf::Matrix3x3(transform.getRotation()).getEulerYPR(yaw, pitch, roll);
    Eigen::AngleAxisf rot_x_btol(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf rot_y_btol(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rot_z_btol(yaw, Eigen::Vector3f::UnitZ());

    // 此矩阵为 child_frame_id 到 base_frame_id 的转换矩阵
    transform_matrix = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();

    return true;
}

bool LookupData(tf::TransformListener& listener_, Eigen::Matrix4f& transform_matrix,
									std::string& target_frame, std::string& source_frame){
    try {
        tf::StampedTransform transform;
        listener_.lookupTransform(target_frame, source_frame, ros::Time(0), transform); //get trans from source to target
        TransformToMatrix(transform, transform_matrix);
        return true;
    } catch (tf::TransformException &ex) {
        return false;
    }
}

void VelocityDataTransformCoordinate(Eigen::Matrix<double, 6, 1>& sync_velo, Eigen::Matrix4f transform_matrix) {
	Eigen::Matrix4d matrix = transform_matrix.cast<double>();
	Eigen::Matrix3d t_R = matrix.block<3,3>(0,0);
	Eigen::Vector3d w(sync_velo(3), sync_velo(4), sync_velo(5)); //angular
	Eigen::Vector3d v(sync_velo(0), sync_velo(1), sync_velo(2)); //velocity

	w = t_R * w;
    v = t_R * v;

    Eigen::Vector3d r(matrix(0,3), matrix(1,3), matrix(2,3));
    Eigen::Vector3d delta_v; //delta_v = 角速度 叉乘 杠臂
	delta_v = w.cross(r);
    v = v + delta_v;
	sync_velo << v(0), v(1), v(2), w(0), w(1), w(2);
}

sensor_msgs::PointCloudPtr depthAssociation(sensor_msgs::PointCloudConstPtr& feature, pcl::PointCloud<pcl::PointXYZ>::Ptr& velodyne_point_cloud) {
	TicToc t_s;
	bool plot = false;

	pcl::PointCloud<pcl::PointXYZI>::Ptr velodyne_points_in_cam(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointXYZI point;
	//只选取前景区域
	for (unsigned int i = 0; i < velodyne_point_cloud->points.size(); i++) {
		if (velodyne_point_cloud->points[i].z >  0) {
			point.intensity = 10.0 / velodyne_point_cloud->points[i].z;
			point.x = velodyne_point_cloud->points[i].x * point.intensity;
			point.y = velodyne_point_cloud->points[i].y * point.intensity;
			point.z = 10.0;
			velodyne_points_in_cam->points.push_back(point);
		}
	}

	pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdTree(new pcl::KdTreeFLANN<pcl::PointXYZI>());
	kdTree->setInputCloud(velodyne_points_in_cam);

	pcl::PointCloud<pcl::PointXYZI>::Ptr feature_point_cloud(new pcl::PointCloud<pcl::PointXYZI>);
	int sum = 0;
	double depth = 0;
	sensor_msgs::ChannelFloat32 depth_channel;
	cv::Mat depth_img(ROW, COL, CV_8UC3, cv::Scalar(255, 255, 255));
	Eigen::Vector3d pts3 = Eigen::Vector3d::Zero();
	Eigen::Vector2d pts2 = Eigen::Vector2d::Zero();
	cv::Point2f pixel;
	if (plot) {
		for (auto& pts : velodyne_points_in_cam->points) {
			pts3 << pts.x, pts.y, pts.z;
			project2D(pts3, pts2); pixel.x = pts2.x(); pixel.y = pts2.y();
			cv::circle(depth_img, pixel, 1, cv::Scalar(255, 0, 0), 1);
	}
	}
	for (unsigned int i = 0; i < feature->points.size(); i++) {
		bool get_depth = false;
		int feature_id = round(feature->channels[0].values[i]);
		point.x = 10.0 * feature->points[i].x; 
		point.y = 10.0 * feature->points[i].y; 
		point.z = 10.0;
		point.intensity = feature_id;
		if (plot) {
			pts3 << point.x, point.y, point.z;
			project2D(pts3, pts2); pixel.x = pts2.x(); pixel.y = pts2.y();
			cv::circle(depth_img, pixel, 3, cv::Scalar(0, 0, 255), 2);
		}
		std::vector<int> pointSearchInd;
		std::vector<float> pointSearchSqrDis;
		kdTree->nearestKSearch(point, 3, pointSearchInd, pointSearchSqrDis); //pointSearchSqrDis from small to big
		if (pointSearchSqrDis.size() == 3 ) {
			pcl::PointXYZI depthPoint = velodyne_points_in_cam->points[pointSearchInd[0]];
			Eigen::Vector3d p_0;
			p_0 <<  depthPoint.x / depthPoint.intensity, depthPoint.y / depthPoint.intensity, 10.0 / depthPoint.intensity;
			depthPoint = velodyne_points_in_cam->points[pointSearchInd[1]];
			Eigen::Vector3d p_1; 
			p_1 <<  depthPoint.x / depthPoint.intensity, depthPoint.y / depthPoint.intensity, 10.0 / depthPoint.intensity;
			depthPoint = velodyne_points_in_cam->points[pointSearchInd[2]];
			Eigen::Vector3d p_2; 
			p_2 <<  depthPoint.x / depthPoint.intensity, depthPoint.y / depthPoint.intensity, 10.0 / depthPoint.intensity;
			Eigen::Vector2d pixel_feature;
			pixel_feature << feature->channels[2].values[i], feature->channels[3].values[i];
			bool whether_select = whetherToSelect(pixel_feature, p_0, p_1, p_2);
			if (whether_select) {
				Eigen::Vector3d p_; 
				p_ << feature->points[i].x, feature->points[i].y, 1.0;
				Eigen::Vector3d temp_0 = (p_0 - p_1).cross(p_0 - p_2);
				double delta_0 = p_0.dot(temp_0); 
				double delta_1 = p_.dot(temp_0);
				depth = delta_0 / delta_1; 
				if (abs( 3 * depth - p_0.z() - p_1.z() - p_2.z()) < 0.3) {
					ROS_DEBUG("%f", abs( 3 * depth - p_0.z() - p_1.z() - p_2.z()));
					sum ++;
					get_depth = true;
					if (plot) {
						project2D(p_0, pts2); pixel.x = pts2.x(); pixel.y = pts2.y();
						if(pixel.x < COL && pixel.y < ROW && pixel.x > 0 && pixel.y > 0)
							cv::circle(depth_img, pixel, 2, cv::Scalar(0, 255, 0), 2);
						project2D(p_1, pts2); pixel.x = pts2.x(); pixel.y = pts2.y();
						if(pixel.x < COL && pixel.y < ROW && pixel.x > 0 && pixel.y > 0)
							cv::circle(depth_img, pixel, 2, cv::Scalar(0, 255, 0), 2);
						project2D(p_2, pts2); pixel.x = pts2.x(); pixel.y = pts2.y();
						if(pixel.x < COL && pixel.y < ROW && pixel.x > 0 && pixel.y > 0)
							cv::circle(depth_img, pixel, 2, cv::Scalar(0, 255, 0), 2);
				}
				}
			}
		}
		if (get_depth)
			depth_channel.values.push_back(depth);
		else
			depth_channel.values.push_back(-1.0);
	}
	double whole_t = t_s.toc();
	ROS_DEBUG("process %f ms, sum of feature: %d, association with depth: %d", whole_t, int(feature->points.size()), sum);
	if (plot) {
		std::ostringstream stream;
		stream << velodyne_points_in_cam->points.size();
		std::string s = stream.str(); 
		std::string s_1 = "/home/dfz/Documents/ROS/demo_velodyne/src/doc/depth_map/depth_img";
		std::string s_2 = ".jpg";
		cv::imwrite( s_1 + s + s_2, depth_img);
	}

	sensor_msgs::PointCloudPtr feature_with_depth(new sensor_msgs::PointCloud);
	feature_with_depth->points = feature->points;
	feature_with_depth->channels.push_back(feature->channels[0]);
	feature_with_depth->channels.push_back(feature->channels[1]);
	feature_with_depth->channels.push_back(feature->channels[2]);
	feature_with_depth->channels.push_back(feature->channels[3]);
	feature_with_depth->channels.push_back(depth_channel);
	return feature_with_depth;
}

void project2D(Eigen::Vector3d& point, Eigen::Vector2d& pixel) {
	Eigen::Matrix3d K;
	K << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1;
	Eigen::Vector3d tmp = K * point  *( 1/ point(2)) ;
	pixel << tmp(0), tmp(1);
}

bool whetherToSelect(Eigen::Vector2d& pixel, Eigen::Vector3d& p_0, Eigen::Vector3d& p_1, Eigen::Vector3d& p_2) {
	Eigen::Vector3d aver = double(1.0/ 3.0) * (p_0 + p_1 + p_2);
	double cov = (aver - p_0).norm() + (aver - p_1).norm() + (aver - p_1).norm();
	if (cov > cov_pts)
		return false;
	Eigen::Vector2d pixel_p0 = Eigen::Vector2d::Zero();
	project2D(p_0, pixel_p0);
	Eigen::Vector2d pixel_p1 = Eigen::Vector2d::Zero();
	project2D(p_1, pixel_p1);
	Eigen::Vector2d pixel_p2 = Eigen::Vector2d::Zero();
	project2D(p_2, pixel_p2);

	Eigen::Vector2d aver_pixel = double(1.0/ 3.0) * (pixel_p0 + pixel_p1 + pixel_p2);
	cov = (pixel - aver_pixel).norm();
	if (cov > cov_pixel)
		return false;

	bool x_big = (pixel.x() > pixel_p0.x()) && (pixel.x() > pixel_p1.x()) && (pixel.x() > pixel_p2.x());
	bool x_small = (pixel.x() < pixel_p0.x()) && (pixel.x() < pixel_p1.x()) && (pixel.x() < pixel_p2.x());
	bool y_big = (pixel.y() > pixel_p0.y()) && (pixel.y() > pixel_p1.y()) && (pixel.y() > pixel_p2.y());
	bool y_small = (pixel.y() < pixel_p0.y()) && (pixel.y() < pixel_p1.y()) && (pixel.y() < pixel_p2.y());
	ROS_DEBUG("%d, %d, %d, %d", int(x_big), int(x_small), int(x_big), int(y_small));
	ROS_DEBUG("pixel: %f, %f, %f, %f", pixel.x(), pixel_p0.x(), pixel_p1.x(), pixel_p2.x());
	ROS_DEBUG("			  %f, %f, %f, %f", pixel.y(), pixel_p0.y(), pixel_p1.y(), pixel_p2.y());

	return true;
}