#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <vector>
#include <list>
#include <map>
using namespace std;
#include <ros/ros.h>
#include "parameters.h"

class FeaturePerFrame {
	public:
    Vector3d point; //un_pts in normalized plane
    Vector2d uv; //pixel
	double depth;

	FeaturePerFrame(const Eigen::Matrix<double, 6, 1>& _point) {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        depth = _point(5);
    }
};

class FeaturePerId {
	public:
	const int feature_id;
    int start_frame; //the first frame which this feature occured
	int used_num;
	double estimated_depth;
    vector<FeaturePerFrame> feature_per_frame;
	int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
	double position[3];

	FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame), used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

	int endFrame();
};

class FeatureManager {
	public:
	list<FeaturePerId> feature;
	int last_track_num; //about 100
	int last_track_with_depth_num; //about 50

	FeatureManager() {
		feature.clear();
		last_track_num = 0;
		last_track_with_depth_num = 0;
	}

	// add feature data with id, track_cnt, pts in nomalized plane, pixel, depth in FeatureManager
	void addFeature(int frame_count, const std::map<int, vector<pair<int, Eigen::Matrix<double, 6, 1>>>> &image);
	// get corrsponding un pts in frame_count_l and frame_count_r
	vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
	// remove the feature which have be seen in the latest frame, and start_frame --
	void removeBack();
	//after get pose, triangulate point and compute depth for FeaturePerId.estimated_depth
	void triangulate(Matrix3d Rs[], Vector3d Ts[]);
	//get feature num which have seen over twice
	int getFeatureCount();
	//get inverse depth which have be seen for twice
	VectorXd getDepthVector();
	//set FeaturePerId estimated_depth
	void setDepth(const VectorXd &x);
	// when slide window, change depth of feature
	void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
};

#endif