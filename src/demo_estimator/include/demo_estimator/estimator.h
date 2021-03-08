#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include "demo_estimator/feature_manager.h"
#include "demo_estimator/parameters.h"
#include "demo_estimator/initial_sfm.h"
#include <map>
#include <std_msgs/Header.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include "demo_estimator/factor/pose_local_parameterization.h"
#include "demo_estimator/factor/projection_factor.h"
#include "demo_estimator/tic_toc.h"
#include "demo_estimator/initial_sfm.h"
#include "demo_estimator/factor/depth_factor.h"

using namespace std;

class Estimator {
	public:

	enum SolverFlag {
		INITIAL,
		NON_LINEAR,
		SFM
	};

	enum SIZE_PARAMETERIZATION
	{
		SIZE_POSE = 7,
		SIZE_FEATURE = 1
	};

	int frame_count;
	FeatureManager f_manager; 
	SolverFlag solver_flag;
	Matrix3d Rs[(WINDOW_SIZE + 1)];
	Vector3d Ts[(WINDOW_SIZE + 1)];
	double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
	double para_R[WINDOW_SIZE + 1][4];
	double para_T[WINDOW_SIZE + 1][3];
	double para_Feature[NUM_OF_F][SIZE_FEATURE];

	Matrix3d back_R0; 
    Vector3d back_T0; 

	Estimator();
	void setParameter();

	void processImage(const std::map<int, vector<pair<int, Eigen::Matrix<double, 6, 1>>>> &image, const std_msgs::Header &header);
	bool initialStructure();
	// get frame with enough parallax, compute pose between l and last frame for initialization
	bool get_frame_to_init(Matrix3d &relative_R, Vector3d &relative_T, int &l);
	// using corres feature to compute R and T, pose: T_lthFrame_oneFrame
	bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);
	//transform feature data to sfm data
	vector<SFMFeature> feature2SFMFeature(list<FeaturePerId>& feature);
	//solve_flag == INITIAL: f_manager.removeBack()
	void slideWindow();
	//nolinear optimazition
	void optimization(); //if do not fix more than one para, the scale will make optimization failure
	void optimization_sfm(); //time is over and it also have scale problem, if fix the pose, the latest pose will be not good
	void optimization_depth(); // optimization by lidar depth
	// trans data into array for ceres
	void vector2double();
	//trans data from array to data
	void double2vector();
	//get initial pose for new frame by triangulate point
	void getLatestPoseByPnP();
	//get initial pose for new frame by depth point
	void getLatestPoseByDepthPnP();
};

#endif