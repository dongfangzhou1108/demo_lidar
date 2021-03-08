#pragma once 
#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <map>
#include <vector>
using namespace std;
#include <ros/ros.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "demo_estimator/parameters.h"

struct SFMFeature
{
    bool state; //all begin with false, whether have been triangulated
    int id; //feature_id
    vector<pair<int, Vector2d>> observation; //pair of frame count and feature un_pts
	vector<pair<int, double>> velo_depth;
    double position[3]; //trangulate 3d point
    double depth;
};

class GlobalSFM
{
public:
	GlobalSFM() {}
	// return pose_lth_?th (trans point in ?th frame to l th frame)
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
		const Matrix3d relative_R, const Vector3d relative_T,
		vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);
	// triangulate points which have not been process in two frame which get pose
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
		int frame1, Eigen::Matrix<double, 3, 4> &Pose1, vector<SFMFeature> &sfm_f);
	// triangulate points in lth(world) frame
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
		Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	// using feature have been triangulated do PnP, return pose_2d_3d(pose_one_frame_world_frame)
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);
	// init odometry scale
	double initScale(int frame_l, int frame_r, vector<SFMFeature> &sfm_f);
	// depth PnP
	bool depthFeaturePnP(int frame_l, int frame_r, vector<SFMFeature> &sfm_f, Matrix3d &R_initial, Vector3d &P_initial); 

	int feature_num;
};

struct ReprojectionError3D 
{
	ReprojectionError3D(double observed_u, double observed_v) 
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];
		residuals[0] = xp - T(observed_u);
		residuals[1] = yp - T(observed_v);
		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

struct DepthError3D
{
	DepthError3D(double depth)
		:depth(depth) 
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, const T* scale, T* residuals) const
	{
		T p[3];
		T point_[3]; 
		point_[0] = scale[0] * point[0];
		point_[1] = scale[0] * point[1];
		point_[2] = scale[0] * point[2];
		ceres::QuaternionRotatePoint(camera_R, point_, p);
		p[0] += scale[0] * camera_T[0]; 
		p[1] += scale[0] * camera_T[1]; 
		p[2] += scale[0] * camera_T[2];
		residuals[0] = p[2] - T(depth);
		return true;
	}

	static ceres::CostFunction* Create(const double depth) {
		return (new ceres::AutoDiffCostFunction<DepthError3D, 1, 4, 3, 3, 1>(new DepthError3D(depth)));//residual, q, t, p, scale
	}

	double depth;
};