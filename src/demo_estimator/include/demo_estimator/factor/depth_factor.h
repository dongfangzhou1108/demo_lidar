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

struct veloTriError {
	veloTriError(double velo_depth): velo_depth(velo_depth)
	{}

	template <typename T>
	bool operator()(const T* inv_tri_depth, T* residual) const {
		residual[0] = T(velo_depth - 1.0 / inv_tri_depth[0]);
		return true;
	}

	static ceres::CostFunction* Create(const double velo_depth) {
		return (new ceres::AutoDiffCostFunction<veloTriError, 1, 1>(
			new veloTriError(velo_depth))
			);
	}

	double velo_depth;
};

struct veloDepthError3D {
	veloDepthError3D(double pts_l_u, double pts_l_v, double pts_r_u, double pts_r_v, double velo_depth) 
		:pts_l_u(pts_l_u), pts_l_v(pts_l_v), pts_r_u(pts_r_u), pts_r_v(pts_r_v), velo_depth(velo_depth)
	{}

	template <typename T>
	bool operator()(const T* const camera_l_R, 
									const T* const camera_l_T, 
									const T* const camera_r_R, 
									const T* const camera_r_T,
									const T* inv_tri_depth,
									T* residuals) const {
		T p_l[3];
		p_l[0] = T(pts_l_u / inv_tri_depth[0]); 
		p_l[1] = T(pts_l_v / inv_tri_depth[0]); 
		p_l[2] = T(1.0 / inv_tri_depth[0]);

		T p_w[3];
		ceres::QuaternionRotatePoint(camera_l_R, p_l, p_w);
		p_w[0] += camera_l_T[0]; 
		p_w[1] += camera_l_T[1]; 
		p_w[2] += camera_l_T[2];

		T camera_r_R_inv[4]; 
		camera_r_R_inv[0] = -camera_r_R[0];
		camera_r_R_inv[1] = -camera_r_R[1];
		camera_r_R_inv[2] = -camera_r_R[2];
		camera_r_R_inv[3] = camera_r_R[3];

		T camera_r_T_inv[3]; 
		ceres::QuaternionRotatePoint(camera_r_R_inv, camera_r_T, camera_r_T_inv);
		camera_r_T_inv[0] = T(-1.0 * camera_r_T_inv[0]);
		camera_r_T_inv[1] = T(-1.0 * camera_r_T_inv[1]);
		camera_r_T_inv[2] = T(-1.0 * camera_r_T_inv[2]);

		T p_r[3];
		ceres::QuaternionRotatePoint(camera_r_R_inv, p_w, p_r);
		p_r[0] += camera_r_T_inv[0]; 
		p_r[1] += camera_r_T_inv[1]; 
		p_r[2] += camera_r_T_inv[2];

		T xp = p_r[0] / p_r[2];
		T yp = p_r[1] / p_r[2];
		residuals[0] = xp - T(pts_r_u);
		residuals[1] = yp - T(pts_r_v);
		residuals[2] = p_r[2] - T(velo_depth);
		return true;
	}

	static ceres::CostFunction* Create(const double pts_l_u, 
																		const double pts_l_v, 
																		const double pts_r_u, 
																		const double pts_r_v, 
																		const double velo_depth) {
		return (new ceres::AutoDiffCostFunction<veloDepthError3D, 3, 4, 3, 4, 3, 1>(
			new veloDepthError3D(pts_l_u, pts_l_v, pts_r_u, pts_r_v, velo_depth))
			);
	}

	double pts_l_u;
	double pts_l_v;
	double pts_r_u;
	double pts_r_v;
	double velo_depth;
};

struct projError {
	projError(double pts_l_u, double pts_l_v, double pts_r_u, double pts_r_v) 
		:pts_l_u(pts_l_u), pts_l_v(pts_l_v), pts_r_u(pts_r_u), pts_r_v(pts_r_v)
	{}

	template <typename T>
	bool operator()(const T* const camera_l_R, 
									const T* const camera_l_T, 
									const T* const camera_r_R, 
									const T* const camera_r_T,
									const T* inv_tri_depth,
									T* residuals) const {
		T p_l[3];
		p_l[0] = T(pts_l_u / inv_tri_depth[0]); 
		p_l[1] = T(pts_l_v / inv_tri_depth[0]); 
		p_l[2] = T(1.0 / inv_tri_depth[0]);

		T p_w[3];
		ceres::QuaternionRotatePoint(camera_l_R, p_l, p_w);
		p_w[0] += camera_l_T[0]; 
		p_w[1] += camera_l_T[1]; 
		p_w[2] += camera_l_T[2];

		T camera_r_R_inv[4]; 
		camera_r_R_inv[0] = -camera_r_R[0];
		camera_r_R_inv[1] = -camera_r_R[1];
		camera_r_R_inv[2] = -camera_r_R[2];
		camera_r_R_inv[3] = camera_r_R[3];

		T camera_r_T_inv[3]; 
		ceres::QuaternionRotatePoint(camera_r_R_inv, camera_r_T, camera_r_T_inv);
		camera_r_T_inv[0] = T(-1.0 * camera_r_T_inv[0]);
		camera_r_T_inv[1] = T(-1.0 * camera_r_T_inv[1]);
		camera_r_T_inv[2] = T(-1.0 * camera_r_T_inv[2]);

		T p_r[3];
		ceres::QuaternionRotatePoint(camera_r_R_inv, p_w, p_r);
		p_r[0] += camera_r_T_inv[0]; 
		p_r[1] += camera_r_T_inv[1]; 
		p_r[2] += camera_r_T_inv[2];

		T xp = p_r[0] / p_r[2];
		T yp = p_r[1] / p_r[2];
		residuals[0] = xp - T(pts_r_u);
		residuals[1] = yp - T(pts_r_v);
		return true;
	}

	static ceres::CostFunction* Create(const double pts_l_u, 
																		const double pts_l_v, 
																		const double pts_r_u, 
																		const double pts_r_v) {
		return (new ceres::AutoDiffCostFunction<projError, 2, 4, 3, 4, 3, 1>(
			new projError(pts_l_u, pts_l_v, pts_r_u, pts_r_v)
			));
	}

	double pts_l_u;
	double pts_l_v;
	double pts_r_u;
	double pts_r_v;
};