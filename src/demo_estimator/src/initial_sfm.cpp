#include "demo_estimator/initial_sfm.h"

bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
	const Matrix3d relative_R, const Vector3d relative_T,
	vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points) {
	feature_num = sfm_f.size();
	//here q and T is pose_world(l)_frame_arbitrary_frame
	q[l].w() = 1; q[l].x() = 0; q[l].y() = 0; q[l].z() = 0; T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R); T[frame_num - 1] = relative_T;

	// when triangulate, we need pose_arbitrary_frame_word_frame, 
	// because we use projection and trans point from world to one frame;
	Quaterniond c_Quat[frame_num];
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];
	// process l th frame and the last frame
	c_Quat[l] = q[l].inverse(); 
	c_Rotation[l] = c_Quat[l].toRotationMatrix(); 
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l]; 
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse(); 
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix(); 
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1]; 
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];
	//1.triangulate l th frame with the last frame
	//2.triangulate all frame with the last frame
	for (int i = l; i < frame_num - 1 ; i++) {
		if (i > l) {
			Matrix3d R_initial = c_Rotation[i - 1]; Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial; c_Translation[i] = P_initial; c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i]; Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3.triangulate all frame with l th frame
	for (int i = l + 1; i < frame_num - 1; i++) {
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	}
	//4.triangulate all frame before l th with l th frame
	for (int i = l - 1; i >= 0; i--) {
		Matrix3d R_initial = c_Rotation[i + 1]; Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial; c_Translation[i] = P_initial; c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i]; Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5.triangulate all other feature
	for (auto& sfmFeature : sfm_f) {
		if (sfmFeature.state == true)
			continue;
		if (sfmFeature.observation.size() >= 2) {
			int frame0 = sfmFeature.observation[0].first; 
			int frame1 = sfmFeature.observation[1].first;
			Vector2d point0 = sfmFeature.observation[0].second; 
			Vector2d point1 = sfmFeature.observation[1].second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame0], Pose[frame1], point0, point1, point_3d);
			sfmFeature.state = true; 
			sfmFeature.position[0] = point_3d(0); 
			sfmFeature.position[1] = point_3d(1); 
			sfmFeature.position[2] = point_3d(2);
		}
	}

	// ceres nolinear optimazition
	/*
	用户在调用AddResidualBlock( )时其实已经隐式地向Problem传递了参数模块，
	但在一些情况下，需要用户显示地向Problem传入参数模块（通常出现在需要对优化参数进行重新参数化的情况）。
	Ceres提供了Problem::AddParameterBlock( )函数用于用户显式传递参数模块　。
	*/
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	for (int i = 0; i < frame_num; i++) {
		c_translation[i][0] =  c_Translation[i].x(); 
		c_translation[i][1] =  c_Translation[i].y(); 
		c_translation[i][2] =  c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w(); 
		c_rotation[i][1] = c_Quat[i].x(); 
		c_rotation[i][2] = c_Quat[i].y(); 
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l)
			problem.SetParameterBlockConstant(c_rotation[i]); //set rotation of l th frame keep constant
		if (i == l || i == frame_num - 1)
			problem.SetParameterBlockConstant(c_translation[i]); //set trans of lth and last frame keep constant
	}
	for (auto& feature : sfm_f) {
		if (feature.state == false)
			continue;
		for (int i = 0; i < int(feature.observation.size()); i++) {
			int l = feature.observation[i].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(feature.observation[i].second.x(), feature.observation[i].second.y());
			problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],  feature.position);
		}
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	// if (summary.termination_type !=  ceres::CONVERGENCE || summary.final_cost > 5e-03) {
	if (summary.termination_type !=  ceres::CONVERGENCE) {
		ROS_WARN("Vision only BA not converge, and summary.final_cost is: %f",  summary.final_cost );
		return false;
	}
/* 	for (int i = 0; i < frame_num; i++)
		ROS_INFO("trans of %d th: %f, %f, %f", i, c_translation[i][0], c_translation[i][1], c_translation[i][2]); */

	/*
	ceres::Problem problem_scale;
	// double scale = initScale(l, frame_num-1, sfm_f);
	double scale = 5.0;
	ROS_WARN("Init scale is %f", scale);
	//深度关联非常失败
	for (auto& feature : sfm_f) {
		ROS_INFO("One feature");
		for (auto& ob : feature.velo_depth) {
			if (ob.second > 0)
				ROS_WARN("depth: %f", ob.second);
		}
	}

	for (int i = 0; i < frame_num; i++) {
		problem_scale.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem_scale.AddParameterBlock(c_translation[i], 3);
		problem_scale.SetParameterBlockConstant(c_rotation[i]);
		problem_scale.SetParameterBlockConstant(c_translation[i]);
	}
	for (auto& feature : sfm_f) {
		if (feature.state == false)
			continue;
		problem_scale.AddParameterBlock(feature.position, 3);
		problem_scale.SetParameterBlockConstant(feature.position);
		for (int i = 0; i < int(feature.observation.size()); i++) {
			int l = feature.observation[i].first;
			if (feature.velo_depth[i].second < 0 )
				continue;
			else {
				ceres::CostFunction* cost_function = DepthError3D::Create(feature.velo_depth[i].second);
				problem_scale.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],  feature.position, &scale);
			}
		}
	}
	ceres::Solver::Options options_scale;
	options.linear_solver_type = ceres::DENSE_QR;
	ceres::Solver::Summary summary_scale;
	ceres::Solve(options_scale, &problem_scale, &summary_scale);
	if (summary_scale.termination_type !=  ceres::CONVERGENCE || summary_scale.final_cost > 5e-03) {
		ROS_WARN("Scale: %f, scale not converge, and summary.final_cost is: %f",  scale, summary_scale.final_cost );
		return false;
	}
	ROS_INFO("Scale: %f", scale);
	*/
	//calc pose of velo depth by PnP
	Vector3d Ts_PnP[(WINDOW_SIZE)];
	for (int i = 0; i < frame_num - 1; i++) {
		Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
		Eigen::Vector3d T = Eigen::Vector3d::Zero();
		bool PnP_success = depthFeaturePnP(i, i + 1, sfm_f, R, T);
		ROS_ASSERT(PnP_success == true);
		Ts_PnP[i] = T;
	}

	for (int i = 0; i < frame_num; i++) {
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
	}

	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}

	//p_0_1 = p_0_w * p_1_w.inverse()
	double scale = 0;
	for (int i = 0; i < WINDOW_SIZE; i++) {
		int idx_r = i + 1; 
		Vector3d t_sfm = T[idx_r] - T[i];
		double t_sfm_norm = t_sfm.norm();
		double t_PnP_norm = Ts_PnP[i].norm();
		scale = scale + t_PnP_norm / t_sfm_norm;
	}
	scale = scale / double(WINDOW_SIZE);

	ROS_INFO("Finish sfm initial! Scale is: %f", scale);

	for (int i = 0; i < frame_num; i++) {
		T[i] = scale * T[i];
		ROS_INFO("sfm_T: %f, %f, %f", T[i].x(), T[i].y(), T[i].z());
	}

	return true;
}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
	int frame1, Eigen::Matrix<double, 3, 4> &Pose1, vector<SFMFeature> &sfm_f) {
	for (int i = 0; i < feature_num; i++) {
		if (sfm_f[i].state == true)
			continue;
		Vector2d point0; Vector2d point1;
		bool have0 = false; bool have1 = false;
		for (auto& observe : sfm_f[i].observation) {
			if (observe.first == frame0) {
				have0 = true;
				point0 = observe.second;
			}
			else if (observe.first == frame1) {
				have1 = true;
				point1 = observe.second;
			}
			if (have0 && have1) {
				Vector3d point_3d;
				triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
				sfm_f[i].state = true;
				sfm_f[i].position[0] = point_3d(0); sfm_f[i].position[1] = point_3d(1); sfm_f[i].position[2] = point_3d(2);
				break;
			}
		}
	}
}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
	Vector2d &point0, Vector2d &point1, Vector3d &point_3d) {
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f) {
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (auto& feature : sfm_f) {
		if (feature.state == true) {
			for (auto& observe : feature.observation) {
				if (observe.first == i) {
					cv::Point2f pts_2(observe.second(0), observe.second(1));
					pts_2_vector.push_back(pts_2);
					cv::Point3f pts_3(feature.position[0], feature.position[1], feature.position[2]);
					pts_3_vector.push_back(pts_3);
					break;
				}
			}
		}
		else
			continue;
	}
	if (int(pts_2_vector.size()) <  15)
		ROS_WARN("Tiangulated point is too small!");

	cv::Mat tmp_r, rvec, t, D;
	cv::eigen2cv(R_initial, tmp_r); cv::Rodrigues(tmp_r, rvec); cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if (pnp_succ == false)
		return false;
	cv::Mat r; cv::Rodrigues(rvec, r); MatrixXd R_pnp; cv::cv2eigen(r, R_pnp); MatrixXd T_pnp; cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp; P_initial = T_pnp;
	return true;
}

bool GlobalSFM::depthFeaturePnP(int frame_l, int frame_r, vector<SFMFeature> &sfm_f, Matrix3d &R_initial, Vector3d &P_initial) {
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (auto& onefeature : sfm_f) {
		bool have_pixel = false;
		bool have_depth = false;
		cv::Point2f pts_2;
		cv::Point3f pts_3;
		for (int i = 0; i < int(onefeature.observation.size()); i++) {
			if (onefeature.observation[i].first == frame_l) {
				have_pixel = true;
				pts_2.x = onefeature.observation[i].second.x();
				pts_2.y = onefeature.observation[i].second.y();
				break;
			}
		}
		for (int i = 0; i < int(onefeature.velo_depth.size()); i++) {
			if (onefeature.velo_depth[i].first == frame_r && onefeature.velo_depth[i].second > 0) {
				have_depth = true;
				double depth = onefeature.velo_depth[i].second;
				pts_3.x = depth * onefeature.observation[i].second.x();
				pts_3.y = depth * onefeature.observation[i].second.y();
				pts_3.z = depth;
				break;
			}
		}
		if (have_pixel && have_depth) {
			pts_2_vector.push_back(pts_2);
			pts_3_vector.push_back(pts_3);
		}
	}
	cv::Mat tmp_r, rvec, t, D;
	cv::eigen2cv(R_initial, tmp_r); 
	cv::Rodrigues(tmp_r, rvec); 
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if (pnp_succ == false) {
		return false;
		ROS_WARN("PnP with depth false!!!!!");
	}

	cv::Mat r; 
	cv::Rodrigues(rvec, r); 
	MatrixXd R_pnp; 
	cv::cv2eigen(r, R_pnp); 
	R_initial = R_pnp;

	MatrixXd T_pnp; 
	cv::cv2eigen(t, T_pnp);
	P_initial = T_pnp;

	ROS_INFO("depth_PnP_T: %f, %f, %f, corres size: %d", P_initial.x(), P_initial.y(), P_initial.z(), int(pts_2_vector.size()));

	return true;
}

double GlobalSFM::initScale(int frame_l, int frame_r, vector<SFMFeature> &sfm_f) {
	double sum_depth = 0; double sum_tri = 0;
	for (auto& sfm_ : sfm_f) {
		if (sfm_.state == false)
			continue;
		bool have0 = false; bool have1 = false; bool velo = false;
		double depth = 0;
		for (int i = 0; i < int(sfm_.observation.size()); i++) {
			if (sfm_.observation[i].first == frame_l) {
				depth = sfm_.velo_depth[i].second;
				if (depth > 0)
					velo = true;
				have0 = true;
			}
			else if (sfm_.observation[i].first == frame_r)
				have1 = true;
			if (have0 && have1 && velo) {
				sum_depth = sum_depth + depth;
				sum_tri = sum_tri + sfm_.position[2];
				ROS_DEBUG("velo_depth: %f, tri_depth: %f", depth, sfm_.position[2]);
				break;
			}
		}
	}
	double scale = sum_depth / sum_tri;
	return scale;
}