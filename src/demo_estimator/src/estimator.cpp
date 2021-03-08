#include "demo_estimator/estimator.h"

Estimator::Estimator() {
	frame_count = 0;
	f_manager.feature.clear();
	solver_flag = INITIAL;
}

void Estimator::setParameter() {
	ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
}

void Estimator::processImage(const std::map<int, vector<pair<int, Eigen::Matrix<double, 6, 1>>>> &image, const std_msgs::Header &header) {
	f_manager.addFeature(frame_count, image);
	if (solver_flag == INITIAL) {
		if (frame_count == WINDOW_SIZE) {
			ROS_INFO("SFM initialization begin!");
			bool initial = initialStructure();
			ROS_ASSERT(initial == true);
			if (initial) {
				solver_flag = NON_LINEAR;
				f_manager.triangulate(Rs, Ts);
/* 
				f_manager.triangulate(Rs, Ts);
				optimization_sfm();
				for (int i = 0; i <= WINDOW_SIZE; i++)
					ROS_INFO("After optimazation Ts[%d]: %f, %f, %f", i, Ts[i].x(), Ts[i].y(), Ts[i].z());
				ROS_INFO("T is %f, %f, %f", Ts[0].x(), Ts[0].y(), Ts[0].z());
 */
				optimization();
				slideWindow();
			}
			else
				slideWindow();
		}
		else
			frame_count ++;
	}
	else {
		TicToc t_nonlinear;
		getLatestPoseByDepthPnP();
		// ROS_INFO("Before optimization, latest T is %f, %f, %f", Ts[WINDOW_SIZE].x(), Ts[WINDOW_SIZE].y(), Ts[WINDOW_SIZE].z());
		f_manager.triangulate(Rs, Ts);
		optimization();
		ROS_INFO("optimation use time: %f ms, T[0] is %f, %f, %f", t_nonlinear.toc(), Ts[0].x(), Ts[0].y(), Ts[0].z());
		// optimization_sfm();
/* 		ROS_INFO("optimation use time: %f ms, T[0] is %f, %f, %f", t_nonlinear.toc(), Ts[0].x(), Ts[0].y(), Ts[0].z());
		for (int i = 0; i <= WINDOW_SIZE; i++)
			ROS_INFO("T[%d] is %f, %f, %f", i, Ts[i].x(), Ts[i].y(), Ts[i].z()); */
		slideWindow();
	}
}

bool Estimator::initialStructure() {
	Matrix3d relative_R;  
	Vector3d relative_T;
	int l;
	bool initial = get_frame_to_init(relative_R, relative_T, l);
	if (!initial)
		return false;

	vector<SFMFeature> sfm_f = feature2SFMFeature(f_manager.feature);
	Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
	GlobalSFM sfm;
	initial = sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points);
	for (int i = 0; i <= frame_count; i++) {
		Rs[i] = Q[i].matrix();
		Ts[i] = T[i];
	}

	return initial;
}

bool Estimator::get_frame_to_init(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
	for (int i = 0; i < WINDOW_SIZE; i++) {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
		if (corres.size() > 20) {
			double sum_parallax = 0;
            double average_parallax = 0;
			for (int j = 0; j < int(corres.size()); j++) {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
			}
			average_parallax = 1.0 * sum_parallax / int(corres.size());
			if (average_parallax * (m_fx + m_fy) / 2 > 30 && solveRelativeRT(corres, relative_R, relative_T)) {
				l = i; //compute pose with respect to l th frame
				ROS_DEBUG("Find %d th frame to do initialization!", l);
				return true;
			}
		}
	}
	return false;
}

bool Estimator::solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation) {
    vector<cv::Point2f> ll, rr;
    for (int i = 0; i < int(corres.size()); i++) { 
        ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
        rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
    }
    cv::Mat mask;
    cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / ((m_fx + m_fy) / 2) , 0.99, mask);
	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask); //give pose T_rr_ll
	Matrix3d R;
	Vector3d T;
    for (int i = 0; i < 3; i++) {   
        T(i) = trans.at<double>(i, 0);
        for (int j = 0; j < 3; j++)
			R(i, j) = rot.at<double>(i, j);
    }
	Rotation = R.transpose();
    Translation = -R.transpose() * T; //compute pose T_ll_rr
	ROS_DEBUG("corres size: %d, inlier_cnt: %d, transform is %f, %f, %f", int(corres.size()), inlier_cnt, Translation(0), Translation(1), Translation(2));
    if(inlier_cnt > 12)
        return true;
    else
        return false;
}

vector<SFMFeature> Estimator::feature2SFMFeature(list<FeaturePerId>& feature) {
	vector<SFMFeature> sfm_f;
	for (auto &it_per_id : feature) {
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
		int f_count = it_per_id.start_frame;
		for (auto &it_per_frame : it_per_id.feature_per_frame){
			Vector3d pts_j = it_per_frame.point;
			tmp_feature.observation.push_back(make_pair(f_count, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
			tmp_feature.velo_depth.push_back(make_pair(f_count, it_per_frame.depth));
			f_count ++;
		}
		sfm_f.push_back(tmp_feature);
	}
	return sfm_f;
}

void Estimator::slideWindow() {
	if(solver_flag == INITIAL)
		f_manager.removeBack();
	else {
		back_R0 = Rs[0];
        back_T0 = Ts[0];
		for (int i = 0; i < WINDOW_SIZE; i++) {
                Rs[i].swap(Rs[i + 1]);
				Ts[i].swap(Ts[i + 1]);
		}
		Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
		Ts[WINDOW_SIZE] = Ts[WINDOW_SIZE - 1];
		f_manager.removeBackShiftDepth(back_R0, back_T0, Rs[0], Ts[0]);
	}
}

void Estimator::optimization() {
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
	//loss_function = new ceres::HuberLoss(1.0);

    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
		if ( i != WINDOW_SIZE)
			problem.SetParameterBlockConstant(para_Pose[i]);
    }

	vector2double();
	int feature_index = -1;
	for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = int(it_per_id.feature_per_frame.size());
        if (it_per_id.used_num < 2 || it_per_id.start_frame > WINDOW_SIZE - 2)
            continue;

		feature_index ++;
		int idx_l = it_per_id.start_frame;
		int idx_r = idx_l - 1;
		Vector3d un_pts_l = it_per_id.feature_per_frame[0].point;
		for (auto &it_per_frame : it_per_id.feature_per_frame) {
            idx_r++;
            if (idx_l == idx_r)
                continue;

			Vector3d un_pts_r = it_per_frame.point;
			ProjectionFactor *f = new ProjectionFactor(un_pts_l, un_pts_r);
			problem.AddResidualBlock(f, loss_function, para_Pose[idx_l], para_Pose[idx_r], para_Feature[feature_index]);
		}
	}

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    // options.max_num_iterations = NUM_ITERATIONS;
	// options.max_solver_time_in_seconds = SOLVER_TIME;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
/* 
	if (summary.termination_type !=  ceres::CONVERGENCE) {
		ROS_WARN("BA not converge, and summary.final_cost is: %f",  summary.final_cost );
		ROS_ASSERT(summary.termination_type == ceres::CONVERGENCE);
	}
 */
	double2vector();
}

void Estimator::optimization_sfm() {
	TicToc t_whole;
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	double para_R[WINDOW_SIZE + 1][4];
	double para_T[WINDOW_SIZE + 1][3];
	for (int i = 0; i <= WINDOW_SIZE; i++) {
		Matrix3d R = Rs[i].inverse();
		Quaterniond Q(R);
		Vector3d T = -1 * R * Ts[i];
		para_T[i][0] =  T.x(); 
		para_T[i][1] =  T.y(); 
		para_T[i][2] =  T.z();
		para_R[i][0] = Q.w(); 
		para_R[i][1] = Q.x(); 
		para_R[i][2] = Q.y(); 
		para_R[i][3] = Q.z();
		problem.AddParameterBlock(para_R[i], 4, local_parameterization);
		problem.AddParameterBlock(para_T[i], 3);
		if (i == 0 || i == WINDOW_SIZE) {
			problem.SetParameterBlockConstant(para_T[i]);
			problem.SetParameterBlockConstant(para_R[i]);
		}
	}
	for (auto &it_per_id : f_manager.feature) {
		if (int(it_per_id.feature_per_frame.size()) >= 2 && it_per_id.start_frame <= WINDOW_SIZE - 2) {
			int idx = it_per_id.start_frame;
			for (auto &it_per_frame : it_per_id.feature_per_frame) {
				ceres::CostFunction* cost_function = ReprojectionError3D::Create(it_per_frame.point.x(), it_per_frame.point.y());
				problem.AddResidualBlock(cost_function, NULL, para_R[idx], para_T[idx],  it_per_id.position);
				idx ++;
			}
		}
		else
			continue;
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	// options.max_solver_time_in_seconds = 0.2;
	// options.num_threads = 2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	for (int i = 0; i <= WINDOW_SIZE; i++) {
		Quaterniond Q;
		Q.w() = para_R[i][0]; 
		Q.x() = para_R[i][1]; 
		Q.y() = para_R[i][2]; 
		Q.z() = para_R[i][3]; 
		Q = Q.inverse().normalized();
		Rs[i] = Q.toRotationMatrix();
		Ts[i] = -1 * (Rs[i] * Vector3d(para_T[i][0], para_T[i][1], para_T[i][2]));
	}
}

void Estimator::optimization_depth() {
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
	/* vector to double for ceres optimization */
	for (int i = 0; i <= WINDOW_SIZE; i++) {
        para_T[i][0] = Ts[i].x();
        para_T[i][1] = Ts[i].y();
        para_T[i][2] = Ts[i].z();
        Quaterniond q{Rs[i]};
        para_R[i][0] = q.x();
        para_R[i][1] = q.y();
        para_R[i][2] = q.z();
        para_R[i][3] = q.w();
		problem.AddParameterBlock(para_R[i], 4, local_parameterization);
		problem.AddParameterBlock(para_T[i], 3);
	}
	VectorXd dep = f_manager.getDepthVector();
	for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

	int feature_index = -1;
	for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = int(it_per_id.feature_per_frame.size());
        if (it_per_id.used_num < 2 || it_per_id.start_frame > WINDOW_SIZE - 2)
            continue;
		feature_index ++;
		int start_frame = it_per_id.start_frame;
		int frame = start_frame - 1;
		Vector3d un_pts_l = it_per_id.feature_per_frame[0].point;
		for (auto &it_per_frame : it_per_id.feature_per_frame) {
			frame ++;
			Vector3d un_pts_r = it_per_frame.point;
			double velo_depth = it_per_frame.depth;
			if (velo_depth > 0 && frame ==  start_frame) {
				ceres::CostFunction* cost_function = veloTriError::Create(velo_depth);
				problem.AddResidualBlock(cost_function, NULL, para_Feature[feature_index]);
			}
			else if (velo_depth > 0 && frame != start_frame) { //if have velo depth
				ceres::CostFunction* cost_function = veloDepthError3D::Create(un_pts_l.x(), un_pts_l.y(),
																																					un_pts_r.x(), un_pts_r.y(), velo_depth);
				problem.AddResidualBlock(cost_function, loss_function, 
																		para_R[start_frame], para_T[start_frame],
																		para_R[frame], para_T[frame], 
																		para_Feature[feature_index]);
			}
			else if (velo_depth < 0 && frame != start_frame) {
				ceres::CostFunction* cost_function = projError::Create(un_pts_l.x(), un_pts_l.y(),
																																un_pts_r.x(), un_pts_r.y());
				problem.AddResidualBlock(cost_function, loss_function, 
																		para_R[start_frame], para_T[start_frame],
																		para_R[frame], para_T[frame], 
																		para_Feature[feature_index]);
			}
		}
	}
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
	ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

	/* double to vector */
    Matrix3d origin_R0 = Rs[0];
    Vector3d origin_T0 = Ts[0];
	Matrix3d origin_R00 = Quaterniond(para_R[0][3], para_R[0][0], para_R[0][1], para_R[0][2]).normalized().toRotationMatrix();
	Matrix3d rot_diff = (origin_R0.inverse() * origin_R00).inverse(); 
	for (int i = 0; i <= WINDOW_SIZE; i++) {
		Rs[i] = rot_diff * Quaterniond(para_R[i][3], para_R[i][0], para_R[i][1], para_R[i][2]).normalized().toRotationMatrix();
		Ts[i] = rot_diff * Vector3d(para_T[i][0] - para_T[0][0], para_T[i][1] - para_T[0][1], para_T[i][2] - para_T[0][2]) + origin_T0;
	}

	dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
}

void Estimator::vector2double() {
	for (int i = 0; i <= WINDOW_SIZE; i++) {
        para_Pose[i][0] = Ts[i].x();
        para_Pose[i][1] = Ts[i].y();
        para_Pose[i][2] = Ts[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();
	}
	VectorXd dep = f_manager.getDepthVector();
	for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
}

void Estimator::double2vector() {
    Matrix3d origin_R0 = Rs[0];
    Vector3d origin_T0 = Ts[0];
	Matrix3d origin_R00 = Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]).normalized().toRotationMatrix();
	Matrix3d rot_diff = (origin_R0.inverse() * origin_R00).inverse(); 

	for (int i = 0; i <= WINDOW_SIZE; i++) {
		Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
		Ts[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0], para_Pose[i][1] - para_Pose[0][1], para_Pose[i][2] - para_Pose[0][2]) + origin_T0;
	}

	VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
}

void Estimator::getLatestPoseByPnP() {
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (auto &it_per_id : f_manager.feature) {
		if (it_per_id.endFrame() == WINDOW_SIZE) {
			if (it_per_id.estimated_depth > 0) {
				cv::Point2f pts_2(it_per_id.feature_per_frame.back().point.x(), 
													it_per_id.feature_per_frame.back().point.y());
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(it_per_id.position[0], it_per_id.position[1], it_per_id.position[2]);
				pts_3_vector.push_back(pts_3);
			}
			else
				continue;
		}
		else
			continue;
	}

	cv::Mat tmp_r, rvec, t, D;
	cv::eigen2cv(Rs[WINDOW_SIZE], tmp_r);
	cv::Rodrigues(tmp_r, rvec); 
	cv::eigen2cv(Ts[WINDOW_SIZE], t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	ROS_ASSERT(pnp_succ == true);
	cv::Mat r; 
	cv::Rodrigues(rvec, r); 
	MatrixXd R_pnp; 
	cv::cv2eigen(r, R_pnp); 
	MatrixXd T_pnp; 
	cv::cv2eigen(t, T_pnp);
	Rs[WINDOW_SIZE] = R_pnp.inverse(); 
	Ts[WINDOW_SIZE] = -1.0 * R_pnp * T_pnp;
}

void Estimator::getLatestPoseByDepthPnP() {
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (auto &it_per_id : f_manager.feature) {
		if (it_per_id.endFrame() == WINDOW_SIZE) {
			int it_per_id_size = int(it_per_id.feature_per_frame.size());
			if (it_per_id_size < 2)
				continue;
			int idx = it_per_id_size - 2;
			if (it_per_id.feature_per_frame[idx].depth > 0) {
				cv::Point2f pts_2(it_per_id.feature_per_frame.back().point.x(), 
													it_per_id.feature_per_frame.back().point.y());
				pts_2_vector.push_back(pts_2);
				double depth = it_per_id.feature_per_frame[idx].depth;
				cv::Point3f pts_3(depth * it_per_id.feature_per_frame[idx].point.x(),
													depth * it_per_id.feature_per_frame[idx].point.y(),
													depth);
				pts_3_vector.push_back(pts_3);
			}
			else
				continue;
		}
		else
			continue;
	}
	cv::Mat tmp_r, rvec, t, D;
	Matrix3d R = Rs[WINDOW_SIZE - 2].inverse() * Rs[WINDOW_SIZE - 1];
	Vector3d T = Ts[WINDOW_SIZE - 1] - Ts[WINDOW_SIZE - 2];
	if(int(pts_2_vector.size() < 10)) {
		ROS_WARN("Pts for PnP too small, use R, T of [WINDOW_SIZE - 2] - [WINDOW_SIZE - 1]");
		Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1] *  R; 
		Ts[WINDOW_SIZE] = Ts[WINDOW_SIZE - 1] + Rs[WINDOW_SIZE - 1] * T;
	}
	else {
		Matrix3d R_ = R.inverse();
		cv::eigen2cv(R_, tmp_r);
		cv::Rodrigues(tmp_r, rvec); 
		Vector3d T_ = -1.0 * R_ * T;
		cv::eigen2cv(T_, t);
		cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		bool pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
		ROS_ASSERT(pnp_succ == true);
		cv::Mat r; 
		cv::Rodrigues(rvec, r); 
		MatrixXd R_pnp; 
		cv::cv2eigen(r, R_pnp); 
		MatrixXd T_pnp; 
		cv::cv2eigen(t, T_pnp);
		//here calc is pose_k-1_k-2, we need pose_w_latest
		T_pnp = -1.0 * R_pnp.inverse() * T_pnp;
		ROS_INFO("Find %d point for PnP, T is %f, %f, %f", int(pts_2_vector.size()), T_pnp(0), T_pnp(1), T_pnp(2));
		Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1] *  R_pnp.inverse(); 
		Ts[WINDOW_SIZE] = Ts[WINDOW_SIZE - 1] + Rs[WINDOW_SIZE - 1] * T_pnp;
	}
}
