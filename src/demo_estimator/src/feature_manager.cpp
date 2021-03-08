#include "demo_estimator/feature_manager.h"
using namespace std;

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

void FeatureManager::addFeature(int frame_count, const std::map<int, vector<pair<int, Eigen::Matrix<double, 6, 1>>>> &image) {
	last_track_num = 0;
	last_track_with_depth_num = 0;
	for (auto &id_pts : image) {
		int feature_id = id_pts.first;
		FeaturePerFrame f_per_fra(id_pts.second[0].second);
        auto it = find_if(feature.begin(), feature.end(), [feature_id] (const FeaturePerId &it) { return it.feature_id == feature_id; });
		if (it == feature.end()) {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
		else if (it->feature_id == feature_id) {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
			if ( id_pts.second[0].second(5, 0) > 0)
				last_track_with_depth_num ++;
        }
	}
	ROS_DEBUG("last_track_num is: %d, last_track_with_depth_num is: %d", last_track_num, last_track_with_depth_num);
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature) {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;
            a = it.feature_per_frame[idx_l].point;
            b = it.feature_per_frame[idx_r].point;
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::removeBack() {
	for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
		it_next ++;
		if (it->start_frame != 0)
            it->start_frame--;
		else {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
		}
	}
}

void FeatureManager::triangulate(Matrix3d Rs[], Vector3d Ts[]) {
	for (auto &it_per_id : feature) {
		if (it_per_id.feature_per_frame.size() < 2 || it_per_id.start_frame > WINDOW_SIZE - 2)
			continue;
		// if (it_per_id.estimated_depth > 0)
        //     continue;

		int row_idx = 0;
		int idx_l = it_per_id.start_frame;
		int idx_r = it_per_id.start_frame - 1;
		Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
		// Rs,Ts is pose_0thCam_KthCam, when triangulate, we need pose_KthCam_IthCam(the point firstly observed - Init)
		for (auto &it_per_frame : it_per_id.feature_per_frame) {
			idx_r ++;
			Eigen::Matrix3d R = Rs[idx_l].transpose() * Rs[idx_r];
            Eigen::Vector3d T = Rs[idx_l].transpose() * (Ts[idx_r] - Ts[idx_l]);
			Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * T;
			Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(row_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(row_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
		}

        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double dep = svd_V[2] / svd_V[3];
		it_per_id.estimated_depth = dep; //May be this value is too small, such < 0.1 !!!
		if (it_per_id.estimated_depth < 0.1){
            it_per_id.estimated_depth = INIT_DEPTH;
        }

		Vector3d pts_;
		pts_ << svd_V[0] / svd_V[3], svd_V[1] / svd_V[3], svd_V[2] / svd_V[3];
		pts_ = Rs[idx_l] * pts_ + Ts[idx_l];
		it_per_id.position[0] = pts_.x();
		it_per_id.position[1] = pts_.y();
		it_per_id.position[2] = pts_.z();

	}
}

int FeatureManager::getFeatureCount() {
    int cnt = 0;
    for (auto &it : feature) {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 2 && it.start_frame <= WINDOW_SIZE - 2)
            cnt++;
    }
    return cnt;
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 2 || it_per_id.start_frame > WINDOW_SIZE - 2)
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::setDepth(const VectorXd &x) {
	int idx = 0;
	for (auto& per_id : feature) {
		if (per_id.feature_per_frame.size() < 2 || per_id.start_frame > WINDOW_SIZE - 2)
			continue;
		per_id.estimated_depth = 1.0 / x(idx);
		idx ++;
		
		if (per_id.estimated_depth < 0)
            per_id.solve_flag = 2;
        else
            per_id.solve_flag = 1;
	}
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P) {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
		it_next++;
        if (it->start_frame != 0)
            it->start_frame--;
		else {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2){
                feature.erase(it);
                continue;
            }
            else {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;

				pts_i << it->position[0], it->position[1], it->position[2];
				w_pts_i = marg_R * pts_i + marg_P;
				pts_j = new_R.transpose() * (w_pts_i - new_P);
				it->position[0] = pts_j.x();
				it->position[1] = pts_j.y();
				it->position[2] = pts_j.z();
            }
		}
	}
}