#include "feature_tracker/feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker() 
{
}

void FeatureTracker::setMask() {
	mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

	vector<pair<int, pair<int, cv::Point2f>>> id_cnt_pts;
	for (unsigned int i = 0; i < cur_pts.size(); i++)
        id_cnt_pts.push_back(make_pair(ids[i], make_pair(track_cnt[i], cur_pts[i])));
    sort(id_cnt_pts.begin(), id_cnt_pts.end(), [](const pair<int, pair<int, cv::Point2f>> &a, const pair<int, pair<int, cv::Point2f>> &b)
        {
            return a.second.first > b.second.first;
        });
    ids.clear();
    track_cnt.clear();
	cur_pts.clear();

    for (auto &it : id_cnt_pts) {
        if (mask.at<uchar>(it.second.second) == 255) {
            ids.push_back(it.first);
            track_cnt.push_back(it.second.first);
			cur_pts.push_back(it.second.second);
            cv::circle(mask, it.second.second, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::rejectWithF() {
	if (cur_pts.size() >= 8) {
		vector<uchar> status;
        cv::findFundamentalMat(prev_pts, cur_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);

        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;

        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
	}
}

void FeatureTracker::addPoints() {
    for (auto &p : n_pts) {
        cur_pts.push_back(p);
        ids.push_back(n_id);
		n_id ++;
        track_cnt.push_back(1);
    }
	n_pts.clear();
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time) {
	cv::Mat img = _img;
	if (cur_img.empty()) {
		prev_time = cur_time = _cur_time;
		prev_img = cur_img = img;
	}
    else {
		prev_time = cur_time;
		prev_img = cur_img;
		prev_pts = cur_pts;
		cur_time = _cur_time;
		cur_img = img;
	}
	cur_pts.clear();

	if (prev_pts.size() > 0) {
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;

        reduceVector(ids, status);
        reduceVector(track_cnt, status);
		reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
		for (auto &n : track_cnt)
			n++;
		rejectWithF();
		setMask();
	}

	int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
	if (n_max_cnt > 0) {
		cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
		addPoints();
	}
}