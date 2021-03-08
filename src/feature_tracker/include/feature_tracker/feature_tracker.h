#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "feature_tracker/parameters.h"
#include "feature_tracker/tic_toc.h"

using namespace std;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<int> &v, vector<uchar> status);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

	void setMask();

	void rejectWithF();

	void addPoints();

	void readImage(const cv::Mat &_img,double _cur_time);

    cv::Mat mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts;
    vector<int> ids;
    vector<int> track_cnt;
    double cur_time = 0;
    double prev_time = 0;

    static int n_id;
};
