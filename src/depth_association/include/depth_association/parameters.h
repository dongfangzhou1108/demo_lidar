#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;

extern std::string IMAGE_TOPIC;
extern std::string POINT_CLOUD_TOPIC;
extern std::string VELOCITY_TOPIC;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;

extern double m_fx;
extern double m_fy;
extern double m_cx;
extern double m_cy;

extern double cov_pixel;
extern double cov_pts;

void readParameters(ros::NodeHandle &n);
