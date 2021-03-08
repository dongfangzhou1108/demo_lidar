#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;

extern std::string IMAGE_TOPIC;
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

void readParameters(ros::NodeHandle &n);
