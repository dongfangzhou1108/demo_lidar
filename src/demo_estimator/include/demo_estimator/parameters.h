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
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 2000; //feature in slide window
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;

extern double m_fx;
extern double m_fy;
extern double m_cx;
extern double m_cy;
extern double FOCAL_LENGTH;
extern int NUM_ITERATIONS;
extern double SOLVER_TIME;
extern double INIT_DEPTH;

void readParameters(ros::NodeHandle &n);
