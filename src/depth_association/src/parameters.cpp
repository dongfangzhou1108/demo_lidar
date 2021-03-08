#include "depth_association/parameters.h"

int ROW;
int COL;

std::string IMAGE_TOPIC;
std::string POINT_CLOUD_TOPIC;
std::string VELOCITY_TOPIC;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
double m_fx;
double m_fy;
double m_cx;
double m_cy;
double cov_pixel;
double cov_pts;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image_topic"] >> IMAGE_TOPIC;
	fsSettings["point_cloud_topic"] >> POINT_CLOUD_TOPIC;
	fsSettings["velocity_topic"] >> VELOCITY_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
	cov_pixel = fsSettings["cov_pixel"];
	cov_pts = fsSettings["cov_pts"];

	cv::FileNode fsNode = fsSettings["projection_parameters"];
    m_fx = static_cast<double>(fsNode["fx"]);
    m_fy = static_cast<double>(fsNode["fy"]);
    m_cx = static_cast<double>(fsNode["cx"]);
    m_cy = static_cast<double>(fsNode["cy"]);

    WINDOW_SIZE = 20;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
