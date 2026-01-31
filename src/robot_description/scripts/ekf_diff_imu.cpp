#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

class EkfDiffImu final : public rclcpp::Node
{
public:
  EkfDiffImu()
  : rclcpp::Node("ekf_diff_imu"),
    wheel_odom_topic_(declare_parameter<std::string>("wheel_odom_topic", "/wheel_encoder/odom")),
    odom_topic_(declare_parameter<std::string>("odom_topic", "/ekf_diff_imu/odom")),
    imu_topic_(declare_parameter<std::string>("imu_topic", "/zed/zed_node/imu/data_raw")),
    sigma_v_(declare_parameter<double>("sigma_v", 0.10)),
    sigma_omega_(std::sqrt(declare_parameter<double>("sigma_omega", 1e-8)))
  {
    x_.setZero();
    P_.setIdentity();
    P_ *= 0.1;

    wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      wheel_odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&EkfDiffImu::onWheelOdom, this, std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&EkfDiffImu::onImu, this, std::placeholders::_1));

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

private:
  static double wrapToPi(double a)
  {
    while (a <= -M_PI) a += 2.0 * M_PI;
    while (a > M_PI) a -= 2.0 * M_PI;
    return a;
  }

  void predict(double v, double omega, double dt)
  {
    const double th = x_(2);
    const double c = std::cos(th);
    const double s = std::sin(th);

    x_(0) += v * c * dt;
    x_(1) += v * s * dt;
    x_(2) = wrapToPi(th + omega * dt);

    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 2) = -v * s * dt;
    F(1, 2) = v * c * dt;

    Eigen::Matrix<double, 3, 2> G;
    G << c * dt, 0.0,
         s * dt, 0.0,
         0.0,    dt;

    Eigen::Matrix2d Qu = Eigen::Matrix2d::Zero();
    Qu(0, 0) = sigma_v_ * sigma_v_;
    Qu(1, 1) = sigma_omega_ * sigma_omega_;

    P_ = F * P_ * F.transpose() + (G * Qu * G.transpose());
  }

  void updateYaw(double yaw_meas, double var_yaw)
  {
    Eigen::RowVector3d H;
    H << 0.0, 0.0, 1.0;

    const double y = wrapToPi(yaw_meas - x_(2));
    const double S = (H * P_ * H.transpose())(0, 0) + var_yaw;
    const Eigen::Vector3d K = (P_ * H.transpose()) / S;

    x_ += K * y;
    x_(2) = wrapToPi(x_(2));
    P_ = (Eigen::Matrix3d::Identity() - K * H) * P_;
  }

  void publish(const rclcpp::Time& stamp)
  {
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, x_(2));
    q.normalize();

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "odom";
    odom.child_frame_id = "base_link";

    odom.pose.pose.position.x = x_(0);
    odom.pose.pose.position.y = x_(1);
    odom.pose.pose.position.z = 0.0;

    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.pose.covariance.fill(0.0);
    odom.pose.covariance[0] = P_(0, 0);
    odom.pose.covariance[7] = P_(1, 1);
    odom.pose.covariance[35] = P_(2, 2);

    odom.twist.twist.linear.x = last_v_;
    odom.twist.twist.angular.z = last_omega_;
    odom_pub_->publish(odom);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = "odom";
    tf_msg.child_frame_id = "base_link";
    tf_msg.transform.translation.x = x_(0);
    tf_msg.transform.translation.y = x_(1);
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf_msg);
  }

  void onWheelOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const double v = msg->twist.twist.linear.x;
    const double omega = msg->twist.twist.angular.z;
    const rclcpp::Time t = msg->header.stamp;

    if (last_predict_time_.nanoseconds() == 0) {
      last_predict_time_ = t;
      last_v_ = v;
      last_omega_ = omega;
      publish(t);
      return;
    }

    double dt = (t - last_predict_time_).seconds();
    dt = std::max(dt, 1e-4);

    predict(v, omega, dt);
    last_v_ = v;
    last_omega_ = omega;

    publish(t);
    last_predict_time_ = t;
  }

  void onImu(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const auto& oq = msg->orientation;
    tf2::Quaternion qt(oq.x, oq.y, oq.z, oq.w);
    qt.normalize();

    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    tf2::Matrix3x3(qt).getRPY(roll, pitch, yaw);

    double var_yaw = 1e-8;
    const double cov_yaw = msg->orientation_covariance[8];
    if (cov_yaw >= 0.0) {
      var_yaw = std::max(cov_yaw, 1e-8);
    }

    updateYaw(yaw, var_yaw);
    publish(msg->header.stamp);
  }

  const std::string wheel_odom_topic_;
  const std::string odom_topic_;
  const std::string imu_topic_;

  const double sigma_v_;
  const double sigma_omega_;

  Eigen::Vector3d x_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d P_{Eigen::Matrix3d::Identity() * 0.1};

  rclcpp::Time last_predict_time_{0, 0, RCL_ROS_TIME};
  double last_v_{0.0};
  double last_omega_{0.0};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EkfDiffImu>());
  rclcpp::shutdown();
  return 0;
}
