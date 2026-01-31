#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64.hpp>

#include <utility>

class CmdVelToWheelAngularVel : public rclcpp::Node
{
public:
  CmdVelToWheelAngularVel()
  : rclcpp::Node("cmd_vel_to_wheel_angular_vel"),
    wheel_radius_(this->declare_parameter<double>("wheel_radius", 0.10)),
    wheel_separation_(this->declare_parameter<double>("wheel_separation", 0.45))
  {
    left_pub_ = this->create_publisher<std_msgs::msg::Float64>(kLeftTopic, 10);
    right_pub_ = this->create_publisher<std_msgs::msg::Float64>(kRightTopic, 10);

    cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      kCmdVelTopic,
      rclcpp::QoS(10),
      std::bind(&CmdVelToWheelAngularVel::onCmdVel, this, std::placeholders::_1));
  }

private:
  static constexpr const char* kCmdVelTopic = "cmd_vel";
  static constexpr const char* kLeftTopic = "left_motor_angular_vel";
  static constexpr const char* kRightTopic = "right_motor_angular_vel";

  void onCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    if (wheel_radius_ <= 0.0) {
      return;
    }

    const auto [omega_left, omega_right] = computeWheelAngularVel(msg->linear.x, msg->angular.z);

    std_msgs::msg::Float64 left_msg;
    std_msgs::msg::Float64 right_msg;
    left_msg.data = omega_left;
    right_msg.data = omega_right;

    left_pub_->publish(left_msg);
    right_pub_->publish(right_msg);
  }

  std::pair<double, double> computeWheelAngularVel(double v, double w) const
  {
    const double half_sep = wheel_separation_ * 0.5;
    const double v_left = v - w * half_sep;
    const double v_right = v + w * half_sep;
    return {v_left / wheel_radius_, v_right / wheel_radius_};
  }

  const double wheel_radius_;
  const double wheel_separation_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelToWheelAngularVel>());
  rclcpp::shutdown();
  return 0;
}
