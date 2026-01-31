#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <string>

class MotorCommandNode final : public rclcpp::Node
{
public:
  MotorCommandNode()
  : rclcpp::Node("motor_command_node"),
    input_topic_(declare_parameter<std::string>("input_topic", "/motor_commands")),
    left_topic_(declare_parameter<std::string>("left_topic", "/left_motor_rpm")),
    right_topic_(declare_parameter<std::string>("right_topic", "/right_motor_rpm"))
  {
    left_pub_ = create_publisher<std_msgs::msg::Float64>(left_topic_, 10);
    right_pub_ = create_publisher<std_msgs::msg::Float64>(right_topic_, 10);

    sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      input_topic_, 10,
      std::bind(&MotorCommandNode::onCommands, this, std::placeholders::_1));
  }

private:
  void onCommands(const std_msgs::msg::Float64MultiArray::SharedPtr msg) const
  {
    if (msg->data.size() < 2) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Expected at least 2 motor commands, got %zu", msg->data.size());
      return;
    }

    std_msgs::msg::Float64 left;
    std_msgs::msg::Float64 right;
    left.data = msg->data[0];
    right.data = msg->data[1];

    left_pub_->publish(left);
    right_pub_->publish(right);
  }

  const std::string input_topic_;
  const std::string left_topic_;
  const std::string right_topic_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MotorCommandNode>());
  rclcpp::shutdown();
  return 0;
}
