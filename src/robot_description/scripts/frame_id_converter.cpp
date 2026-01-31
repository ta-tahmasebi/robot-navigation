#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

class LaserScanFrameIdConverter final : public rclcpp::Node
{
public:
  LaserScanFrameIdConverter()
  : rclcpp::Node("frame_id_converter_node"),
    input_topic_(declare_parameter<std::string>("input_topic", "/gz_lidar/scan")),
    output_topic_(declare_parameter<std::string>("output_topic", "/scan")),
    frame_id_(declare_parameter<std::string>("frame_id", "rplidar_c1"))
  {
    pub_ = create_publisher<sensor_msgs::msg::LaserScan>(output_topic_, rclcpp::SensorDataQoS());
    sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&LaserScanFrameIdConverter::onScan, this, std::placeholders::_1));
  }

private:
  void onScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) const
  {
    auto out = *msg;
    out.header.frame_id = frame_id_;
    pub_->publish(out);
  }

  const std::string input_topic_;
  const std::string output_topic_;
  const std::string frame_id_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserScanFrameIdConverter>());
  rclcpp::shutdown();
  return 0;
}
