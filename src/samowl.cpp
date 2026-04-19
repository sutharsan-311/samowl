#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace enc = sensor_msgs::image_encodings;

namespace
{
struct Options
{
  std::string image;
  std::string depth_image;
  std::string rgb_topic;
  std::string depth_topic;
  std::string camera_info_topic;
  std::string text;
  std::string output_mask{"mask.png"};
  std::string output_boundary{"boundary.png"};
  std::string output_depth_mask{"masked_depth.png"};
  std::string output_points{"object_points_map.pcd"};
  std::string output_hotspots{"hotspots.json"};
  std::string room_id{"simulation_room"};
  std::string camera_model;
  std::string map_frame{"map"};
  std::string owl_model{"data/owlvit-base-patch32"};
  std::string image_encoder{"data/resnet18_image_encoder.engine"};
  std::string mask_decoder{"data/mobile_sam_mask_decoder.engine"};
  std::string threshold{"0.1"};
  std::string mask_threshold{"0.0"};
  std::string merge_radius{"0.10"};
  std::string python{"python3"};
  std::string work_dir{"/tmp/samowl"};
  bool continuous{false};
};

void print_usage(const char * program)
{
  std::cerr
    << "Usage:\n"
    << "  File mode:\n"
    << "    " << program << " --image <image> --text <prompt> [options]\n\n"
    << "  Topic mode:\n"
    << "    " << program << " --rgb-topic <topic> --depth-topic <topic> --text <prompt> [options]\n\n"
    << "Options:\n"
    << "  --output-mask <path>        Mask image to save (default: mask.png)\n"
    << "  --output-boundary <path>    Boundary image to save (default: boundary.png)\n"
    << "  --output-depth-mask <path>  Depth image masked by SAM output (default: masked_depth.png)\n"
    << "  --output-points <path>      Masked 3D points in map frame (default: object_points_map.pcd)\n"
    << "  --output-hotspots <path>    Hotspot JSON to save (default: hotspots.json)\n"
    << "  --camera-info-topic <topic> RGB camera info topic (default: derived from RGB topic)\n"
    << "  --map-frame <frame>         Global frame for saved 3D points (default: map)\n"
    << "  --room-id <id>              Room id for hotspot JSON (default: simulation_room)\n"
    << "  --work-dir <path>           Temporary topic frame directory (default: /tmp/samowl)\n"
    << "  --continuous                Keep processing synchronized topic frames\n"
    << "  --owl-model <path>          Package-local OWL-ViT model directory\n"
    << "  --image-encoder <path>      SAM image encoder TensorRT engine\n"
    << "  --mask-decoder <path>       SAM mask decoder TensorRT engine\n"
    << "  --threshold <float>         OWL detection threshold (default: 0.1)\n"
    << "  --mask-threshold <float>    SAM logits threshold (default: 0.0)\n"
    << "  --merge-radius <meters>     Hotspot merge radius (default: 0.10)\n"
    << "  --python <path>             Python executable (default: python3)\n"
    << "  --help                      Show this help\n\n"
    << "Environment:\n"
    << "  SAMOWL_PIPELINE_SCRIPT can override the bundled Python model bridge.\n";
}

bool read_value(int & index, int argc, char ** argv, std::string & value)
{
  if (index + 1 >= argc) {
    std::cerr << "Missing value for " << argv[index] << "\n";
    return false;
  }
  value = argv[++index];
  return true;
}

std::string derive_camera_info_topic(const std::string & rgb_topic)
{
  const std::string suffix = "/image_raw";
  if (rgb_topic.size() >= suffix.size() &&
    rgb_topic.compare(rgb_topic.size() - suffix.size(), suffix.size(), suffix) == 0)
  {
    return rgb_topic.substr(0, rgb_topic.size() - suffix.size()) + "/camera_info";
  }
  return rgb_topic + "/camera_info";
}

bool parse_args(int argc, char ** argv, Options & options)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--image") {
      if (!read_value(i, argc, argv, options.image)) return false;
    } else if (arg == "--depth-image") {
      if (!read_value(i, argc, argv, options.depth_image)) return false;
    } else if (arg == "--rgb-topic") {
      if (!read_value(i, argc, argv, options.rgb_topic)) return false;
    } else if (arg == "--depth-topic") {
      if (!read_value(i, argc, argv, options.depth_topic)) return false;
    } else if (arg == "--camera-info-topic") {
      if (!read_value(i, argc, argv, options.camera_info_topic)) return false;
    } else if (arg == "--text" || arg == "--prompt") {
      if (!read_value(i, argc, argv, options.text)) return false;
    } else if (arg == "--output-mask") {
      if (!read_value(i, argc, argv, options.output_mask)) return false;
    } else if (arg == "--output-boundary") {
      if (!read_value(i, argc, argv, options.output_boundary)) return false;
    } else if (arg == "--output-depth-mask") {
      if (!read_value(i, argc, argv, options.output_depth_mask)) return false;
    } else if (arg == "--output-points") {
      if (!read_value(i, argc, argv, options.output_points)) return false;
    } else if (arg == "--output-hotspots") {
      if (!read_value(i, argc, argv, options.output_hotspots)) return false;
    } else if (arg == "--map-frame") {
      if (!read_value(i, argc, argv, options.map_frame)) return false;
    } else if (arg == "--room-id") {
      if (!read_value(i, argc, argv, options.room_id)) return false;
    } else if (arg == "--work-dir") {
      if (!read_value(i, argc, argv, options.work_dir)) return false;
    } else if (arg == "--continuous") {
      options.continuous = true;
    } else if (arg == "--owl-model") {
      if (!read_value(i, argc, argv, options.owl_model)) return false;
    } else if (arg == "--image-encoder") {
      if (!read_value(i, argc, argv, options.image_encoder)) return false;
    } else if (arg == "--mask-decoder") {
      if (!read_value(i, argc, argv, options.mask_decoder)) return false;
    } else if (arg == "--threshold") {
      if (!read_value(i, argc, argv, options.threshold)) return false;
    } else if (arg == "--mask-threshold") {
      if (!read_value(i, argc, argv, options.mask_threshold)) return false;
    } else if (arg == "--merge-radius") {
      if (!read_value(i, argc, argv, options.merge_radius)) return false;
    } else if (arg == "--python") {
      if (!read_value(i, argc, argv, options.python)) return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (options.text.empty()) {
    std::cerr << "--text is required\n";
    return false;
  }
  if (options.image.empty() && (options.rgb_topic.empty() || options.depth_topic.empty())) {
    std::cerr << "Provide either --image, or both --rgb-topic and --depth-topic\n";
    return false;
  }
  if (!options.image.empty() && (!options.rgb_topic.empty() || !options.depth_topic.empty())) {
    std::cerr << "Use either file mode or topic mode, not both\n";
    return false;
  }
  if (options.image.empty() && options.camera_info_topic.empty()) {
    options.camera_info_topic = derive_camera_info_topic(options.rgb_topic);
  }
  return true;
}

std::string find_script()
{
  if (const char * env_script = std::getenv("SAMOWL_PIPELINE_SCRIPT")) {
    if (fs::exists(env_script)) {
      return env_script;
    }
    std::cerr << "SAMOWL_PIPELINE_SCRIPT does not exist: " << env_script << "\n";
  }

  try {
    const fs::path installed_script =
      fs::path(ament_index_cpp::get_package_share_directory("samowl")) / "scripts" / "samowl_pipeline.py";
    if (fs::exists(installed_script)) {
      return installed_script.string();
    }
  } catch (const std::exception &) {
  }

  if (fs::exists(SAMOWL_SOURCE_SCRIPT_PATH)) {
    return SAMOWL_SOURCE_SCRIPT_PATH;
  }

  return "samowl_pipeline.py";
}

int run_python(const Options & options, const std::string & script)
{
  std::vector<std::string> args = {
    options.python,
    script,
    "--image", options.image,
    "--text", options.text,
    "--output-mask", options.output_mask,
    "--output-boundary", options.output_boundary,
    "--owl-model", options.owl_model,
    "--image-encoder", options.image_encoder,
    "--mask-decoder", options.mask_decoder,
    "--threshold", options.threshold,
    "--mask-threshold", options.mask_threshold
  };

  if (!options.depth_image.empty()) {
    args.push_back("--depth-image");
    args.push_back(options.depth_image);
    args.push_back("--output-depth-mask");
    args.push_back(options.output_depth_mask);
    args.push_back("--camera-model");
    args.push_back(options.camera_model);
    args.push_back("--output-points");
    args.push_back(options.output_points);
    args.push_back("--output-hotspots");
    args.push_back(options.output_hotspots);
    args.push_back("--room-id");
    args.push_back(options.room_id);
    args.push_back("--merge-radius");
    args.push_back(options.merge_radius);
  }

  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (auto & arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return EXIT_FAILURE;
  }

  if (pid == 0) {
    execvp(options.python.c_str(), argv.data());
    perror("execvp");
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    perror("waitpid");
    return EXIT_FAILURE;
  }

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    std::cerr << "Python bridge terminated by signal " << WTERMSIG(status) << "\n";
  }
  return EXIT_FAILURE;
}

void write_camera_model(
  const fs::path & path,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg,
  const geometry_msgs::msg::TransformStamped & transform,
  const Options & options)
{
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not write camera model: " + path.string());
  }

  out << "{\n";
  out << "  \"width\": " << info_msg->width << ",\n";
  out << "  \"height\": " << info_msg->height << ",\n";
  out << "  \"fx\": " << info_msg->k[0] << ",\n";
  out << "  \"fy\": " << info_msg->k[4] << ",\n";
  out << "  \"cx\": " << info_msg->k[2] << ",\n";
  out << "  \"cy\": " << info_msg->k[5] << ",\n";
  out << "  \"depth_scale\": 0.001,\n";
  out << "  \"source_frame\": \"" << transform.header.frame_id << "\",\n";
  out << "  \"camera_frame\": \"" << transform.child_frame_id << "\",\n";
  out << "  \"map_frame\": \"" << options.map_frame << "\",\n";
  out << "  \"translation\": ["
      << transform.transform.translation.x << ", "
      << transform.transform.translation.y << ", "
      << transform.transform.translation.z << "],\n";
  out << "  \"rotation_xyzw\": ["
      << transform.transform.rotation.x << ", "
      << transform.transform.rotation.y << ", "
      << transform.transform.rotation.z << ", "
      << transform.transform.rotation.w << "]\n";
  out << "}\n";
}

cv::Mat rgb_to_bgr(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  const auto cv_ptr = cv_bridge::toCvShare(msg);
  const cv::Mat & image = cv_ptr->image;
  cv::Mat bgr;

  if (msg->encoding == enc::BGR8) {
    bgr = image.clone();
  } else if (msg->encoding == enc::RGB8) {
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
  } else if (msg->encoding == enc::BGRA8) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  } else if (msg->encoding == enc::RGBA8) {
    cv::cvtColor(image, bgr, cv::COLOR_RGBA2BGR);
  } else if (msg->encoding == enc::MONO8) {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  } else {
    throw std::runtime_error("Unsupported RGB image encoding: " + msg->encoding);
  }

  return bgr;
}

cv::Mat depth_to_png_image(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  const auto cv_ptr = cv_bridge::toCvShare(msg);
  const cv::Mat & depth = cv_ptr->image;

  if (msg->encoding == enc::TYPE_16UC1 || msg->encoding == enc::MONO16) {
    return depth.clone();
  }

  if (msg->encoding == enc::TYPE_32FC1) {
    cv::Mat depth_mm(depth.rows, depth.cols, CV_16UC1, cv::Scalar(0));
    for (int y = 0; y < depth.rows; ++y) {
      const float * src = depth.ptr<float>(y);
      uint16_t * dst = depth_mm.ptr<uint16_t>(y);
      for (int x = 0; x < depth.cols; ++x) {
        const float meters = src[x];
        if (std::isfinite(meters) && meters > 0.0f) {
          const float mm = std::min(meters * 1000.0f, 65535.0f);
          dst[x] = static_cast<uint16_t>(mm);
        }
      }
    }
    return depth_mm;
  }

  throw std::runtime_error("Unsupported depth image encoding: " + msg->encoding);
}

class TopicRunner : public rclcpp::Node
{
public:
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image, CameraInfo>;

  TopicRunner(const Options & options, std::string script)
  : Node("samowl_topic_runner"),
    options_(options),
    script_(std::move(script)),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    fs::create_directories(options_.work_dir);
    rgb_sub_.subscribe(this, options_.rgb_topic);
    depth_sub_.subscribe(this, options_.depth_topic);
    camera_info_sub_.subscribe(this, options_.camera_info_topic);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), rgb_sub_, depth_sub_, camera_info_sub_);
    sync_->registerCallback(&TopicRunner::callback, this);

    RCLCPP_INFO(get_logger(), "Waiting for RGB '%s', depth '%s', camera info '%s', TF to '%s'",
      options_.rgb_topic.c_str(),
      options_.depth_topic.c_str(),
      options_.camera_info_topic.c_str(),
      options_.map_frame.c_str());
  }

  int last_status() const
  {
    return last_status_;
  }

private:
  void callback(
    const Image::ConstSharedPtr & rgb_msg,
    const Image::ConstSharedPtr & depth_msg,
    const CameraInfo::ConstSharedPtr & info_msg)
  {
    if (processing_.exchange(true)) {
      return;
    }

    try {
      const auto stamp = std::to_string(rgb_msg->header.stamp.sec) + "_" +
        std::to_string(rgb_msg->header.stamp.nanosec);
      const fs::path rgb_path = fs::path(options_.work_dir) / ("rgb_" + stamp + ".png");
      const fs::path depth_path = fs::path(options_.work_dir) / ("depth_" + stamp + ".png");
      const fs::path camera_model_path = fs::path(options_.work_dir) / ("camera_model_" + stamp + ".json");

      const auto transform = tf_buffer_.lookupTransform(
        options_.map_frame,
        rgb_msg->header.frame_id,
        rgb_msg->header.stamp,
        tf2::durationFromSec(0.25));

      cv::imwrite(rgb_path.string(), rgb_to_bgr(rgb_msg));
      cv::imwrite(depth_path.string(), depth_to_png_image(depth_msg));
      write_camera_model(camera_model_path, info_msg, transform, options_);

      Options run_options = options_;
      run_options.image = rgb_path.string();
      run_options.depth_image = depth_path.string();
      run_options.camera_model = camera_model_path.string();

      RCLCPP_INFO(get_logger(), "Running OWL + SAM on synchronized RGB/depth frames");
      last_status_ = run_python(run_options, script_);
      if (last_status_ != EXIT_SUCCESS) {
        RCLCPP_ERROR(get_logger(), "OWL + SAM pipeline failed with status %d", last_status_);
      } else {
        RCLCPP_INFO(get_logger(), "Saved mask to '%s'", options_.output_mask.c_str());
      }
    } catch (const std::exception & error) {
      last_status_ = EXIT_FAILURE;
      RCLCPP_ERROR(get_logger(), "%s", error.what());
    }

    processing_ = false;
    if (!options_.continuous) {
      rclcpp::shutdown();
    }
  }

  Options options_;
  std::string script_;
  message_filters::Subscriber<Image> rgb_sub_;
  message_filters::Subscriber<Image> depth_sub_;
  message_filters::Subscriber<CameraInfo> camera_info_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::atomic_bool processing_{false};
  int last_status_{EXIT_SUCCESS};
};
}  // namespace

int main(int argc, char ** argv)
{
  Options options;
  if (!parse_args(argc, argv, options)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string script = find_script();
  if (!options.image.empty()) {
    if (!fs::exists(options.image)) {
      std::cerr << "Input image does not exist: " << options.image << "\n";
      return EXIT_FAILURE;
    }
    std::cout << "Running OWL + SAM pipeline for prompt: " << options.text << "\n";
    return run_python(options, script);
  }

  int ros_argc = 0;
  char ** ros_argv = nullptr;
  rclcpp::init(ros_argc, ros_argv);
  auto node = std::make_shared<TopicRunner>(options, script);
  rclcpp::spin(node);
  const int status = node->last_status();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return status;
}
