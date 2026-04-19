#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>
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
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
  std::string config;
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
  std::string daemon_socket{"/tmp/samowl/daemon.sock"};
  double depth_scale{0.001};
  int daemon_startup_timeout_ms{30000};
  int daemon_poll_interval_ms{100};
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
    << "  --config <path>             YAML config file (default: share/samowl/config/samowl.yaml)\n"
    << "  --output-mask <path>        Mask image to save (default: mask.png)\n"
    << "  --output-boundary <path>    Boundary image to save (default: boundary.png)\n"
    << "  --output-depth-mask <path>  Depth image masked by SAM output (default: masked_depth.png)\n"
    << "  --output-points <path>      Masked 3D points in map frame (default: object_points_map.pcd)\n"
    << "  --output-hotspots <path>    Hotspot JSON to save (default: hotspots.json)\n"
    << "  --camera-info-topic <topic> RGB camera info topic (default: derived from RGB topic)\n"
    << "  --map-frame <frame>         Global frame for saved 3D points (default: map)\n"
    << "  --room-id <id>              Room id for hotspot JSON (default: simulation_room)\n"
    << "  --work-dir <path>           Temporary topic frame directory (default: /tmp/samowl)\n"
    << "  --daemon-socket <path>      Unix socket for persistent Python daemon (default: /tmp/samowl/daemon.sock)\n"
    << "  --depth-scale <float>       Depth pixel to metres scale factor (default: 0.001)\n"
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
    << "  SAMOWL_PIPELINE_SCRIPT can override the bundled Python model bridge.\n"
    << "  SAMOWL_DAEMON_SCRIPT can override the bundled Python daemon script.\n";
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
    } else if (arg == "--config") {
      if (!read_value(i, argc, argv, options.config)) return false;
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
    } else if (arg == "--daemon-socket") {
      if (!read_value(i, argc, argv, options.daemon_socket)) return false;
    } else if (arg == "--depth-scale") {
      std::string val;
      if (!read_value(i, argc, argv, val)) return false;
      try {
        options.depth_scale = std::stod(val);
      } catch (...) {
        std::cerr << "--depth-scale must be a floating-point number\n";
        return false;
      }
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

// ---------------------------------------------------------------------------
// Script discovery
// ---------------------------------------------------------------------------
std::string find_script(const char * env_var, const char * filename, const char * source_path)
{
  if (const char * env_script = std::getenv(env_var)) {
    if (fs::exists(env_script)) {
      return env_script;
    }
    std::cerr << env_var << " does not exist: " << env_script << "\n";
  }

  try {
    const fs::path installed =
      fs::path(ament_index_cpp::get_package_share_directory("samowl")) / "scripts" / filename;
    if (fs::exists(installed)) {
      return installed.string();
    }
  } catch (const std::exception &) {
  }

  if (fs::exists(source_path)) {
    return source_path;
  }

  return filename;
}

// ---------------------------------------------------------------------------
// YAML config loading
// ---------------------------------------------------------------------------

// Scan argv for --config before full argument parsing (avoids chicken-and-egg).
std::string find_config(int argc, char ** argv)
{
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == "--config") {
      return argv[i + 1];
    }
  }
  try {
    const fs::path installed =
      fs::path(ament_index_cpp::get_package_share_directory("samowl")) / "config" / "samowl.yaml";
    if (fs::exists(installed)) {
      return installed.string();
    }
  } catch (const std::exception &) {}
  if (fs::exists(SAMOWL_SOURCE_CONFIG_PATH)) {
    return SAMOWL_SOURCE_CONFIG_PATH;
  }
  return "";
}

void load_config(Options & opts, const std::string & path)
{
  if (path.empty() || !fs::exists(path)) {
    return;
  }
  try {
    const YAML::Node cfg = YAML::LoadFile(path);

    auto load_str = [](const YAML::Node & node, const char * key, std::string & target) {
      if (node[key]) {
        target = node[key].as<std::string>();
      }
    };

    if (cfg["topics"]) {
      const auto & t = cfg["topics"];
      load_str(t, "rgb", opts.rgb_topic);
      load_str(t, "depth", opts.depth_topic);
      load_str(t, "camera_info", opts.camera_info_topic);
    }
    if (cfg["models"]) {
      const auto & m = cfg["models"];
      load_str(m, "owl", opts.owl_model);
      load_str(m, "image_encoder", opts.image_encoder);
      load_str(m, "mask_decoder", opts.mask_decoder);
    }
    if (cfg["detection"]) {
      const auto & d = cfg["detection"];
      load_str(d, "threshold", opts.threshold);
      load_str(d, "mask_threshold", opts.mask_threshold);
      load_str(d, "merge_radius", opts.merge_radius);
    }
    if (cfg["system"]) {
      const auto & s = cfg["system"];
      load_str(s, "python", opts.python);
      load_str(s, "work_dir", opts.work_dir);
      load_str(s, "map_frame", opts.map_frame);
      load_str(s, "room_id", opts.room_id);
    }
    if (cfg["daemon"]) {
      const auto & dm = cfg["daemon"];
      load_str(dm, "socket", opts.daemon_socket);
      if (dm["startup_timeout_ms"]) {
        opts.daemon_startup_timeout_ms = dm["startup_timeout_ms"].as<int>();
      }
      if (dm["poll_interval_ms"]) {
        opts.daemon_poll_interval_ms = dm["poll_interval_ms"].as<int>();
      }
    }
    if (cfg["outputs"]) {
      const auto & o = cfg["outputs"];
      load_str(o, "mask", opts.output_mask);
      load_str(o, "boundary", opts.output_boundary);
      load_str(o, "depth_mask", opts.output_depth_mask);
      load_str(o, "points", opts.output_points);
      load_str(o, "hotspots", opts.output_hotspots);
    }
    if (cfg["depth"]) {
      const auto & dep = cfg["depth"];
      if (dep["scale"]) {
        opts.depth_scale = dep["scale"].as<double>();
      }
    }
  } catch (const YAML::Exception & e) {
    std::cerr << "Warning: could not parse config " << path << ": " << e.what() << "\n";
  }
}

// ---------------------------------------------------------------------------
// Daemon lifecycle — fork once at startup; wait for the socket to appear.
// ---------------------------------------------------------------------------

// Global daemon PID so we can reap it on exit.
static pid_t g_daemon_pid = -1;

// Returns 0 if the daemon was started (or was already running), non-zero on error.
int start_daemon(const Options & options, const std::string & daemon_script)
{
  // If the socket already exists a daemon is already running — nothing to do.
  if (fs::exists(options.daemon_socket)) {
    std::cerr << "Daemon socket already present at " << options.daemon_socket
              << " — reusing existing daemon\n";
    return 0;
  }

  // Ensure the work directory exists so the socket path is valid.
  fs::create_directories(options.work_dir);

  std::vector<std::string> args = {
    options.python,
    daemon_script,
    "--socket",        options.daemon_socket,
    "--owl-model",     options.owl_model,
    "--image-encoder", options.image_encoder,
    "--mask-decoder",  options.mask_decoder,
    "--threshold",     options.threshold,
  };

  std::vector<char *> argv_ptrs;
  argv_ptrs.reserve(args.size() + 1);
  for (auto & a : args) {
    argv_ptrs.push_back(a.data());
  }
  argv_ptrs.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    perror("start_daemon: fork");
    return EXIT_FAILURE;
  }

  if (pid == 0) {
    // Child: become the daemon.
    execvp(options.python.c_str(), argv_ptrs.data());
    perror("start_daemon: execvp");
    _exit(127);
  }

  g_daemon_pid = pid;
  std::cerr << "Daemon PID " << pid << " — waiting for socket " << options.daemon_socket << "\n";

  // Poll until the socket file appears or the timeout expires.
  const int max_polls = options.daemon_startup_timeout_ms / options.daemon_poll_interval_ms;
  for (int attempt = 0; attempt < max_polls; ++attempt) {
    struct timespec ts{0, options.daemon_poll_interval_ms * 1000000L};
    nanosleep(&ts, nullptr);
    if (fs::exists(options.daemon_socket)) {
      std::cerr << "Daemon socket ready after " << (attempt + 1) * DAEMON_SOCKET_POLL_MS << " ms\n";
      return 0;
    }
    // If the child already exited, fail fast.
    int wstatus = 0;
    const pid_t ret = waitpid(pid, &wstatus, WNOHANG);
    if (ret == pid) {
      std::cerr << "Daemon process exited unexpectedly (status " << WEXITSTATUS(wstatus) << ")\n";
      g_daemon_pid = -1;
      return EXIT_FAILURE;
    }
  }

  std::cerr << "Timed out waiting for daemon socket at " << options.daemon_socket << "\n";
  return EXIT_FAILURE;
}

// ---------------------------------------------------------------------------
// Socket-based inference client
// ---------------------------------------------------------------------------

// Build a JSON request object from options (no trailing newline).
static std::string build_request_json(const Options & options)
{
  // Very small hand-rolled JSON — avoids a heavy JSON library dependency.
  // All values are either numbers or strings with no special characters that
  // would require escaping beyond what we produce here.
  auto esc = [](const std::string & s) -> std::string {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (const char c : s) {
      if (c == '"')  { out += "\\\""; }
      else if (c == '\\') { out += "\\\\"; }
      else { out += c; }
    }
    out += '"';
    return out;
  };

  std::string j = "{";
  j += "\"image_path\":"        + esc(options.image)            + ",";
  j += "\"depth_image_path\":"  + esc(options.depth_image)      + ",";
  j += "\"camera_model_path\":" + esc(options.camera_model)     + ",";
  j += "\"text\":"              + esc(options.text)             + ",";
  j += "\"threshold\":"         + options.threshold             + ",";
  j += "\"mask_threshold\":"    + options.mask_threshold        + ",";
  j += "\"output_mask\":"       + esc(options.output_mask)      + ",";
  j += "\"output_boundary\":"   + esc(options.output_boundary)  + ",";
  j += "\"output_depth_mask\":" + esc(options.output_depth_mask)+ ",";
  j += "\"output_points\":"     + esc(options.output_points)    + ",";
  j += "\"output_hotspots\":"   + esc(options.output_hotspots)  + ",";
  j += "\"room_id\":"           + esc(options.room_id)          + ",";
  j += "\"merge_radius\":"      + options.merge_radius;
  j += "}";
  return j;
}

// Returns 0 on success, non-zero on error.
// On success, response_out receives the raw JSON response string.
static int socket_call(
  const std::string & socket_path,
  const std::string & request_json,
  std::string & response_out)
{
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    std::cerr << "socket_call: socket(): " << std::strerror(errno) << "\n";
    return EXIT_FAILURE;
  }

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(addr.sun_path)) {
    std::cerr << "socket_call: socket path too long\n";
    close(fd);
    return EXIT_FAILURE;
  }
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
    std::cerr << "socket_call: connect(" << socket_path << "): " << std::strerror(errno) << "\n";
    close(fd);
    return EXIT_FAILURE;
  }

  // Send request terminated by a newline.
  const std::string message = request_json + "\n";
  std::size_t sent = 0;
  while (sent < message.size()) {
    const ssize_t n = write(fd, message.data() + sent, message.size() - sent);
    if (n < 0) {
      std::cerr << "socket_call: write(): " << std::strerror(errno) << "\n";
      close(fd);
      return EXIT_FAILURE;
    }
    sent += static_cast<std::size_t>(n);
  }

  // Read response until newline or EOF.
  std::string response;
  response.reserve(4096);
  char buf[4096];
  bool got_newline = false;
  while (!got_newline) {
    const ssize_t n = read(fd, buf, sizeof(buf));
    if (n < 0) {
      std::cerr << "socket_call: read(): " << std::strerror(errno) << "\n";
      close(fd);
      return EXIT_FAILURE;
    }
    if (n == 0) {
      break;  // EOF — treat whatever we got as the full response.
    }
    response.append(buf, static_cast<std::size_t>(n));
    if (response.find('\n') != std::string::npos) {
      got_newline = true;
    }
  }
  close(fd);

  // Strip trailing newline before returning.
  while (!response.empty() && (response.back() == '\n' || response.back() == '\r')) {
    response.pop_back();
  }
  response_out = std::move(response);
  return 0;
}

// Returns 0 if the daemon reports success, non-zero otherwise.
int run_python(const Options & options, const std::string & /*script_unused*/)
{
  const std::string request = build_request_json(options);
  std::string response;

  const int rc = socket_call(options.daemon_socket, request, response);
  if (rc != 0) {
    std::cerr << "run_python: failed to contact daemon at " << options.daemon_socket << "\n";
    return EXIT_FAILURE;
  }

  // Minimal success check: look for "\"success\": true" in the response.
  // We print the response so downstream tooling can parse it if needed.
  std::cout << response << "\n";

  if (response.find("\"success\": true") == std::string::npos &&
      response.find("\"success\":true") == std::string::npos)
  {
    // Try to extract the error message for a readable log line.
    const auto err_pos = response.find("\"error\":");
    if (err_pos != std::string::npos) {
      std::cerr << "run_python: daemon reported failure: "
                << response.substr(err_pos, 200) << "\n";
    } else {
      std::cerr << "run_python: daemon reported failure (no error field in response)\n";
    }
    return EXIT_FAILURE;
  }

  return 0;
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
  out << "  \"depth_scale\": " << options.depth_scale << ",\n";
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
        // Clean up per-frame scratch files so /tmp/samowl does not grow
        // indefinitely.  Errors are logged but do not affect last_status_.
        for (const auto & p : {rgb_path, depth_path, camera_model_path}) {
          std::error_code ec;
          fs::remove(p, ec);
          if (ec) {
            RCLCPP_WARN(get_logger(), "Could not remove temp file %s: %s",
              p.string().c_str(), ec.message().c_str());
          }
        }
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

  // Precedence: CLI args > YAML config > hardcoded struct defaults.
  const std::string config_path = find_config(argc, argv);
  load_config(options, config_path);

  if (!parse_args(argc, argv, options)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  // Find the daemon script (used to launch the persistent Python process).
  const std::string daemon_script = find_script(
    "SAMOWL_DAEMON_SCRIPT",
    "samowl_daemon.py",
    SAMOWL_SOURCE_DAEMON_PATH
  );

  // Start the daemon (loads models once; all subsequent calls go via socket).
  if (start_daemon(options, daemon_script) != 0) {
    std::cerr << "Failed to start samowl daemon\n";
    return EXIT_FAILURE;
  }

  if (!options.image.empty()) {
    if (!fs::exists(options.image)) {
      std::cerr << "Input image does not exist: " << options.image << "\n";
      return EXIT_FAILURE;
    }
    std::cout << "Running OWL + SAM pipeline for prompt: " << options.text << "\n";
    return run_python(options, "");
  }

  int ros_argc = 0;
  char ** ros_argv = nullptr;
  rclcpp::init(ros_argc, ros_argv);
  auto node = std::make_shared<TopicRunner>(options, "");
  rclcpp::spin(node);
  const int status = node->last_status();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return status;
}
