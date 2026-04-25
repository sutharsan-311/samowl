#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nlohmann/json.hpp>
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
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
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
  std::string owl_encoder{"data/owl_image_encoder_patch32.engine"};
  std::string image_encoder{"data/mobile_sam_image_encoder.engine"};
  std::string mask_decoder{"data/mobile_sam_mask_decoder.engine"};
  std::string threshold{"0.1"};
  std::string mask_threshold{"0.0"};
  std::string merge_radius{"0.10"};
  std::string python{"python3"};
  std::string work_dir{"/tmp/samowl"};
  std::string daemon_socket{"/tmp/samowl/daemon.sock"};
  std::string mode{"live"};           // "live" or "scan"
  std::string output_dir{"/output"};  // scan mode: directory for final JSON output
  double depth_scale{0.001};
  double near_threshold{1.0};
  double depth_min{0.1};
  double depth_max{10.0};
  int daemon_startup_timeout_ms{30000};
  int daemon_poll_interval_ms{100};
  int scan_idle_timeout_ms{5000};  // scan mode: ms of silence → finalize
  bool continuous{false};
  bool debug{false};
};

struct DetectionEntry
{
  std::string label;
  float score{0.0f};
  float cx{0.0f}, cy{0.0f}, cz{0.0f};
  std::string output_points;  // per-label PCD path; empty if no 3D projection
};

struct ParsedResult
{
  bool success{false};
  bool valid{false};
  std::string error;
  std::vector<float> bbox;       // legacy single-detection: [x1,y1,x2,y2]
  float score{0.0f};
  int points_count{0};
  std::string output_points;     // legacy: primary PCD path
  std::string output_hotspots;
  std::string hotspot_json;
  std::vector<DetectionEntry> detections;  // all per-label results (new format)
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
    << "  --mode <live|scan>          Operating mode: 'live' streams topics; 'scan' processes every frame and saves JSON (default: live)\n"
    << "  --output-dir <path>         Scan mode: directory for scene_graph.json and hotspots.json (default: /output)\n"
    << "  --scan-idle-timeout <sec>   Scan mode: seconds of inactivity before finalization (default: 5)\n"
    << "  --continuous                Keep processing synchronized topic frames\n"
    << "  --owl-model <path>          Package-local OWL-ViT model directory\n"
    << "  --image-encoder <path>      SAM image encoder TensorRT engine\n"
    << "  --mask-decoder <path>       SAM mask decoder TensorRT engine\n"
    << "  --threshold <float>         OWL detection threshold (default: 0.1)\n"
    << "  --mask-threshold <float>    SAM logits threshold (default: 0.0)\n"
    << "  --merge-radius <meters>     Hotspot merge radius (default: 0.10)\n"
    << "  --python <path>             Python executable (default: python3)\n"
    << "  --debug                     Enable verbose debug logging\n"
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
    } else if (arg == "--debug") {
      options.debug = true;
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
    } else if (arg == "--mode") {
      if (!read_value(i, argc, argv, options.mode)) return false;
    } else if (arg == "--output-dir") {
      if (!read_value(i, argc, argv, options.output_dir)) return false;
    } else if (arg == "--scan-idle-timeout") {
      std::string val;
      if (!read_value(i, argc, argv, val)) return false;
      try {
        options.scan_idle_timeout_ms = std::stoi(val) * 1000;
      } catch (...) {
        std::cerr << "--scan-idle-timeout must be an integer number of seconds\n";
        return false;
      }
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
  if (options.mode != "live" && options.mode != "scan") {
    std::cerr << "--mode must be 'live' or 'scan'\n";
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
      load_str(m, "owl_encoder", opts.owl_encoder);
      load_str(m, "image_encoder", opts.image_encoder);
      load_str(m, "mask_decoder", opts.mask_decoder);
    }
    if (cfg["detection"]) {
      const auto & d = cfg["detection"];
      load_str(d, "text", opts.text);
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
      load_str(s, "mode", opts.mode);
      load_str(s, "output_dir", opts.output_dir);
      if (s["debug"]) {
        opts.debug = s["debug"].as<bool>();
      }
      if (s["scan_idle_timeout_ms"]) {
        opts.scan_idle_timeout_ms = s["scan_idle_timeout_ms"].as<int>();
      }
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
      if (dep["min"]) { opts.depth_min = dep["min"].as<double>(); }
      if (dep["max"]) { opts.depth_max = dep["max"].as<double>(); }
    }
    if (cfg["graph"]) {
      const auto & g = cfg["graph"];
      if (g["near_threshold"])  { opts.near_threshold  = g["near_threshold"].as<double>(); }
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
  // If the socket file exists, probe whether the daemon is actually alive.
  if (fs::exists(options.daemon_socket)) {
    const int probe_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (probe_fd >= 0) {
      struct sockaddr_un probe_addr{};
      probe_addr.sun_family = AF_UNIX;
      std::strncpy(probe_addr.sun_path, options.daemon_socket.c_str(),
                   sizeof(probe_addr.sun_path) - 1);
      const bool alive = (::connect(probe_fd,
        reinterpret_cast<struct sockaddr *>(&probe_addr),
        sizeof(probe_addr)) == 0);
      ::close(probe_fd);
      if (alive) {
        std::cerr << "Daemon socket alive at " << options.daemon_socket
                  << " — reusing existing daemon\n";
        return 0;
      }
      std::cerr << "Stale daemon socket at " << options.daemon_socket
                << " — removing and restarting\n";
    }
    std::error_code ec;
    fs::remove(options.daemon_socket, ec);
  }

  // Ensure the work directory exists so the socket path is valid.
  fs::create_directories(options.work_dir);

  std::vector<std::string> args = {
    options.python,
    daemon_script,
    "--socket",        options.daemon_socket,
    "--owl-model",     options.owl_model,
    "--owl-encoder",   options.owl_encoder,
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
      std::cerr << "Daemon socket ready after " << (attempt + 1) * options.daemon_poll_interval_ms << " ms\n";
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
  using json = nlohmann::json;
  json j;
  j["image_path"]        = options.image;
  j["depth_image_path"]  = options.depth_image;
  j["camera_model_path"] = options.camera_model;
  j["text"]              = options.text;
  j["threshold"]         = std::stod(options.threshold);
  j["mask_threshold"]    = std::stod(options.mask_threshold);
  j["output_mask"]       = options.output_mask;
  j["output_boundary"]   = options.output_boundary;
  j["output_depth_mask"] = options.output_depth_mask;
  j["output_points"]     = options.output_points;
  j["output_hotspots"]   = options.output_hotspots;
  j["room_id"]           = options.room_id;
  j["merge_radius"]      = std::stod(options.merge_radius);
  return j.dump();
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

  // 120s receive timeout — prevents indefinite hang if the GPU stalls mid-inference.
  struct timeval tv{};
  tv.tv_sec = 120;
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
    std::cerr << "socket_call: setsockopt(SO_RCVTIMEO): " << std::strerror(errno) << "\n";
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

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

// Parses a JSON response from the daemon into a ParsedResult.
// Returns true on success, false on any parse or daemon-side error.
// out.error is set on failure; all other fields default to zero/empty.
bool parse_response(const std::string & s, ParsedResult & out)
{
  using json = nlohmann::json;
  try {
    const json j = json::parse(s);
    out.success = j.value("success", false);
    if (!out.success) {
      out.error = j.value("error", std::string("daemon reported failure"));
      return false;
    }

    // bbox: validate shape and each element individually; reject partial results.
    if (j.contains("bbox") && j["bbox"].is_array() && j["bbox"].size() == 4) {
      out.bbox.reserve(4);
      for (const auto & v : j["bbox"]) {
        if (v.is_number()) {
          out.bbox.push_back(v.get<float>());
        }
      }
      if (out.bbox.size() != 4) {
        out.bbox.clear();
      }
    }

    // Explicit type checks to prevent silent coercion into downstream ROS fields.
    if (j.contains("score") && j["score"].is_number()) {
      out.score = j["score"].get<float>();
    }
    if (j.contains("points_count") && j["points_count"].is_number_integer()) {
      out.points_count = j["points_count"].get<int>();
    }

    // Note: daemon writes 3D points to a PCD file; path is returned here.
    // Step 4.2 will read output_points to build a PointCloud2 message.
    if (j.contains("output_points") && j["output_points"].is_string()) {
      out.output_points = j["output_points"].get<std::string>();
    }
    if (j.contains("output_hotspots") && j["output_hotspots"].is_string()) {
      out.output_hotspots = j["output_hotspots"].get<std::string>();
    }
    if (j.contains("hotspot_map") && j["hotspot_map"].is_object()) {
      out.hotspot_json = j["hotspot_map"].dump();
    }

    if (j.contains("detections") && j["detections"].is_array()) {
      for (const auto & d : j["detections"]) {
        DetectionEntry e;
        if (d.contains("label") && d["label"].is_string()) {
          e.label = d["label"].get<std::string>();
        }
        if (d.contains("score") && d["score"].is_number()) {
          e.score = d["score"].get<float>();
        }
        if (d.contains("centroid") && d["centroid"].is_array() && d["centroid"].size() == 3) {
          e.cx = d["centroid"][0].get<float>();
          e.cy = d["centroid"][1].get<float>();
          e.cz = d["centroid"][2].get<float>();
        }
        if (d.contains("output_points") && d["output_points"].is_string()) {
          e.output_points = d["output_points"].get<std::string>();
        }
        out.detections.push_back(e);
      }
    }

    // valid = success + at least one usable detection field present.
    const bool has_bbox = out.bbox.size() == 4;
    const bool has_points_file = !out.output_points.empty();
    const bool has_detections = !out.detections.empty();
    out.valid = has_bbox || has_points_file || has_detections;

    return true;
  } catch (const json::exception & e) {
    out.error = std::string("JSON parse error: ") + e.what();
    return false;
  }
}

// Returns 0 if the daemon reports success, non-zero otherwise.
// If result_out is non-null it is populated with the parsed response on success.
int run_python(
  const Options & options,
  const std::string & daemon_script,
  ParsedResult * result_out = nullptr)
{
  const std::string request = build_request_json(options);
  std::string response;

  int rc = socket_call(options.daemon_socket, request, response);
  if (rc != 0 && g_daemon_pid > 0 && !daemon_script.empty()) {
    // Check whether the daemon process has exited; if so, restart it once.
    int wstatus = 0;
    if (::waitpid(g_daemon_pid, &wstatus, WNOHANG) == g_daemon_pid) {
      g_daemon_pid = -1;
      std::cerr << "run_python: daemon exited (code " << WEXITSTATUS(wstatus)
                << ") — attempting restart\n";
      if (start_daemon(options, daemon_script) == 0) {
        rc = socket_call(options.daemon_socket, request, response);
      }
    }
  }
  if (rc != 0) {
    std::cerr << "run_python: failed to contact daemon at " << options.daemon_socket << "\n";
    return EXIT_FAILURE;
  }

  // TODO(4.4): Remove raw stdout once all fields are published to ROS topics.
  std::cout << response << "\n";

  ParsedResult result;
  if (!parse_response(response, result)) {
    std::cerr << "run_python: " << result.error << "\n";
    return EXIT_FAILURE;
  }

  if (!result.valid) {
    std::cerr << "run_python: daemon response incomplete — no bbox or points file\n";
  }

  if (options.debug) {
    std::cerr << "run_python: score=" << result.score << " bbox=[";
    if (result.bbox.size() == 4) {
      std::cerr << result.bbox[0] << "," << result.bbox[1] << ","
                << result.bbox[2] << "," << result.bbox[3];
    }
    std::cerr << "] points=" << result.points_count << "\n";

    if (!result.hotspot_json.empty()) {
      const std::string preview = result.hotspot_json.size() > 200
        ? result.hotspot_json.substr(0, 200) + "..."
        : result.hotspot_json;
      std::cerr << "run_python: hotspot_map=" << preview << "\n";
    }
  }

  if (result_out) {
    *result_out = result;
  }
  return 0;
}

void write_camera_model(
  const fs::path & path,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg,
  const geometry_msgs::msg::TransformStamped & transform,
  const Options & options)
{
  using json = nlohmann::json;
  const auto & t = transform.transform;
  json j;
  j["width"]       = info_msg->width;
  j["height"]      = info_msg->height;
  j["fx"]          = info_msg->k[0];
  j["fy"]          = info_msg->k[4];
  j["cx"]          = info_msg->k[2];
  j["cy"]          = info_msg->k[5];
  j["depth_scale"] = options.depth_scale;
  j["depth_min"]   = options.depth_min;
  j["depth_max"]   = options.depth_max;
  j["source_frame"]  = transform.header.frame_id;
  j["camera_frame"]  = transform.child_frame_id;
  j["map_frame"]     = options.map_frame;
  j["translation"]   = {t.translation.x, t.translation.y, t.translation.z};
  j["rotation_xyzw"] = {t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not write camera model: " + path.string());
  }
  out << j.dump(2) << "\n";
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

struct DetectionRecord
{
  std::string label;
  float score;
  float cx, cy, cz;
  double stamp;
};

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
    rgb_sub_.subscribe(this, options_.rgb_topic, rmw_qos_profile_sensor_data);
    depth_sub_.subscribe(this, options_.depth_topic, rmw_qos_profile_sensor_data);
    camera_info_sub_.subscribe(this, options_.camera_info_topic, rmw_qos_profile_sensor_data);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), rgb_sub_, depth_sub_, camera_info_sub_);
    sync_->registerCallback(&TopicRunner::callback, this);

    rclcpp::QoS sensor_qos{1};
    sensor_qos.best_effort();
    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/samowl/points", sensor_qos);
    objects_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/samowl/objects", sensor_qos);
    detections_pub_ = create_publisher<std_msgs::msg::String>("/samowl/detections", rclcpp::QoS(10));

    RCLCPP_INFO(get_logger(), "Waiting for RGB '%s', depth '%s', camera info '%s', TF to '%s'",
      options_.rgb_topic.c_str(),
      options_.depth_topic.c_str(),
      options_.camera_info_topic.c_str(),
      options_.map_frame.c_str());
    RCLCPP_INFO(get_logger(),
      "Bag replay: run with '--ros-args -p use_sim_time:=true' and 'ros2 bag play --clock' "
      "so TF timestamps match image stamps");

    if (options_.mode == "scan") {
      startup_watchdog_ = create_wall_timer(
        std::chrono::seconds(30),
        [this]() {
          startup_watchdog_->cancel();
          if (total_frame_count_ == 0) {
            RCLCPP_FATAL(get_logger(),
              "[scan] No frames received in 30 s — check topic names and bag playback. "
              "Expected RGB='%s' depth='%s'",
              options_.rgb_topic.c_str(), options_.depth_topic.c_str());
            // Detach shutdown to avoid deadlock: calling rclcpp::shutdown() directly
            // from a timer callback while the single-threaded executor is spinning
            // can deadlock on the executor's internal mutex in ROS2 Humble.
            std::thread([]() { rclcpp::shutdown(); }).detach();
          }
        });
      periodic_flush_timer_ = create_wall_timer(
        std::chrono::seconds(10),
        [this]() { save_scan_outputs(false); });
    }
  }

  int last_status() const { return last_status_; }

  // Called from main() after rclcpp::spin() returns so SIGINT saves scan outputs.
  void finalize()
  {
    if (finalized_.exchange(true)) {
      return;
    }
    if (idle_timer_) {
      idle_timer_->cancel();
    }
    if (periodic_flush_timer_) {
      periodic_flush_timer_->cancel();
    }
    RCLCPP_INFO(get_logger(), "[scan] Finalizing after shutdown — %d frames, %zu detections",
      frame_count_, scan_detections_.size());
    save_scan_outputs(true);
  }

private:
  void callback(
    const Image::ConstSharedPtr & rgb_msg,
    const Image::ConstSharedPtr & depth_msg,
    const CameraInfo::ConstSharedPtr & info_msg)
  {
    // In scan mode every frame must be processed; the single-threaded executor
    // serializes callbacks naturally so no frame-drop guard is needed.
    if (options_.mode == "live" && processing_.exchange(true)) {
      return;
    }
    if (options_.mode == "scan") {
      ++total_frame_count_;
    }

    try {
      const auto stamp = std::to_string(rgb_msg->header.stamp.sec) + "_" +
        std::to_string(rgb_msg->header.stamp.nanosec);
      const fs::path rgb_path = fs::path(options_.work_dir) / ("rgb_" + stamp + ".png");
      const fs::path depth_path = fs::path(options_.work_dir) / ("depth_" + stamp + ".png");
      const fs::path camera_model_path = fs::path(options_.work_dir) / ("camera_model_" + stamp + ".json");

      geometry_msgs::msg::TransformStamped transform;
      try {
        transform = tf_buffer_.lookupTransform(
          options_.map_frame,
          rgb_msg->header.frame_id,
          rgb_msg->header.stamp,
          tf2::durationFromSec(0.25));
      } catch (const tf2::ExtrapolationException &) {
        // Bag playback timing skew: fall back to latest available transform.
        // NOTE: accurate bag replay requires --ros-args -p use_sim_time:=true
        // and ros2 bag play --clock so the TF buffer uses bag timestamps.
        transform = tf_buffer_.lookupTransform(
          options_.map_frame,
          rgb_msg->header.frame_id,
          tf2::TimePointZero);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(get_logger(),
          "TF lookup failed (%s → %s): %s — skipping frame",
          rgb_msg->header.frame_id.c_str(), options_.map_frame.c_str(), ex.what());
        processing_ = false;
        if (options_.mode == "scan") {
          reset_idle_timer();
        }
        return;
      }

      cv::imwrite(rgb_path.string(), rgb_to_bgr(rgb_msg));
      cv::imwrite(depth_path.string(), depth_to_png_image(depth_msg));
      write_camera_model(camera_model_path, info_msg, transform, options_);

      Options run_options = options_;
      run_options.image = rgb_path.string();
      run_options.depth_image = depth_path.string();
      run_options.camera_model = camera_model_path.string();

      RCLCPP_INFO(get_logger(), "Running OWL + SAM on synchronized RGB/depth frames");
      ParsedResult result;
      last_status_ = run_python(run_options, script_, &result);
      if (last_status_ != EXIT_SUCCESS) {
        RCLCPP_ERROR(get_logger(), "OWL + SAM pipeline failed with status %d", last_status_);
      } else {
        ++frame_count_;
        if (startup_watchdog_) {
          startup_watchdog_->cancel();
          startup_watchdog_.reset();
        }

        const double ts = static_cast<double>(rgb_msg->header.stamp.sec) +
          rgb_msg->header.stamp.nanosec * 1e-9;

        if (!result.detections.empty()) {
          // Multi-label path: centroids come directly from daemon JSON response.
          for (const auto & det : result.detections) {
            if (options_.mode == "scan") {
              constexpr float kEps = 1e-3f;
              const bool at_origin =
                std::abs(det.cx) < kEps && std::abs(det.cy) < kEps && std::abs(det.cz) < kEps;
              if (!at_origin) {
                scan_detections_.push_back({det.label, det.score, det.cx, det.cy, det.cz, ts});
              }
              RCLCPP_DEBUG(get_logger(), "[scan] frame=%d label=%s score=%.3f pos=(%.2f,%.2f,%.2f)",
                frame_count_, det.label.c_str(), det.score, det.cx, det.cy, det.cz);
            }
            // Publish markers in both modes.
            publish_objects(det.cx, det.cy, det.cz, det.label, det.score, rgb_msg->header.stamp);
            publish_detections(det.cx, det.cy, det.cz, det.label, det.score, rgb_msg->header.stamp);
            // In live mode also stream the PointCloud2; in scan mode the daemon
            // writes PCD files already handled by hotspot fusion.
            if (options_.mode != "scan" && !det.output_points.empty()) {
              pcl::PointCloud<pcl::PointXYZ> cloud;
              if (load_stable_pcd(det.output_points, cloud)) {
                publish_points(cloud, rgb_msg->header.stamp);
              }
            }
            // Clean up per-detection PCD (both modes — daemon wrote it for us).
            if (!det.output_points.empty()) {
              std::error_code ec;
              fs::remove(det.output_points, ec);
            }
          }
        } else if (!result.output_points.empty()) {
          // Legacy single-detection path (daemon older than multi-label support).
          pcl::PointCloud<pcl::PointXYZ> cloud;
          if (load_stable_pcd(result.output_points, cloud)) {
            float cx, cy, cz;
            compute_centroid(cloud, cx, cy, cz);
            if (options_.mode == "scan") {
              constexpr float kEps = 1e-3f;
              if (!(std::abs(cx) < kEps && std::abs(cy) < kEps && std::abs(cz) < kEps)) {
                scan_detections_.push_back({options_.text, result.score, cx, cy, cz, ts});
              } else {
                RCLCPP_WARN(get_logger(), "[scan] legacy: skipping origin centroid for '%s'", options_.text.c_str());
              }
            }
            publish_objects(cx, cy, cz, options_.text, result.score, rgb_msg->header.stamp);
            publish_detections(cx, cy, cz, options_.text, result.score, rgb_msg->header.stamp);
            if (options_.mode != "scan") {
              publish_points(cloud, rgb_msg->header.stamp);
            }
          }
          std::error_code ec;
          fs::remove(result.output_points, ec);
        }

        RCLCPP_INFO(get_logger(), "[%s] frame=%d detections=%zu",
          options_.mode.c_str(), frame_count_, result.detections.size());
      }

      // Always clean up scratch files — even when inference fails.
      for (const auto & p : {rgb_path, depth_path, camera_model_path}) {
        std::error_code ec;
        fs::remove(p, ec);
        if (ec && ec != std::errc::no_such_file_or_directory) {
          RCLCPP_WARN(get_logger(), "Could not remove temp file %s: %s",
            p.string().c_str(), ec.message().c_str());
        }
      }
    } catch (const std::exception & error) {
      last_status_ = EXIT_FAILURE;
      RCLCPP_ERROR(get_logger(), "%s", error.what());
    }

    processing_ = false;
    if (options_.mode == "scan") {
      reset_idle_timer();
    } else if (!options_.continuous) {
      rclcpp::shutdown();
    }
  }

  void reset_idle_timer()
  {
    if (!idle_timer_) {
      idle_timer_ = create_wall_timer(
        std::chrono::milliseconds(options_.scan_idle_timeout_ms),
        std::bind(&TopicRunner::finalize_and_shutdown, this));
      RCLCPP_INFO(get_logger(), "[scan] Idle timeout armed: %d ms",
        options_.scan_idle_timeout_ms);
    } else {
      idle_timer_->reset();
    }
  }

  void finalize_and_shutdown()
  {
    finalize();
    std::thread([]() { rclcpp::shutdown(); }).detach();
  }

  void save_scan_outputs(bool is_final = false)
  {
    using json = nlohmann::json;

    const fs::path out_dir{options_.output_dir};
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
      RCLCPP_ERROR(get_logger(), "[scan] Cannot create output dir %s: %s",
        out_dir.string().c_str(), ec.message().c_str());
      return;
    }

    // Copy hotspots.json accumulated by the daemon (already deduplicated per-frame).
    if (!options_.output_hotspots.empty() && fs::exists(options_.output_hotspots)) {
      fs::copy_file(options_.output_hotspots, out_dir / "hotspots.json",
        fs::copy_options::overwrite_existing, ec);
      if (!ec) {
        RCLCPP_INFO(get_logger(), "[scan] Saved hotspots.json");
      } else {
        RCLCPP_WARN(get_logger(), "[scan] Could not copy hotspots.json: %s",
          ec.message().c_str());
      }
    }

    // Deduplicate scan_detections_ into nodes using merge_radius.
    const float merge_r = [&]() -> float {
      try { return std::stof(options_.merge_radius); } catch (...) { return 0.10f; }
    }();

    auto dist3 = [](float ax, float ay, float az, float bx, float by, float bz) {
      const float dx = ax - bx, dy = ay - by, dz = az - bz;
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    struct Node {
      std::string id;
      std::string label;
      float cx, cy, cz;
      float confidence;
      int detection_count;
    };
    std::vector<Node> node_list;
    std::map<std::string, int> label_counter;

    for (const auto & det : scan_detections_) {
      bool merged = false;
      for (auto & node : node_list) {
        if (node.label == det.label &&
            dist3(node.cx, node.cy, node.cz, det.cx, det.cy, det.cz) < merge_r)
        {
          const float w = static_cast<float>(node.detection_count);
          node.cx = (node.cx * w + det.cx) / (w + 1.0f);
          node.cy = (node.cy * w + det.cy) / (w + 1.0f);
          node.cz = (node.cz * w + det.cz) / (w + 1.0f);
          node.confidence = std::max(node.confidence, det.score);
          ++node.detection_count;
          merged = true;
          break;
        }
      }
      if (!merged) {
        const std::string nid = det.label + "_" + std::to_string(label_counter[det.label]++);
        node_list.push_back({nid, det.label, det.cx, det.cy, det.cz, det.score, 1});
      }
    }

    json nodes_arr = json::array();
    for (const auto & n : node_list) {
      nodes_arr.push_back({
        {"id", n.id}, {"label", n.label},
        {"position", {n.cx, n.cy, n.cz}},
        {"confidence", n.confidence},
        {"detection_count", n.detection_count},
      });
    }

    // Build "near" edges — matches NEAR_THRESHOLD used in scene_graph_node.py.
    json edges_arr = json::array();
    const float near_thresh = static_cast<float>(options_.near_threshold);
    for (size_t i = 0; i < node_list.size(); ++i) {
      for (size_t j = i + 1; j < node_list.size(); ++j) {
        const auto & a = node_list[i];
        const auto & b = node_list[j];
        if (dist3(a.cx, a.cy, a.cz, b.cx, b.cy, b.cz) < near_thresh) {
          edges_arr.push_back({
            {"source", a.id}, {"target", b.id}, {"relation", "near"},
          });
        }
      }
    }

    json sg;
    sg["schema_version"]    = "1.1";
    sg["scan_complete"]     = is_final;
    sg["nodes"] = nodes_arr;
    sg["edges"] = edges_arr;
    sg["frame_count"]       = frame_count_;
    sg["total_frames_seen"] = total_frame_count_;
    sg["frames_dropped"]    = total_frame_count_ - frame_count_;

    const fs::path sg_path = out_dir / "scene_graph.json";
    const fs::path sg_tmp  = fs::path(sg_path.string() + ".tmp");
    {
      std::ofstream sg_out(sg_tmp);
      if (!sg_out) {
        RCLCPP_ERROR(get_logger(), "[scan] Failed to write %s", sg_tmp.string().c_str());
        return;
      }
      sg_out << sg.dump(2) << "\n";
    }
    std::error_code rename_ec;
    fs::rename(sg_tmp, sg_path, rename_ec);
    if (rename_ec) {
      RCLCPP_ERROR(get_logger(), "[scan] Failed to rename scene_graph.json: %s",
        rename_ec.message().c_str());
    } else {
      RCLCPP_INFO(get_logger(), "[scan] Saved scene_graph.json (%zu nodes, %zu edges)",
        node_list.size(), edges_arr.size());
    }
  }

  static void compute_centroid(
    const pcl::PointCloud<pcl::PointXYZ> & cloud,
    float & cx, float & cy, float & cz)
  {
    cx = cy = cz = 0.0f;
    for (const auto & pt : cloud) { cx += pt.x; cy += pt.y; cz += pt.z; }
    const float n = static_cast<float>(cloud.size());
    cx /= n; cy /= n; cz /= n;
  }

  bool load_stable_pcd(
    const std::string & pcd_path,
    pcl::PointCloud<pcl::PointXYZ> & cloud)
  {
    std::error_code ec;
    const auto sz = fs::file_size(pcd_path, ec);
    if (ec || sz == 0) {
      RCLCPP_WARN(get_logger(), "PCD missing or empty: %s", pcd_path.c_str());
      return false;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, cloud) < 0) {
      if ((now() - last_pcd_warn_).seconds() > 2.0) {
        RCLCPP_WARN(get_logger(), "Failed to load PCD: %s", pcd_path.c_str());
        last_pcd_warn_ = now();
      }
      return false;
    }
    cloud.is_dense = false;
    if (cloud.empty()) {
      RCLCPP_DEBUG(get_logger(), "Empty cloud: %s", pcd_path.c_str());
      return false;
    }
    return true;
  }

  void publish_points(
    const pcl::PointCloud<pcl::PointXYZ> & cloud,
    const builtin_interfaces::msg::Time & stamp)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.is_dense = false;
    msg.header.frame_id = options_.map_frame;
    msg.header.stamp = stamp;
    points_pub_->publish(msg);
    if (options_.debug) {
      RCLCPP_DEBUG(get_logger(), "Publishing cloud @ stamp=%.3f frame=%s",
        rclcpp::Time(stamp).seconds(), options_.map_frame.c_str());
      RCLCPP_INFO(get_logger(), "Published %zu points", cloud.size());
    }
  }

  void publish_objects(
    float cx, float cy, float cz,
    const std::string & label,
    float score,
    const builtin_interfaces::msg::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray array;

    // Clear stale markers from previous frames before adding new ones.
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(clear);

    const float c = std::clamp(score, 0.0f, 1.0f);
    const auto lifetime = rclcpp::Duration::from_seconds(0.3);

    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id = options_.map_frame;
    sphere.header.stamp = stamp;
    sphere.ns = "samowl_objects";
    sphere.id = 0;
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = cx;
    sphere.pose.position.y = cy;
    sphere.pose.position.z = cz;
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2;
    sphere.color.r = 1.0f - c;
    sphere.color.g = c;
    sphere.color.a = 1.0f;
    sphere.lifetime = lifetime;
    array.markers.push_back(sphere);

    visualization_msgs::msg::Marker text;
    text.header = sphere.header;
    text.ns = "samowl_labels";
    text.id = 0;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = cx;
    text.pose.position.y = cy;
    text.pose.position.z = cz + 0.25f;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.15;
    text.color.r = text.color.g = text.color.b = 1.0f;
    text.color.a = 1.0f;
    text.text = label + " (" + std::to_string(static_cast<int>(score * 100.0f)) + "%)";
    text.lifetime = lifetime;
    array.markers.push_back(text);

    objects_pub_->publish(array);
  }

  void publish_detections(
    float cx, float cy, float cz,
    const std::string & label,
    float score,
    const builtin_interfaces::msg::Time & stamp)
  {
    const std::string id = label + "_" + std::to_string(detection_counter_++);
    const double ts = static_cast<double>(stamp.sec) + stamp.nanosec * 1e-9;

    using json = nlohmann::json;
    json j;
    j["id"]       = id;
    j["frame_id"] = options_.map_frame;
    j["stamp"]    = ts;
    j["label"]    = label;
    j["score"]    = score;
    j["position"] = {cx, cy, cz};

    std_msgs::msg::String msg;
    msg.data = j.dump();
    detections_pub_->publish(msg);
  }

  Options options_;
  std::string script_;
  message_filters::Subscriber<Image> rgb_sub_;
  message_filters::Subscriber<Image> depth_sub_;
  message_filters::Subscriber<CameraInfo> camera_info_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr objects_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detections_pub_;
  rclcpp::Time last_pcd_warn_{0, 0, RCL_ROS_TIME};
  int detection_counter_{0};
  std::atomic_bool processing_{false};
  std::atomic_bool finalized_{false};
  int last_status_{EXIT_SUCCESS};
  int frame_count_{0};
  int total_frame_count_{0};
  std::vector<DetectionRecord> scan_detections_;
  rclcpp::TimerBase::SharedPtr idle_timer_;
  rclcpp::TimerBase::SharedPtr startup_watchdog_;
  rclcpp::TimerBase::SharedPtr periodic_flush_timer_;
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

  // Resolve output paths to absolute so file existence checks and copies work
  // regardless of the process CWD (important for scan mode output).
  {
    auto to_abs = [](std::string & p) {
      if (!p.empty()) {
        p = fs::absolute(p).string();
      }
    };
    to_abs(options.output_mask);
    to_abs(options.output_boundary);
    to_abs(options.output_depth_mask);
    to_abs(options.output_points);
    to_abs(options.output_hotspots);
    to_abs(options.output_dir);
  }

  // In scan mode, remove any hotspot file left over from a prior run so
  // fresh scans never inherit phantom objects.
  if (options.mode == "scan" && !options.output_hotspots.empty()) {
    std::error_code ec;
    fs::remove(options.output_hotspots, ec);
    if (!ec) {
      std::cerr << "[scan] Removed stale hotspot file: " << options.output_hotspots << "\n";
    }
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

  // Install SIGTERM handler after rclcpp::init so rclcpp's SIGINT handler is
  // already in place and we don't clobber it.  SIGTERM (systemd, docker stop)
  // is not handled by rclcpp by default; without this, the process terminates
  // immediately and finalize() never runs.
  std::signal(SIGTERM, [](int) { rclcpp::shutdown(); });

  auto node = std::make_shared<TopicRunner>(options, "");
  try {
    rclcpp::spin(node);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(node->get_logger(), "Executor exception: %s", ex.what());
  }

  // finalize() is idempotent — always safe to call; writes scan outputs even
  // when spin exits early via exception, SIGINT, SIGTERM, or watchdog timeout.
  if (options.mode == "scan") {
    node->finalize();
  }

  const int status = node->last_status();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  // Reap daemon so no orphan process is left behind after the scan.
  if (g_daemon_pid > 0) {
    ::kill(g_daemon_pid, SIGTERM);
    ::waitpid(g_daemon_pid, nullptr, 0);
    g_daemon_pid = -1;
  }

  return status;
}
