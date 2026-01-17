#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "linalg.hpp"

using namespace sfm;

namespace {

[[nodiscard]] std::string_view trim_token(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.remove_suffix(1);
  for (char q : {'"', '\''}) {
    if (!s.empty() && s.front() == q) s.remove_prefix(1);
    if (!s.empty() && s.back()  == q) s.remove_suffix(1);
  }
  return s;
}

struct Camera {
  Mat33 K{};
  Mat33 R{};
  Vec3  t{};   // world -> camera: x_cam = R * X_world + t
};

struct Keyframe {
  int kf_id{};
  std::string image;
};

struct Edge {
  int i{};
  int j{};
  std::string kind;
  Vec3 rvec{};
  Vec3 t{};
};

[[nodiscard]] std::vector<std::string_view> split_ws(std::string_view s) {
  std::vector<std::string_view> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) break;
    const auto start = i;
    while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    out.emplace_back(s.substr(start, i - start));
  }
  return out;
}

[[nodiscard]] std::vector<std::string_view> split_csv_line(std::string_view s) {
  // Simple CSV split (no quoted commas expected in our generated files).
  std::vector<std::string_view> out;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == ',') {
      out.emplace_back(trim_token(s.substr(start, i - start)));
      start = i + 1;
    }
  }
  return out;
}


[[nodiscard]] std::optional<double> to_double(std::string_view s) {
  // Fast-ish parse via stringstream to keep portable and robust.
  s = trim_token(s);
  std::stringstream ss{std::string(s)};
  double v{};
  ss >> v;
  if (!ss.fail() && ss.eof()) return v;
  return std::nullopt;
}

[[nodiscard]] std::optional<int> to_int(std::string_view s) {
  s = trim_token(s);
  std::stringstream ss{std::string(s)};
  int v{};
  ss >> v;
  if (!ss.fail() && ss.eof()) return v;
  return std::nullopt;
}

[[nodiscard]] Vec3 normalize_or_zero(const Vec3& v) {
  const double n = norm(v);
  if (n < 1e-12) return {0,0,0};
  return v / n;
}

[[nodiscard]] double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

[[nodiscard]] double rad2deg(double r) {
  return r * (180.0 / std::numbers::pi);
}

[[nodiscard]] std::optional<std::unordered_map<std::string, Camera>>
load_middlebury_par(const std::string& par_path) {
  std::ifstream f(par_path);
  if (!f) return std::nullopt;

  // Middlebury format (one line per image):
  // imgname.png k11..k33 r11..r33 t1 t2 t3, and P = K [R t].
  // We parse and store K, R, t for each image.
  std::unordered_map<std::string, Camera> cams;

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto toks = split_ws(line);
    if (toks.size() < 22) continue; // allow trailing spaces, but require core tokens
    const std::string img(toks[0]);

    auto parse_mat33 = [&](std::size_t off) -> std::optional<Mat33> {
      Mat33 M{};
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          const auto v = to_double(toks[off + r*3 + c]);
          if (!v) return std::nullopt;
          M(r,c) = *v;
        }
      }
      return M;
    };

    const auto K = parse_mat33(1);
    const auto R = parse_mat33(10);
    const auto t1 = to_double(toks[19]);
    const auto t2 = to_double(toks[20]);
    const auto t3 = to_double(toks[21]);
    if (!K || !R || !t1 || !t2 || !t3) continue;

    cams.emplace(img, Camera{*K, *R, Vec3{*t1, *t2, *t3}});
  }
  return cams;
}

[[nodiscard]] std::optional<std::vector<Keyframe>>
load_keyframes(const std::string& keyframes_csv) {
  std::ifstream f(keyframes_csv);
  if (!f) return std::nullopt;

  std::string header;
  if (!std::getline(f, header)) return std::nullopt;

  const auto cols = split_csv_line(header);
  // Expect at least: kf_id, frame_idx, image, ...
  int idx_kf_id = -1;
  int idx_image = -1;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    if (cols[i] == "kf_id") idx_kf_id = i;
    if (cols[i] == "image") idx_image = i;
  }
  if (idx_kf_id < 0 || idx_image < 0) return std::nullopt;

  std::vector<Keyframe> kfs;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto toks = split_csv_line(line);
    if (static_cast<int>(toks.size()) <= std::max(idx_kf_id, idx_image)) continue;

    const auto id = to_int(toks[idx_kf_id]);
    if (!id) continue;
    kfs.push_back(Keyframe{*id, std::string(toks[idx_image])});
  }

  // Ensure vector is indexable by kf_id (common in our outputs).
  // If ids are contiguous and sorted, keep as-is; else remap.
  bool ok = true;
  for (std::size_t i = 0; i < kfs.size(); ++i) {
    if (kfs[i].kf_id != static_cast<int>(i)) { ok = false; break; }
  }
  if (!ok) {
    std::vector<Keyframe> remap;
    remap.resize(kfs.size());
    for (const auto& k : kfs) {
      if (k.kf_id >= 0 && static_cast<std::size_t>(k.kf_id) < remap.size())
        remap[static_cast<std::size_t>(k.kf_id)] = k;
    }
    return remap;
  }
  return kfs;
}

[[nodiscard]] std::optional<std::vector<Edge>>
load_edges(const std::string& edges_csv) {
  std::ifstream f(edges_csv);
  if (!f) return std::nullopt;

  std::string header;
  if (!std::getline(f, header)) return std::nullopt;

  const auto cols = split_csv_line(header);
  auto idx_of = [&](std::string_view name) -> int {
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) if (cols[i] == name) return i;
    return -1;
  };

  const int idx_i = idx_of("i");
  const int idx_j = idx_of("j");
  const int idx_kind = idx_of("kind");
  const int idx_rvec_x = idx_of("rvec_x");
  const int idx_rvec_y = idx_of("rvec_y");
  const int idx_rvec_z = idx_of("rvec_z");
  const int idx_t_x = idx_of("t_x");
  const int idx_t_y = idx_of("t_y");
  const int idx_t_z = idx_of("t_z");

  if (idx_i < 0 || idx_j < 0 || idx_kind < 0 ||
      idx_rvec_x < 0 || idx_rvec_y < 0 || idx_rvec_z < 0 ||
      idx_t_x < 0 || idx_t_y < 0 || idx_t_z < 0) return std::nullopt;

  std::vector<Edge> edges;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto toks = split_csv_line(line);
    const int need = std::max({idx_i, idx_j, idx_kind, idx_rvec_x, idx_rvec_y, idx_rvec_z, idx_t_x, idx_t_y, idx_t_z});
    if (static_cast<int>(toks.size()) <= need) continue;

    const auto I = to_int(toks[idx_i]);
    const auto J = to_int(toks[idx_j]);
    const auto rx = to_double(toks[idx_rvec_x]);
    const auto ry = to_double(toks[idx_rvec_y]);
    const auto rz = to_double(toks[idx_rvec_z]);
    const auto tx = to_double(toks[idx_t_x]);
    const auto ty = to_double(toks[idx_t_y]);
    const auto tz = to_double(toks[idx_t_z]);
    if (!I || !J || !rx || !ry || !rz || !tx || !ty || !tz) continue;

    edges.push_back(Edge{
      *I, *J, std::string(toks[idx_kind]),
      Vec3{*rx, *ry, *rz},
      Vec3{*tx, *ty, *tz}
    });
  }
  return edges;
}

[[nodiscard]] std::optional<std::string> arg_value(int argc, char** argv, std::string_view key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view(argv[i]) == key) return std::string(argv[i+1]);
  }
  return std::nullopt;
}

[[nodiscard]] bool has_flag(int argc, char** argv, std::string_view key) {
  for (int i = 1; i < argc; ++i) if (std::string_view(argv[i]) == key) return true;
  return false;
}

[[nodiscard]] std::optional<int> arg_int(int argc, char** argv, std::string_view key) {
  if (auto v = arg_value(argc, argv, key)) {
    return to_int(*v);
  }
  return std::nullopt;
}

void print_usage() {
  std::cerr
    << "Usage:\n"
    << "  gt_keyframe_edge --par <*_par.txt> --keyframes <keyframes_camera_centers.csv> --i <kf_id> --j <kf_id> [--edges <posegraph_edges.csv>] [--emit-csv]\n\n"
    << "Outputs:\n"
    << "  - Ground-truth relative pose edge (Rodrigues rvec + translation direction) between the two keyframes.\n"
    << "  - If --edges is provided, also prints rotation and translation-direction errors versus the estimated edge.\n";
}

} // namespace

int main(int argc, char** argv) {
  const auto par_path = arg_value(argc, argv, "--par");
  const auto kf_path  = arg_value(argc, argv, "--keyframes");
  const auto i_id     = arg_int(argc, argv, "--i");
  const auto j_id     = arg_int(argc, argv, "--j");
  const auto edges_path = arg_value(argc, argv, "--edges");
  const bool emit_csv = has_flag(argc, argv, "--emit-csv");

  if (!par_path || !kf_path || !i_id || !j_id) {
    print_usage();
    return 2;
  }

  const auto cams_opt = load_middlebury_par(*par_path);
  if (!cams_opt) {
    std::cerr << "Failed to read par file: " << *par_path << "\n";
    return 2;
  }
  const auto kfs_opt = load_keyframes(*kf_path);
  if (!kfs_opt) {
    std::cerr << "Failed to read keyframes CSV: " << *kf_path << "\n";
    return 2;
  }
  const auto& cams = *cams_opt;
  const auto& kfs  = *kfs_opt;

  if (*i_id < 0 || *j_id < 0 ||
      static_cast<std::size_t>(*i_id) >= kfs.size() ||
      static_cast<std::size_t>(*j_id) >= kfs.size()) {
    std::cerr << "Keyframe id out of range. Have " << kfs.size() << " keyframes.\n";
    return 2;
  }

  const auto& img_i = kfs[static_cast<std::size_t>(*i_id)].image;
  const auto& img_j = kfs[static_cast<std::size_t>(*j_id)].image;

  const auto it_i = cams.find(img_i);
  const auto it_j = cams.find(img_j);
  if (it_i == cams.end() || it_j == cams.end()) {
    std::cerr << "Image not found in par file: "
              << (it_i == cams.end() ? img_i : img_j) << "\n";
    return 2;
  }

  const Camera& ci = it_i->second;
  const Camera& cj = it_j->second;

  // Relative pose (world->cam extrinsics): x_j = R_ij * x_i + t_ij
  const Mat33 R_ij = cj.R * transpose(ci.R);
  const Vec3  t_ij = cj.t - (R_ij * ci.t);

  const Vec3 rvec_gt = so3_log(R_ij);
  const Vec3 tdir_gt = normalize_or_zero(t_ij);

  if (emit_csv) {
    // Match the posegraph_edges.csv column order.
    std::cout << "i,j,kind,rvec_x,rvec_y,rvec_z,t_x,t_y,t_z\n";
    std::cout << *i_id << "," << *j_id << ",gt,"
              << std::setprecision(10)
              << rvec_gt.x << "," << rvec_gt.y << "," << rvec_gt.z << ","
              << tdir_gt.x << "," << tdir_gt.y << "," << tdir_gt.z << "\n";
    return 0;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Keyframe edge (ground truth)\n";
  std::cout << "  i=" << *i_id << " (" << img_i << ")\n";
  std::cout << "  j=" << *j_id << " (" << img_j << ")\n";
  std::cout << "  rvec_gt = [" << rvec_gt.x << ", " << rvec_gt.y << ", " << rvec_gt.z << "]\n";
  std::cout << "  tdir_gt = [" << tdir_gt.x << ", " << tdir_gt.y << ", " << tdir_gt.z << "]\n";

  if (!edges_path) return 0;

  const auto edges_opt = load_edges(*edges_path);
  if (!edges_opt) {
    std::cerr << "Failed to read edges CSV: " << *edges_path << "\n";
    return 2;
  }

  const auto& edges = *edges_opt;
  const auto it = std::find_if(edges.begin(), edges.end(), [&](const Edge& e){
    return e.i == *i_id && e.j == *j_id;
  });
  if (it == edges.end()) {
    std::cerr << "Edge (i,j)=(" << *i_id << "," << *j_id << ") not found in " << *edges_path << "\n";
    return 2;
  }

  const Vec3 rvec_est = it->rvec;
  const Vec3 tdir_est = normalize_or_zero(it->t);

  const Mat33 R_est = so3_exp(rvec_est);
  const Mat33 R_err = R_est * transpose(R_ij);
  const Vec3  w_err = so3_log(R_err);
  const double rot_err_deg = rad2deg(norm(w_err));

  const double d1 = clamp(dot(tdir_est, tdir_gt), -1.0, 1.0);
  const double d2 = clamp(dot(tdir_est, Vec3{-tdir_gt.x, -tdir_gt.y, -tdir_gt.z}), -1.0, 1.0);
  const double trans_err_deg = rad2deg(std::min(std::acos(d1), std::acos(d2)));

  std::cout << "\nEstimated edge (from posegraph_edges.csv)\n";
  std::cout << "  kind     = " << it->kind << "\n";
  std::cout << "  rvec_est = [" << rvec_est.x << ", " << rvec_est.y << ", " << rvec_est.z << "]\n";
  std::cout << "  tdir_est = [" << tdir_est.x << ", " << tdir_est.y << ", " << tdir_est.z << "]\n";

  std::cout << "\nErrors vs ground truth\n";
  std::cout << "  rotation error (deg)            = " << rot_err_deg << "\n";
  std::cout << "  translation direction error (deg)= " << trans_err_deg << "\n";

  return 0;
}
