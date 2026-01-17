#include <fstream>
#include <algorithm>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "linalg.hpp"

using sfm::Vec3;
using sfm::Mat33;
using sfm::dot;
using sfm::cross;
using sfm::norm;
using sfm::unit;
using sfm::transpose;

namespace {

struct Args {
  std::string par_path;
  std::string keyframes_csv;
  int i = 0;
  int j = 1;
  bool sim3 = true; // default for monocular
};

std::optional<std::string_view> get_flag_value(std::vector<std::string_view>& argv, std::string_view flag) {
  for (size_t k = 0; k < argv.size(); ++k) {
    if (argv[k] == flag) {
      if (k + 1 >= argv.size()) return std::nullopt;
      return argv[k + 1];
    }
  }
  return std::nullopt;
}

bool has_flag(const std::vector<std::string_view>& argv, std::string_view flag) {
  for (auto a : argv) if (a == flag) return true;
  return false;
}

std::optional<int> parse_int(std::string_view s) {
  try {
    size_t pos = 0;
    int v = std::stoi(std::string(s), &pos);
    if (pos != s.size()) return std::nullopt;
    return v;
  } catch (...) {
    return std::nullopt;
  }
}

Args parse_args(int argc, char** argv_raw) {
  std::vector<std::string_view> argv;
  argv.reserve(static_cast<size_t>(argc));
  for (int k = 0; k < argc; ++k) argv.emplace_back(argv_raw[k]);

  Args a{};
  if (auto v = get_flag_value(argv, "--par")) a.par_path = std::string(*v);
  if (auto v = get_flag_value(argv, "--keyframes")) a.keyframes_csv = std::string(*v);

  if (auto v = get_flag_value(argv, "--i")) {
    if (auto iv = parse_int(*v)) a.i = *iv;
  }
  if (auto v = get_flag_value(argv, "--j")) {
    if (auto jv = parse_int(*v)) a.j = *jv;
  }

  if (has_flag(argv, "--se3")) a.sim3 = false;
  if (has_flag(argv, "--sim3")) a.sim3 = true;

  return a;
}

void usage() {
  std::cerr
      << "ate_two_frames (C++20, no OpenCV)\n"
      << "Compute ATE RMSE for two keyframes using ground-truth poses from Middlebury *_par.txt.\n\n"
      << "Usage:\n"
      << "  ate_two_frames --par <templeR_par.txt> --keyframes <keyframes_camera_centers.csv> [--i 0 --j 1] [--sim3|--se3]\n\n"
      << "Notes:\n"
      << "  - --sim3 (default) uses similarity alignment (scale + rotation + translation), typical for monocular.\n"
      << "  - --se3 uses rigid alignment (rotation + translation only).\n";
}

struct KeyframeRow {
  std::string image;
  Vec3 c_est{};
};

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  cur.reserve(line.size());
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char ch = line[i];
    if (ch == '"') {
      in_quotes = !in_quotes;
      continue;
    }
    if (!in_quotes && ch == ',') {
      out.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(ch);
  }
  out.push_back(cur);
  return out;
}

std::optional<std::vector<KeyframeRow>> read_keyframes(const std::string& csv_path) {
  std::ifstream f(csv_path);
  if (!f) return std::nullopt;

  std::string header;
  if (!std::getline(f, header)) return std::nullopt;
  const auto cols = split_csv_line(header);

  auto col_index = [&](std::string_view name) -> std::optional<size_t> {
    for (size_t i = 0; i < cols.size(); ++i) {
      if (cols[i] == name) return i;
    }
    return std::nullopt;
  };

  const auto idx_image = col_index("image");
  const auto idx_x = col_index("x");
  const auto idx_y = col_index("y");
  const auto idx_z = col_index("z");

  if (!idx_image || !idx_x || !idx_y || !idx_z) return std::nullopt;

  std::vector<KeyframeRow> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto parts = split_csv_line(line);
    if (parts.size() <= std::max({*idx_image, *idx_x, *idx_y, *idx_z})) continue;

    KeyframeRow r;
    r.image = parts[*idx_image];
    try {
      r.c_est = {std::stod(parts[*idx_x]), std::stod(parts[*idx_y]), std::stod(parts[*idx_z])};
    } catch (...) {
      continue;
    }
    out.push_back(std::move(r));
  }
  return out;
}

struct ParEntry {
  Mat33 R{};
  Vec3 t{};
};

std::optional<std::unordered_map<std::string, ParEntry>> read_par(const std::string& par_path) {
  std::ifstream f(par_path);
  if (!f) return std::nullopt;

  std::string line;
  if (!std::getline(f, line)) return std::nullopt;
  // first line: number of cameras (not strictly needed)
  std::unordered_map<std::string, ParEntry> out;

  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string name;
    iss >> name;
    if (name.empty()) continue;

    // Format (Middlebury): name + 9 K + 9 R + 3 t  => 22 tokens total.
    // We only need R and t.
    double k[9]{};
    double r[9]{};
    double t[3]{};

    for (int i = 0; i < 9; ++i) {
      if (!(iss >> k[i])) return std::nullopt;
    }
    for (int i = 0; i < 9; ++i) {
      if (!(iss >> r[i])) return std::nullopt;
    }
    for (int i = 0; i < 3; ++i) {
      if (!(iss >> t[i])) return std::nullopt;
    }

    ParEntry e;
    for (int rr = 0; rr < 3; ++rr) {
      for (int cc = 0; cc < 3; ++cc) {
        e.R(rr, cc) = r[3 * rr + cc];
      }
    }
    e.t = {t[0], t[1], t[2]};
    out.emplace(std::move(name), e);
  }
  return out;
}

Vec3 camera_center_world(const Mat33& R, const Vec3& t) {
  // World->Cam: x_cam = R x_world + t
  // Center in world: C = -R^T t
  return -(transpose(R) * t);
}

Mat33 mat_add(const Mat33& A, const Mat33& B) {
  Mat33 C{};
  for (int i = 0; i < 9; ++i) C.a[i] = A.a[i] + B.a[i];
  return C;
}
Mat33 mat_sub(const Mat33& A, const Mat33& B) {
  Mat33 C{};
  for (int i = 0; i < 9; ++i) C.a[i] = A.a[i] - B.a[i];
  return C;
}
Mat33 mat_scale(const Mat33& A, double s) {
  Mat33 C{};
  for (int i = 0; i < 9; ++i) C.a[i] = A.a[i] * s;
  return C;
}
Mat33 outer(const Vec3& u, const Vec3& v) {
  Mat33 M{};
  M(0,0) = u.x*v.x; M(0,1) = u.x*v.y; M(0,2) = u.x*v.z;
  M(1,0) = u.y*v.x; M(1,1) = u.y*v.y; M(1,2) = u.y*v.z;
  M(2,0) = u.z*v.x; M(2,1) = u.z*v.y; M(2,2) = u.z*v.z;
  return M;
}
Mat33 skew(const Vec3& v) {
  Mat33 K{};
  K(0,0)=0;     K(0,1)=-v.z;  K(0,2)= v.y;
  K(1,0)= v.z;  K(1,1)=0;     K(1,2)=-v.x;
  K(2,0)=-v.y;  K(2,1)= v.x;  K(2,2)=0;
  return K;
}

Mat33 rotation_align(const Vec3& a_raw, const Vec3& b_raw) {
  // Minimal rotation R such that R*a = b (for non-zero a,b).
  const Vec3 a = unit(a_raw);
  const Vec3 b = unit(b_raw);
  const double c = dot(a, b);
  const Vec3 v = cross(a, b);
  const double s = norm(v);

  if (s < 1e-12) {
    // Vectors are parallel or anti-parallel.
    if (c > 0.0) return Mat33::I();

    // 180-degree rotation: choose an axis orthogonal to a.
    Vec3 axis{};
    if (std::fabs(a.x) < std::fabs(a.y) && std::fabs(a.x) < std::fabs(a.z)) axis = {1,0,0};
    else if (std::fabs(a.y) < std::fabs(a.z)) axis = {0,1,0};
    else axis = {0,0,1};
    axis = unit(cross(a, axis));  // orthogonal
    // Rodrigues for angle pi: R = -I + 2*axis*axis^T
    Mat33 I = Mat33::I();
    Mat33 aaT = outer(axis, axis);
    return mat_add(mat_scale(aaT, 2.0), mat_scale(I, -1.0));
  }

  const Vec3 k = {v.x / s, v.y / s, v.z / s};
  const double angle = std::atan2(s, c);
  const double ca = std::cos(angle);
  const double sa = std::sin(angle);

  Mat33 I = Mat33::I();
  Mat33 kkT = outer(k, k);
  Mat33 K = skew(k);

  // R = ca*I + (1-ca)*k k^T + sa*[k]_x
  Mat33 R = mat_add(mat_add(mat_scale(I, ca), mat_scale(kkT, 1.0 - ca)), mat_scale(K, sa));
  return R;
}

struct Alignment {
  double s = 1.0;
  Mat33 R = Mat33::I();
  Vec3 t{};
};

Alignment align_two_points(const Vec3& p1_est, const Vec3& p2_est, const Vec3& p1_gt, const Vec3& p2_gt, bool sim3) {
  const Vec3 v_est = p2_est - p1_est;
  const Vec3 v_gt  = p2_gt  - p1_gt;

  Alignment A{};
  A.R = rotation_align(v_est, v_gt);

  const double len_est = norm(v_est);
  const double len_gt  = norm(v_gt);

  if (sim3) {
    if (len_est > 1e-12) A.s = len_gt / len_est;
    else A.s = 1.0;
  } else {
    A.s = 1.0;
  }

  A.t = p1_gt - (A.R * p1_est) * A.s;
  return A;
}

Vec3 apply(const Alignment& A, const Vec3& p) {
  return (A.R * p) * A.s + A.t;
}

double rmse_two(const Vec3& e1, const Vec3& e2) {
  const double d1 = dot(e1, e1);
  const double d2 = dot(e2, e2);
  return std::sqrt(0.5 * (d1 + d2));
}

} // namespace

int main(int argc, char** argv) {
  const auto args = parse_args(argc, argv);

  if (args.par_path.empty() || args.keyframes_csv.empty()) {
    usage();
    return 2;
  }
  if (args.i < 0 || args.j < 0 || args.i == args.j) {
    std::cerr << "Invalid indices: --i and --j must be >=0 and different.\n";
    return 2;
  }

  const auto kf_opt = read_keyframes(args.keyframes_csv);
  if (!kf_opt) {
    std::cerr << "Failed to read keyframes CSV: " << args.keyframes_csv << "\n";
    return 2;
  }
  const auto& kfs = *kf_opt;

  if (args.i >= static_cast<int>(kfs.size()) || args.j >= static_cast<int>(kfs.size())) {
    std::cerr << "Index out of range. Keyframes in CSV: " << kfs.size() << "\n";
    return 2;
  }

  const auto par_opt = read_par(args.par_path);
  if (!par_opt) {
    std::cerr << "Failed to read par file: " << args.par_path << "\n";
    return 2;
  }
  const auto& par = *par_opt;

  const auto& ki = kfs[static_cast<size_t>(args.i)];
  const auto& kj = kfs[static_cast<size_t>(args.j)];

  const auto it_i = par.find(ki.image);
  const auto it_j = par.find(kj.image);
  if (it_i == par.end() || it_j == par.end()) {
    std::cerr << "Image name not found in par file. Missing: "
              << (it_i == par.end() ? ki.image : "") << " "
              << (it_j == par.end() ? kj.image : "") << "\n";
    return 2;
  }

  const Vec3 c_gt_i = camera_center_world(it_i->second.R, it_i->second.t);
  const Vec3 c_gt_j = camera_center_world(it_j->second.R, it_j->second.t);

  const Vec3 c_est_i = ki.c_est;
  const Vec3 c_est_j = kj.c_est;

  const Alignment A = align_two_points(c_est_i, c_est_j, c_gt_i, c_gt_j, args.sim3);

  const Vec3 c_est_i_al = apply(A, c_est_i);
  const Vec3 c_est_j_al = apply(A, c_est_j);

  const Vec3 err_i = c_est_i_al - c_gt_i;
  const Vec3 err_j = c_est_j_al - c_gt_j;

  const double ate_rmse = rmse_two(err_i, err_j);

  const double len_est = norm(c_est_j - c_est_i);
  const double len_gt  = norm(c_gt_j - c_gt_i);

  std::cout.setf(std::ios::scientific);
  std::cout.precision(12);

  std::cout << "ATE (two keyframes)\n";
  std::cout << "  mode: " << (args.sim3 ? "Sim(3)" : "SE(3)") << "\n";
  std::cout << "  keyframes: [" << args.i << "] " << ki.image << "  ->  [" << args.j << "] " << kj.image << "\n";
  std::cout << "  baseline_len_est: " << len_est << "\n";
  std::cout << "  baseline_len_gt : " << len_gt << "\n";
  if (args.sim3) std::cout << "  scale (s): " << A.s << "\n";
  std::cout << "  ATE_RMSE: " << ate_rmse << "\n";
  std::cout << "  per_frame_error:\n";
  std::cout << "    " << ki.image << ": " << norm(err_i) << "\n";
  std::cout << "    " << kj.image << ": " << norm(err_j) << "\n";

  return 0;
}
