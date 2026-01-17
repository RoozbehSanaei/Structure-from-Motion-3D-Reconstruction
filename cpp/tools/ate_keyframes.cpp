#include <algorithm>
#include <array>
#include <numeric>
#include <cmath>
#include <fstream>
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
using sfm::det;

namespace {

struct Args {
  std::string par_path;
  std::string keyframes_csv;
  int start = 0;
  int count = 4;
  bool sim3 = true;
};

std::optional<std::string_view> get_flag_value(const std::vector<std::string_view>& argv, std::string_view flag) {
  for (size_t i = 0; i + 1 < argv.size(); ++i) {
    if (argv[i] == flag) return argv[i + 1];
  }
  return std::nullopt;
}

bool has_flag(const std::vector<std::string_view>& argv, std::string_view flag) {
  for (const auto& a : argv) if (a == flag) return true;
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
  if (auto v = get_flag_value(argv, "--start")) {
    if (auto iv = parse_int(*v)) a.start = *iv;
  }
  if (auto v = get_flag_value(argv, "--count")) {
    if (auto iv = parse_int(*v)) a.count = *iv;
  }
  if (has_flag(argv, "--se3")) a.sim3 = false;
  if (has_flag(argv, "--sim3")) a.sim3 = true;
  return a;
}

void usage() {
  std::cerr
      << "ate_keyframes (C++20, no OpenCV)\n"
      << "Compute ATE RMSE over N keyframes using ground-truth poses from Middlebury *_par.txt.\n\n"
      << "Usage:\n"
      << "  ate_keyframes --par <templeR_par.txt> --keyframes <keyframes_camera_centers.csv>\n"
      << "               [--start 0 --count 4] [--sim3|--se3]\n\n"
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
    if (ch == '"') { in_quotes = !in_quotes; continue; }
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

std::optional<size_t> find_col(const std::vector<std::string>& cols, std::string_view name) {
  for (size_t i = 0; i < cols.size(); ++i) {
    if (cols[i] == name) return i;
  }
  return std::nullopt;
}

std::vector<KeyframeRow> read_keyframes_csv(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};

  std::string header;
  if (!std::getline(f, header)) return {};

  const auto cols = split_csv_line(header);
  auto idx_img = find_col(cols, "image");
  auto idx_x   = find_col(cols, "x");
  auto idx_y   = find_col(cols, "y");
  auto idx_z   = find_col(cols, "z");
  if (!idx_img || !idx_x || !idx_y || !idx_z) return {};

  std::vector<KeyframeRow> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto parts = split_csv_line(line);
    if (parts.size() != cols.size()) continue;
    KeyframeRow r{};
    r.image = parts[*idx_img];
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
  if (!std::getline(f, line)) return std::nullopt; // first line: camera count (ignored)

  std::unordered_map<std::string, ParEntry> out;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string name;
    iss >> name;
    if (name.empty()) continue;

    // line: name + 9*K + 9*R + 3*t  (total 22 tokens)
    double Ktmp[9];
    for (int i = 0; i < 9; ++i) if (!(iss >> Ktmp[i])) { name.clear(); break; }
    if (name.empty()) continue;

    ParEntry e{};
    for (int i = 0; i < 9; ++i) {
      double v{};
      if (!(iss >> v)) { name.clear(); break; }
      e.R.a[i] = v;
    }
    if (name.empty()) continue;

    double tx{}, ty{}, tz{};
    if (!(iss >> tx >> ty >> tz)) continue;
    e.t = {tx, ty, tz};
    out.emplace(std::move(name), e);
  }
  return out;
}

Vec3 camera_center_world(const Mat33& R, const Vec3& t) {
  // x_cam = R * X_world + t  =>  camera center in world: C = -R^T * t
  return -1.0 * (transpose(R) * t);
}

// ----- Minimal symmetric 3x3 eigen decomposition (Jacobi) -----

struct EigenSym3 {
  Mat33 V = Mat33::I(); // columns are eigenvectors
  Vec3 eval{};          // eigenvalues
};

EigenSym3 jacobi_eigen(const Mat33& A_in) {
  Mat33 A = A_in;
  Mat33 V = Mat33::I();

  auto absd = [](double x) { return x < 0 ? -x : x; };

  for (int it = 0; it < 64; ++it) {
    int p = 0, q = 1;
    double max_off = absd(A(0,1));
    const double a02 = absd(A(0,2));
    const double a12 = absd(A(1,2));
    if (a02 > max_off) { max_off = a02; p = 0; q = 2; }
    if (a12 > max_off) { max_off = a12; p = 1; q = 2; }

    if (max_off < 1e-15) break;

    const double app = A(p,p);
    const double aqq = A(q,q);
    const double apq = A(p,q);

    const double phi = 0.5 * std::atan2(2.0*apq, (aqq - app));
    const double c = std::cos(phi);
    const double s = std::sin(phi);

    // Rotate A: A' = J^T A J
    for (int k = 0; k < 3; ++k) {
      const double aik = A(p,k);
      const double aqk = A(q,k);
      A(p,k) = c*aik - s*aqk;
      A(q,k) = s*aik + c*aqk;
    }
    for (int k = 0; k < 3; ++k) {
      const double akp = A(k,p);
      const double akq = A(k,q);
      A(k,p) = c*akp - s*akq;
      A(k,q) = s*akp + c*akq;
    }

    // Enforce symmetry numerically
    A(p,q) = A(q,p) = 0.5 * (A(p,q) + A(q,p));

    // Update eigenvectors
    for (int k = 0; k < 3; ++k) {
      const double vkp = V(k,p);
      const double vkq = V(k,q);
      V(k,p) = c*vkp - s*vkq;
      V(k,q) = s*vkp + c*vkq;
    }
  }

  EigenSym3 E{};
  E.V = V;
  E.eval = {A(0,0), A(1,1), A(2,2)};
  return E;
}

Vec3 col(const Mat33& M, int j) {
  return {M(0,j), M(1,j), M(2,j)};
}

void set_col(Mat33& M, int j, const Vec3& v) {
  M(0,j) = v.x; M(1,j) = v.y; M(2,j) = v.z;
}

struct SVD3 {
  Mat33 U = Mat33::I();
  Mat33 V = Mat33::I();
  Vec3 S{}; // singular values (descending)
};

SVD3 svd3(const Mat33& M) {
  // SVD via eigen-decomposition of M^T M (symmetric).
  const Mat33 MtM = transpose(M) * M;
  EigenSym3 E = jacobi_eigen(MtM);

  struct IdxVal { int idx; double val; };
  std::array<IdxVal,3> sv{};
  for (int i = 0; i < 3; ++i) {
    const double lam = (i==0?E.eval.x:(i==1?E.eval.y:E.eval.z));
    sv[i] = {i, std::sqrt(std::max(0.0, lam))};
  }
  std::sort(sv.begin(), sv.end(), [](const IdxVal& a, const IdxVal& b){ return a.val > b.val; });

  Mat33 V{};
  Vec3 S{};
  for (int j = 0; j < 3; ++j) {
    const int src = sv[j].idx;
    set_col(V, j, col(E.V, src));
    if (j==0) S.x = sv[j].val;
    if (j==1) S.y = sv[j].val;
    if (j==2) S.z = sv[j].val;
  }

  // Compute U columns: u_i = M v_i / s_i
  Vec3 u0{}, u1{}, u2{};
  auto safe_div = [](const Vec3& v, double s) {
    if (s < 1e-12) return Vec3{0,0,0};
    return v / s;
  };
  u0 = safe_div(M * col(V,0), S.x);
  u1 = safe_div(M * col(V,1), S.y);
  u2 = safe_div(M * col(V,2), S.z);

  // Orthonormalize U using Gram-Schmidt; fall back to cross products.
  auto gs = [](Vec3 a, const Vec3& b) {
    return a - dot(a,b) * b;
  };
  u0 = (norm(u0) > 1e-12) ? unit(u0) : Vec3{1,0,0};
  u1 = gs(u1, u0);
  u1 = (norm(u1) > 1e-12) ? unit(u1) : unit(cross(u0, Vec3{0,0,1}));
  if (norm(u1) < 1e-12) u1 = unit(cross(u0, Vec3{0,1,0}));
  u2 = unit(cross(u0, u1));

  Mat33 U{};
  set_col(U, 0, u0);
  set_col(U, 1, u1);
  set_col(U, 2, u2);

  // Ensure right-handedness: det(U)*det(V) should be positive for proper rotation in Umeyama.
  // (We let Umeyama fix reflection explicitly.)
  return {U, V, S};
}

struct Alignment {
  double s = 1.0;
  Mat33 R = Mat33::I();
  Vec3 t{};
};

Alignment umeyama(const std::vector<Vec3>& src, const std::vector<Vec3>& dst, bool with_scale) {
  const int N = static_cast<int>(src.size());
  Alignment A{};

  Vec3 mu_src{}, mu_dst{};
  for (int i = 0; i < N; ++i) {
    mu_src = mu_src + src[i];
    mu_dst = mu_dst + dst[i];
  }
  mu_src = mu_src / static_cast<double>(N);
  mu_dst = mu_dst / static_cast<double>(N);

  std::vector<Vec3> X; X.reserve(N);
  std::vector<Vec3> Y; Y.reserve(N);
  for (int i = 0; i < N; ++i) {
    X.push_back(src[i] - mu_src);
    Y.push_back(dst[i] - mu_dst);
  }

  // Covariance: (1/N) sum Y_i X_i^T
  Mat33 cov{};
  for (int i = 0; i < N; ++i) {
    const Vec3& y = Y[i];
    const Vec3& x = X[i];
    cov(0,0) += y.x*x.x; cov(0,1) += y.x*x.y; cov(0,2) += y.x*x.z;
    cov(1,0) += y.y*x.x; cov(1,1) += y.y*x.y; cov(1,2) += y.y*x.z;
    cov(2,0) += y.z*x.x; cov(2,1) += y.z*x.y; cov(2,2) += y.z*x.z;
  }
  const double invN = 1.0 / static_cast<double>(N);
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) cov(r,c) *= invN;

  const SVD3 svd = svd3(cov);

  // Reflection handling
  Mat33 D = Mat33::I();
  if (det(svd.U) * det(svd.V) < 0.0) D(2,2) = -1.0;

  const Mat33 R = svd.U * D * transpose(svd.V);

  double var_src = 0.0;
  for (int i = 0; i < N; ++i) var_src += dot(X[i], X[i]);
  var_src *= invN;

  double s = 1.0;
  if (with_scale) {
    const double tr = svd.S.x * D(0,0) + svd.S.y * D(1,1) + svd.S.z * D(2,2);
    if (var_src > 1e-15) s = tr / var_src;
  }

  const Vec3 t = mu_dst - (R * mu_src) * s;

  A.s = with_scale ? s : 1.0;
  A.R = R;
  A.t = t;
  return A;
}

Vec3 apply(const Alignment& A, const Vec3& p) {
  return (A.R * p) * A.s + A.t;
}

} // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (args.par_path.empty() || args.keyframes_csv.empty() || args.count <= 1 || args.start < 0) {
    usage();
    return 2;
  }

  const auto kfs = read_keyframes_csv(args.keyframes_csv);
  if (kfs.empty()) {
    std::cerr << "Failed to read keyframes CSV or missing columns: " << args.keyframes_csv << "\n";
    return 2;
  }
  if (args.start + args.count > static_cast<int>(kfs.size())) {
    std::cerr << "Requested range exceeds keyframes CSV rows: start=" << args.start
              << " count=" << args.count << " rows=" << kfs.size() << "\n";
    return 2;
  }

  const auto par_opt = read_par(args.par_path);
  if (!par_opt) {
    std::cerr << "Failed to read par file: " << args.par_path << "\n";
    return 2;
  }
  const auto& par = *par_opt;

  std::vector<Vec3> est, gt;
  std::vector<std::string> names;
  est.reserve(static_cast<size_t>(args.count));
  gt.reserve(static_cast<size_t>(args.count));
  names.reserve(static_cast<size_t>(args.count));

  for (int k = 0; k < args.count; ++k) {
    const auto& row = kfs[static_cast<size_t>(args.start + k)];
    const auto it = par.find(row.image);
    if (it == par.end()) {
      std::cerr << "Image name not found in par file: " << row.image << "\n";
      return 2;
    }
    est.push_back(row.c_est);
    gt.push_back(camera_center_world(it->second.R, it->second.t));
    names.push_back(row.image);
  }

  const Alignment A = umeyama(est, gt, args.sim3);

  std::vector<double> errs;
  errs.reserve(est.size());

  double mse = 0.0;
  for (size_t i = 0; i < est.size(); ++i) {
    const Vec3 e = apply(A, est[i]) - gt[i];
    const double d = norm(e);
    errs.push_back(d);
    mse += d*d;
  }
  mse /= static_cast<double>(errs.size());
  const double rmse = std::sqrt(mse);

  std::vector<double> sorted = errs;
  std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(errs.begin(), errs.end(), 0.0) / static_cast<double>(errs.size());
  const double median = sorted[sorted.size()/2];
  const double maxv = sorted.back();

  std::cout << "ATE (N keyframes)\n";
  std::cout << "  mode: " << (args.sim3 ? "Sim(3)" : "SE(3)") << "\n";
  std::cout << "  start: " << args.start << "  count: " << args.count << "\n";
  std::cout << "  keyframes:\n";
  for (size_t i = 0; i < names.size(); ++i) {
    std::cout << "    [" << (args.start + static_cast<int>(i)) << "] " << names[i] << "\n";
  }
  if (args.sim3) std::cout << "  scale (s): " << A.s << "\n";
  std::cout << "  ATE_RMSE: " << rmse << "\n";
  std::cout << "  mean/median/max: " << mean << " / " << median << " / " << maxv << "\n";
  std::cout << "  per_frame_error:\n";
  for (size_t i = 0; i < names.size(); ++i) {
    std::cout << "    " << names[i] << ": " << errs[i] << "\n";
  }

  return 0;
}
