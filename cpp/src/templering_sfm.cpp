#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "linalg.hpp"
#include "dense.hpp"
#include "so3.hpp"
#include "pgm_io.hpp"
#include "minijson.hpp"

namespace fs = std::filesystem;
using sfm::GrayImage;
using sfm::RGBImage;
using sfm::Mat33;
using sfm::Vec2;
using sfm::Vec3;

// ----------------------------
// Optional geometry export
// ----------------------------
enum class ExportGeometry {
  NONE,
  POINTCLOUD,
  MESH,
  BOTH,
};

static std::optional<ExportGeometry> parse_export_geometry(const std::string& s){
  if(s == "none") return ExportGeometry::NONE;
  if(s == "pointcloud") return ExportGeometry::POINTCLOUD;
  if(s == "mesh") return ExportGeometry::MESH;
  // For CLI symmetry with Python runs; in this C++ pipeline, mesh is built from the sparse map
  // projected into a chosen keyframe.
  if(s == "mesh_stereo") return ExportGeometry::MESH;
  if(s == "both") return ExportGeometry::BOTH;
  return std::nullopt;
}

// ----------------------------
// Shared config.json loading (dependency-free)
// ----------------------------

static std::string read_text_file(const fs::path& p){
  std::ifstream f(p);
  if(!f) throw std::runtime_error("Failed to open: " + p.string());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static const minijson::Value* jget(const minijson::Value& v, std::initializer_list<const char*> path){
  const minijson::Value* cur = &v;
  for(const char* k : path){
    if(!cur->is_object()) return nullptr;
    auto it = cur->obj.find(k);
    if(it == cur->obj.end()) return nullptr;
    cur = &it->second;
  }
  return cur;
}

static const minijson::Value* jpick(const minijson::Value& root,
                                   std::initializer_list<const char*> a,
                                   std::initializer_list<const char*> b)
{
  if(const auto* va = jget(root, a)) return va;
  return jget(root, b);
}

static std::optional<int> jint(const minijson::Value* v){
  if(!v) return std::nullopt;
  if(v->is_number()) return (int)std::llround(v->num);
  return std::nullopt;
}

static std::optional<double> jdouble(const minijson::Value* v){
  if(!v) return std::nullopt;
  if(v->is_number()) return v->num;
  return std::nullopt;
}

static std::optional<bool> jbool(const minijson::Value* v){
  if(!v) return std::nullopt;
  if(v->is_bool()) return v->b;
  return std::nullopt;
}

static std::optional<std::string> jstring(const minijson::Value* v){
  if(!v) return std::nullopt;
  if(v->is_string()) return v->str;
  return std::nullopt;
}

// ----------------------------
// TempleRing par/ang parsing
// ----------------------------
struct MBRecord {
  std::string img;
  Mat33 K;
  Mat33 Rwc;
  Vec3 twc;
};

struct MBAngle { double lat{}, lon{}; };

static std::vector<MBRecord> read_par(const fs::path& par_path){
  std::ifstream f(par_path);
  if(!f) throw std::runtime_error("Failed to open: " + par_path.string());
  int n=0;
  f >> n;
  std::vector<MBRecord> recs;
  recs.reserve((size_t)n);
  for(int i=0;i<n;i++){
    MBRecord r;
    f >> r.img;
    std::array<double,21> v{};
    for(int k=0;k<21;k++) f >> v[k];
    // K
    for(int rr=0;rr<3;rr++) for(int cc=0;cc<3;cc++) r.K(rr,cc) = v[rr*3+cc];
    // R
    for(int rr=0;rr<3;rr++) for(int cc=0;cc<3;cc++) r.Rwc(rr,cc) = v[9 + rr*3+cc];
    r.twc = {v[18], v[19], v[20]};
    recs.push_back(r);
  }
  return recs;
}

static std::unordered_map<std::string, MBAngle> read_ang(const fs::path& ang_path){
  std::ifstream f(ang_path);
  if(!f) throw std::runtime_error("Failed to open: " + ang_path.string());
  std::unordered_map<std::string, MBAngle> a;
  std::string img;
  double lat, lon;
  while(f >> lat >> lon >> img){
    a.emplace(img, MBAngle{lat, lon});
  }
  return a;
}

// ----------------------------
// Pose (camera-to-world)
// ----------------------------
struct PoseCW {
  Mat33 R; // camera->world
  Vec3 t;  // camera center in world coords
  static PoseCW Identity(){ return {Mat33::I(), {0,0,0}}; }

  // world->camera: Xc = Rwc Xw + twc
  std::pair<Mat33, Vec3> inv_wc() const {
    const Mat33 Rwc = sfm::transpose(R);
    const Vec3 twc = -(Rwc * t);
    return {Rwc, twc};
  }
};

static PoseCW compose_right_inv_ij(const PoseCW& cur, const Mat33& R_ji, const Vec3& t_ji){
  // Apply inverse of i->j on the right: (j->i)
  const Mat33 R_delta = sfm::transpose(R_ji);
  const Vec3 t_delta  = -(sfm::transpose(R_ji) * t_ji);
  PoseCW out;
  out.R = cur.R * R_delta;
  out.t = (cur.R * t_delta) + cur.t;
  return out;
}

// ----------------------------
// Image ops: bilinear + gradients
// ----------------------------
static inline double sample_bilinear(const GrayImage& im, double x, double y){
  const int x0 = (int)std::floor(x);
  const int y0 = (int)std::floor(y);
  const int x1 = x0+1;
  const int y1 = y0+1;
  if(x0 < 0 || y0 < 0 || x1 >= im.w || y1 >= im.h) return 0.0;
  const double dx = x - x0;
  const double dy = y - y0;
  const double v00 = im.at(x0,y0);
  const double v10 = im.at(x1,y0);
  const double v01 = im.at(x0,y1);
  const double v11 = im.at(x1,y1);
  const double v0 = v00*(1-dx) + v10*dx;
  const double v1 = v01*(1-dx) + v11*dx;
  return v0*(1-dy) + v1*dy;
}

static GrayImage downsample2(const GrayImage& im){
  GrayImage out;
  out.w = im.w/2;
  out.h = im.h/2;
  out.pix.resize((size_t)out.w*out.h);
  for(int y=0;y<out.h;y++){
    for(int x=0;x<out.w;x++){
      // simple 2x2 box
      const int sx = 2*x;
      const int sy = 2*y;
      const int s00 = im.at(sx,sy);
      const int s10 = im.at(std::min(sx+1, im.w-1), sy);
      const int s01 = im.at(sx, std::min(sy+1, im.h-1));
      const int s11 = im.at(std::min(sx+1, im.w-1), std::min(sy+1, im.h-1));
      out.at(x,y) = (std::uint8_t)((s00+s10+s01+s11)/4);
    }
  }
  return out;
}

struct Pyramid {
  std::vector<GrayImage> lvl;
};

static Pyramid build_pyr(const GrayImage& im, int levels){
  Pyramid p;
  p.lvl.reserve((size_t)levels);
  p.lvl.push_back(im);
  for(int i=1;i<levels;i++){
    p.lvl.push_back(downsample2(p.lvl.back()));
  }
  return p;
}

// ----------------------------
// Shi-Tomasi corner detection (very small approximation)
// ----------------------------
static std::vector<Vec2> shi_tomasi(const GrayImage& im, int max_corners, double quality, int min_dist){
  // compute gradients and structure tensor response per pixel
  const int w=im.w, h=im.h;
  std::vector<double> score((size_t)w*h, 0.0);

  auto gradx = [&](int x,int y){
    const int xm = std::max(0, x-1), xp = std::min(w-1, x+1);
    return 0.5 * (double(im.at(xp,y)) - double(im.at(xm,y)));
  };
  auto grady = [&](int x,int y){
    const int ym = std::max(0, y-1), yp = std::min(h-1, y+1);
    return 0.5 * (double(im.at(x,yp)) - double(im.at(x,ym)));
  };

  // box filter window radius
  constexpr int r = 2;
  for(int y=r; y<h-r; ++y){
    for(int x=r; x<w-r; ++x){
      double Sxx=0, Sxy=0, Syy=0;
      for(int yy=y-r; yy<=y+r; ++yy){
        for(int xx=x-r; xx<=x+r; ++xx){
          const double gx = gradx(xx,yy);
          const double gy = grady(xx,yy);
          Sxx += gx*gx;
          Sxy += gx*gy;
          Syy += gy*gy;
        }
      }
      // min eigenvalue of 2x2: (tr - sqrt(tr^2 - 4 det))/2
      const double tr = Sxx + Syy;
      const double det = Sxx*Syy - Sxy*Sxy;
      const double disc = std::max(0.0, tr*tr - 4.0*det);
      const double lmin = 0.5*(tr - std::sqrt(disc));
      score[(size_t)y*w + x] = lmin;
    }
  }

  const double maxv = *std::max_element(score.begin(), score.end());
  const double thr = maxv * quality;

  struct Cand { int x,y; double s; };
  std::vector<Cand> cands;
  cands.reserve((size_t)w*h/50);
  for(int y=0;y<h;y++){
    for(int x=0;x<w;x++){
      const double s = score[(size_t)y*w + x];
      if (s >= thr) cands.push_back({x,y,s});
    }
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.s > b.s; });

  std::vector<Vec2> out;
  out.reserve((size_t)max_corners);
  for(const auto& c : cands){
    bool ok=true;
    for(const auto& p : out){
      const double dx=p.x - c.x;
      const double dy=p.y - c.y;
      if (dx*dx + dy*dy < (double)min_dist*min_dist){ ok=false; break; }
    }
    if(!ok) continue;
    out.push_back(Vec2{double(c.x), double(c.y)});
    if ((int)out.size() >= max_corners) break;
  }
  return out;
}

// ----------------------------
// Lucas-Kanade tracking (pyramidal, forward additive, small patch)
// ----------------------------
struct LKConfig {
  int max_tracks=2200;
  int min_tracks=900;
  double quality=0.01;
  int min_distance=8;
  int pyr_levels=3;
  int win_radius=5;
  int iters=10;
  double fb_thresh=1.0;
};

struct Track {
  int id;
  Vec2 p;
};

class KLTTracker {
public:
  explicit KLTTracker(LKConfig cfg): cfg_(cfg) {}

  void reset(const GrayImage& gray){
    prev_ = gray;
    tracks_.clear();
    const auto pts = shi_tomasi(gray, cfg_.max_tracks, cfg_.quality, cfg_.min_distance);
    for(const auto& p : pts) tracks_.push_back({next_id_++, p});
  }

  struct StepOut {
    std::vector<Vec2> prev_pts;
    std::vector<Vec2> cur_pts;
    std::vector<int> ids;
  };

  StepOut step(const GrayImage& gray){
    if(prev_.w == 0 || tracks_.empty()){
      reset(gray);
      return {};
    }
    Pyramid pyr0 = build_pyr(prev_, cfg_.pyr_levels);
    Pyramid pyr1 = build_pyr(gray,  cfg_.pyr_levels);

    std::vector<Track> kept;
    kept.reserve(tracks_.size());

    StepOut out;
    out.prev_pts.reserve(tracks_.size());
    out.cur_pts.reserve(tracks_.size());
    out.ids.reserve(tracks_.size());

    for(const auto& tr : tracks_){
      const auto p0 = tr.p;
      const auto p1_opt = track_one(pyr0, pyr1, p0);
      // forward-backward
      const auto p0_back = track_one(pyr1, pyr0, p1_opt);
      const double fb = std::hypot(p0_back.x - p0.x, p0_back.y - p0.y);
      if (fb >= cfg_.fb_thresh) continue;

      kept.push_back({tr.id, p1_opt});
      out.prev_pts.push_back(p0);
      out.cur_pts.push_back(p1_opt);
      out.ids.push_back(tr.id);
    }

    prev_ = gray;
    tracks_ = std::move(kept);

    // replenish
    if((int)tracks_.size() < cfg_.min_tracks){
      const int need = cfg_.max_tracks - (int)tracks_.size();
      // build simple mask by min-distance to existing tracks
      auto pts = shi_tomasi(gray, need*3, cfg_.quality, cfg_.min_distance);
      for(const auto& p : pts){
        bool ok=true;
        for(const auto& t : tracks_){
          const double dx=t.p.x - p.x;
          const double dy=t.p.y - p.y;
          if (dx*dx + dy*dy < (double)cfg_.min_distance*cfg_.min_distance){ ok=false; break; }
        }
        if(!ok) continue;
        tracks_.push_back({next_id_++, p});
        if ((int)tracks_.size() >= cfg_.max_tracks) break;
      }
    }
    return out;
  }

  const std::vector<Track>& tracks() const { return tracks_; }

  // Expose a minimal point tracker for loop-closure verification (still no external deps).
  Vec2 track_one_public(const Pyramid& a, const Pyramid& b, Vec2 p0) const {
    return track_one(a, b, p0);
  }

private:
  // Track one point using pyramid LK
  Vec2 track_one(const Pyramid& a, const Pyramid& b, Vec2 p0) const {
    Vec2 p = p0;
    // scale down to coarsest level
    for(int l=cfg_.pyr_levels-1; l>=0; --l){
      const double scale = 1.0 / (1<<l);
      Vec2 pl = {p.x*scale, p.y*scale};
      Vec2 dl = {0,0};
      // run LK at this level
      const GrayImage& I0 = a.lvl[l];
      const GrayImage& I1 = b.lvl[l];
      for(int it=0; it<cfg_.iters; ++it){
        const auto step = lk_step(I0, I1, pl + dl);
        dl.x += step.x;
        dl.y += step.y;
        if (std::hypot(step.x, step.y) < 1e-3) break;
      }
      // update p at full scale
      p = { (pl.x + dl.x) * (1<<l), (pl.y + dl.y) * (1<<l) };
    }
    return p;
  }

  Vec2 lk_step(const GrayImage& I0, const GrayImage& I1, Vec2 p1) const {
    // compute normal equations A dp = b for patch around p1 comparing I0 at p0 and I1 at p1
    // Here we do a simplified "forward additive" assuming p0 ~ p1 at this level.
    const int r = cfg_.win_radius;
    const double x = p1.x, y = p1.y;

    double A00=0, A01=0, A11=0;
    double b0=0, b1=0;

    for(int dy=-r; dy<=r; ++dy){
      for(int dx=-r; dx<=r; ++dx){
        const double xx = x + dx;
        const double yy = y + dy;
        // gradients on I1 (current)
        const double Ix = 0.5*(sample_bilinear(I1, xx+1, yy) - sample_bilinear(I1, xx-1, yy));
        const double Iy = 0.5*(sample_bilinear(I1, xx, yy+1) - sample_bilinear(I1, xx, yy-1));
        const double Iref = sample_bilinear(I0, xx, yy);
        const double Icur = sample_bilinear(I1, xx, yy);
        const double err = (Iref - Icur);

        A00 += Ix*Ix;
        A01 += Ix*Iy;
        A11 += Iy*Iy;
        b0  += Ix*err;
        b1  += Iy*err;
      }
    }
    const double detA = A00*A11 - A01*A01;
    if (std::fabs(detA) < 1e-9) return {0,0};
    const double inv00 =  A11/detA;
    const double inv01 = -A01/detA;
    const double inv11 =  A00/detA;

    const double dp0 = inv00*b0 + inv01*b1;
    const double dp1 = inv01*b0 + inv11*b1;
    return {dp0, dp1};
  }

  LKConfig cfg_;
  GrayImage prev_{};
  std::vector<Track> tracks_{};
  int next_id_=0;
};

// ----------------------------
// Geometry: normalize by K, 8-point + RANSAC, pose recovery
// ----------------------------
static Mat33 invert_K(const Mat33& K){
  // explicit inverse for 3x3
  const double d = sfm::det(K);
  if (std::fabs(d) < 1e-12) throw std::runtime_error("Singular K");
  Mat33 inv{};
  inv(0,0) =  (K(1,1)*K(2,2)-K(1,2)*K(2,1))/d;
  inv(0,1) = -(K(0,1)*K(2,2)-K(0,2)*K(2,1))/d;
  inv(0,2) =  (K(0,1)*K(1,2)-K(0,2)*K(1,1))/d;
  inv(1,0) = -(K(1,0)*K(2,2)-K(1,2)*K(2,0))/d;
  inv(1,1) =  (K(0,0)*K(2,2)-K(0,2)*K(2,0))/d;
  inv(1,2) = -(K(0,0)*K(1,2)-K(0,2)*K(1,0))/d;
  inv(2,0) =  (K(1,0)*K(2,1)-K(1,1)*K(2,0))/d;
  inv(2,1) = -(K(0,0)*K(2,1)-K(0,1)*K(2,0))/d;
  inv(2,2) =  (K(0,0)*K(1,1)-K(0,1)*K(1,0))/d;
  return inv;
}

static Vec3 homog(const Vec2& p){ return {p.x, p.y, 1.0}; }

static Vec2 project_pix(const Mat33& K, const Vec3& Xc){
  const double x = Xc.x / Xc.z;
  const double y = Xc.y / Xc.z;
  const double u = K(0,0)*x + K(0,2);
  const double v = K(1,1)*y + K(1,2);
  return {u, v};
}

static Vec2 norm_point(const Mat33& Kinv, const Vec2& p){
  const Vec3 hp = Kinv * homog(p);
  return {hp.x/hp.z, hp.y/hp.z};
}

static std::vector<double> AtA_from_A(const std::vector<double>& A, int rows, int cols){
  // A row-major rows x cols. Return cols x cols symmetric (row-major).
  std::vector<double> M((size_t)cols*(size_t)cols, 0.0);
  for(int i=0;i<cols;i++){
    for(int j=i;j<cols;j++){
      double s=0;
      for(int r=0;r<rows;r++){
        s += A[(size_t)r*cols + i] * A[(size_t)r*cols + j];
      }
      M[(size_t)i*cols + j] = s;
      M[(size_t)j*cols + i] = s;
    }
  }
  return M;
}

static Mat33 vec9_to_mat33(const std::array<double,9>& e){
  Mat33 E{};
  for(int r=0;r<3;r++) for(int c=0;c<3;c++) E(r,c) = e[3*r+c];
  return E;
}

static std::array<double,9> mat33_to_vec9(const Mat33& E){
  std::array<double,9> v{};
  for(int r=0;r<3;r++) for(int c=0;c<3;c++) v[3*r+c]=E(r,c);
  return v;
}

struct SVD3 {
  Mat33 U;
  std::array<double,3> s{};
  Mat33 V;
};

static SVD3 svd3(const Mat33& A){
  // Use eigen of A^T A to get V and singular values, then U = A V S^-1.
  const Mat33 At = sfm::transpose(A);
  // Build AtA (symmetric)
  std::vector<double> AtA(9,0.0);
  for(int r=0;r<3;r++){
    for(int c=0;c<3;c++){
      double s=0;
      for(int k=0;k<3;k++) s += At(r,k)*A(k,c);
      AtA[(size_t)r*3+c]=s;
    }
  }
  auto eig = sfm::jacobi_eig_sym(AtA, 3, 80);
  // eig.w ascending
  std::array<double,3> svals{
    std::sqrt(std::max(0.0, eig.w[0])),
    std::sqrt(std::max(0.0, eig.w[1])),
    std::sqrt(std::max(0.0, eig.w[2]))
  };
  // V columns are eigenvectors
  Mat33 V{};
  for(int r=0;r<3;r++) for(int c=0;c<3;c++) V(r,c)=eig.V[(size_t)r*3+c];

  // sort descending by singular value
  std::array<int,3> ord{0,1,2};
  std::sort(ord.begin(), ord.end(), [&](int i,int j){ return svals[i] > svals[j]; });

  Mat33 Vd{};
  std::array<double,3> sd{};
  for(int cc=0; cc<3; ++cc){
    sd[cc] = svals[ord[cc]];
    for(int r=0;r<3;r++) Vd(r,cc)=V(r, ord[cc]);
  }
  V = Vd;

  Mat33 U{};
  for(int c=0;c<3;c++){
    const double sc = sd[c];
    Vec3 vc{V(0,c), V(1,c), V(2,c)};
    Vec3 u = A * vc;
    if (sc > 1e-12) u = u / sc;
    else u = sfm::unit(u);
    U(0,c)=u.x; U(1,c)=u.y; U(2,c)=u.z;
  }
  // Orthonormalize U columns (Gram-Schmidt)
  Vec3 u0{U(0,0),U(1,0),U(2,0)};
  Vec3 u1{U(0,1),U(1,1),U(2,1)};
  Vec3 u2{U(0,2),U(1,2),U(2,2)};
  u0 = sfm::unit(u0);
  u1 = u1 - sfm::dot(u0,u1)*u0; u1 = sfm::unit(u1);
  u2 = sfm::cross(u0,u1); u2 = sfm::unit(u2);
  U(0,0)=u0.x; U(1,0)=u0.y; U(2,0)=u0.z;
  U(0,1)=u1.x; U(1,1)=u1.y; U(2,1)=u1.z;
  U(0,2)=u2.x; U(1,2)=u2.y; U(2,2)=u2.z;

  return {U, sd, V};
}

static Mat33 enforce_rank2(const Mat33& E){
  auto svd = svd3(E);
  // set smallest singular to 0
  const double s1=svd.s[0], s2=svd.s[1];
  Mat33 S{};
  S(0,0)=s1; S(1,1)=s2; S(2,2)=0.0;

  const Mat33 Ut = sfm::transpose(svd.U);
  // E2 = U S V^T
  Mat33 US = svd.U * S;
  Mat33 Vt = sfm::transpose(svd.V);
  return US * Vt;
}

static Mat33 eight_point_E(const std::vector<Vec2>& xn, const std::vector<Vec2>& yn, const std::vector<int>& idx8){
  // Build A (8x9) row-major
  std::vector<double> A(8*9, 0.0);
  for(int r=0;r<8;r++){
    const int i = idx8[r];
    const double x=xn[i].x, y=xn[i].y;
    const double xp=yn[i].x, yp=yn[i].y;
    const double row[9] = {xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1.0};
    for(int c=0;c<9;c++) A[(size_t)r*9 + c]=row[c];
  }
  // Solve min ||A e|| with ||e||=1 => smallest eigenvector of AtA (9x9)
  const auto AtA = AtA_from_A(A, 8, 9);
  auto eig = sfm::jacobi_eig_sym(AtA, 9, 120); // ascending
  // smallest eigenvector is column 0
  std::array<double,9> e{};
  for(int r=0;r<9;r++) e[r]=eig.V[(size_t)r*9 + 0];
  Mat33 E = vec9_to_mat33(e);
  return enforce_rank2(E);
}

static double sampson_err(const Mat33& E, const Vec2& x, const Vec2& xp){
  // x, xp are normalized (x, y)
  const Vec3 xh{x.x, x.y, 1.0};
  const Vec3 xph{xp.x, xp.y, 1.0};
  const Vec3 Ex = E * xh;
  const Vec3 Etxp = sfm::transpose(E) * xph;
  const double xpxEx = sfm::dot(xph, Ex);
  const double denom = Ex.x*Ex.x + Ex.y*Ex.y + Etxp.x*Etxp.x + Etxp.y*Etxp.y + 1e-12;
  return (xpxEx*xpxEx) / denom;
}

struct RelPose {
  Mat33 R_ji;
  Vec3 t_ji;
  std::vector<int> inliers;
};

static std::optional<RelPose> find_E_ransac(const Mat33& K, const std::vector<Vec2>& pi, const std::vector<Vec2>& pj,
                                            int iters=2000, double thr=1e-4, int min_inliers=80){
  if(pi.size() < 8) return std::nullopt;
  const Mat33 Kinv = invert_K(K);

  std::vector<Vec2> xi(pi.size()), xj(pj.size());
  for(size_t i=0;i<pi.size();++i){
    xi[i] = norm_point(Kinv, pi[i]);
    xj[i] = norm_point(Kinv, pj[i]);
  }

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> uni(0, (int)pi.size()-1);

  Mat33 bestE{};
  std::vector<int> best_inl;

  std::vector<int> idx8(8);
  for(int it=0; it<iters; ++it){
    for(int k=0;k<8;k++) idx8[k] = uni(rng);
    const Mat33 E = eight_point_E(xi, xj, idx8);
    std::vector<int> inl;
    inl.reserve(pi.size());
    for(int i=0;i<(int)pi.size();++i){
      const double e = sampson_err(E, xi[i], xj[i]);
      if (e < thr) inl.push_back(i);
    }
    if ((int)inl.size() > (int)best_inl.size()){
      best_inl = std::move(inl);
      bestE = E;
    }
  }
  if((int)best_inl.size() < min_inliers) return std::nullopt;

  // Decompose E -> R,t
  const auto svd = svd3(bestE);
  // W
  Mat33 W{};
  W(0,1)=-1; W(1,0)=1; W(2,2)=1;
  Mat33 Vt = sfm::transpose(svd.V);
  Mat33 U = svd.U;

  Mat33 R1 = U * W * Vt;
  Mat33 R2 = U * sfm::transpose(W) * Vt;

  // ensure det +1
  if (sfm::det(R1) < 0) { for(int r=0;r<3;r++) for(int c=0;c<3;c++) R1(r,c) = -R1(r,c); }
  if (sfm::det(R2) < 0) { for(int r=0;r<3;r++) for(int c=0;c<3;c++) R2(r,c) = -R2(r,c); }

  Vec3 t = {U(0,2), U(1,2), U(2,2)};
  t = sfm::unit(t);

  // choose the correct among 4 by cheirality test on a handful
  auto triangulate = [&](const Mat33& R, const Vec3& t, const Vec2& x, const Vec2& xp)->Vec3{
    // normalized coordinates, P1=[I|0], P2=[R|t]
    // Build A 4x4 for DLT, solve smallest eigenvector of AtA
    double A[16]{};
    // rows:
    // x * P1_3 - P1_1, y * P1_3 - P1_2
    // xp * P2_3 - P2_1, yp * P2_3 - P2_2
    // P1 = [I|0]
    auto setrow = [&](int r, double a0,double a1,double a2,double a3){
      A[r*4+0]=a0; A[r*4+1]=a1; A[r*4+2]=a2; A[r*4+3]=a3;
    };
    setrow(0, -1, 0, x.x, 0);
    setrow(1, 0, -1, x.y, 0);
    // P2 rows
    // P2 row1 = [R00 R01 R02 t0]
    // row2 = [R10 R11 R12 t1]
    // row3 = [R20 R21 R22 t2]
    setrow(2, xp.x*R(2,0)-R(0,0), xp.x*R(2,1)-R(0,1), xp.x*R(2,2)-R(0,2), xp.x*t.z - t.x);
    setrow(3, xp.y*R(2,0)-R(1,0), xp.y*R(2,1)-R(1,1), xp.y*R(2,2)-R(1,2), xp.y*t.z - t.y);
    // Note: above is a simplified form; for robustness we use AtA eigenvector.
    // Build AtA 4x4
    std::vector<double> Ar(A, A+16);
    const auto AtA = AtA_from_A(Ar, 4, 4);
    auto eig = sfm::jacobi_eig_sym(AtA, 4, 80);
    // smallest eigenvector column 0
    std::array<double,4> Xh{};
    for(int r=0;r<4;r++) Xh[r]=eig.V[(size_t)r*4 + 0];
    const double w = Xh[3];
    return {Xh[0]/w, Xh[1]/w, Xh[2]/w};
  };

  auto count_cheirality = [&](const Mat33& R, const Vec3& t)->int{
    int ok=0;
    const int M = std::min((int)best_inl.size(), 20);
    for(int k=0;k<M;k++){
      const int i = best_inl[k];
      const Vec2 x = xi[i];
      const Vec2 xp= xj[i];
      const Vec3 X = triangulate(R,t,x,xp);
      // depth in cam1 (Z = X.z)
      const double z1 = X.z;
      // depth in cam2: X2 = R X + t
      const Vec3 X2 = (R * X) + t;
      const double z2 = X2.z;
      if (z1 > 0 && z2 > 0) ok++;
    }
    return ok;
  };

  struct Cand { Mat33 R; Vec3 t; };
  std::array<Cand,4> cands{ Cand{R1, t}, Cand{R1, {-t.x,-t.y,-t.z}}, Cand{R2, t}, Cand{R2, {-t.x,-t.y,-t.z}} };
  int best=0, bestok=-1;
  for(int i=0;i<4;i++){
    const int ok = count_cheirality(cands[i].R, cands[i].t);
    if(ok > bestok){ bestok=ok; best=i; }
  }

  RelPose rp;
  rp.R_ji = cands[best].R;
  rp.t_ji = cands[best].t;
  rp.inliers = best_inl;
  return rp;
}

// ----------------------------
// Simple keyframe/map structures
// ----------------------------
struct Keyframe {
  int kf_id{};
  int frame_idx{};
  std::string img_name;
  PoseCW pose;
  std::unordered_map<int, Vec2> obs; // track_id -> pixel
};

struct MapPoint {
  int pid{};
  int tid{};
  Vec3 Xw{};
  std::vector<std::pair<int, Vec2>> obs;
};

struct MapState {
  int next_pid=0;
  std::unordered_map<int,int> tid2pid;
  std::unordered_map<int, MapPoint> pts;
  bool has(int tid) const { return tid2pid.find(tid) != tid2pid.end(); }
  int add(int tid, Vec3 Xw){
    int pid = next_pid++;
    MapPoint mp; mp.pid=pid; mp.tid=tid; mp.Xw=Xw;
    pts.emplace(pid, mp);
    tid2pid.emplace(tid, pid);
    return pid;
  }
  void add_obs(int tid, int kf_id, Vec2 uv){
    auto it = tid2pid.find(tid);
    if(it==tid2pid.end()) return;
    pts[it->second].obs.push_back({kf_id, uv});
  }
};

// ----------------------------
// Local bundle adjustment (sliding window) + translation pose-graph optimization.
// These blocks mirror the Python pipeline stages:
//  - Local BA: refine last W keyframes + active points (robust reprojection error).
//  - Loop closure: detect loop candidates via global image descriptor, then verify with LK+E.
//  - Pose graph: optimize keyframe camera centers using translation constraints from edges.
//
// Notes:
//  - Monocular scale is not observable; ATE is typically evaluated after Sim(3) alignment.
//  - This implementation remains dependency-free (no OpenCV, no Eigen).

struct BAConfig {
  int window = 6;
  int iters = 5;
  int max_points = 600;
  double huber_delta = 3.0;   // pixels
  double lambda = 1e-3;       // LM damping
};

struct PGEdge {
  int i = -1;
  int j = -1;
  Mat33 R_ji{};   // i->j rotation (from Essential matrix decomposition)
  Vec3  t_ji{};   // i->j translation (unit, up to sign/scale)
  int inliers = 0;
  bool is_loop = false;
};

static inline Mat33 cross_xc(const Vec3& Xc){
  // [Xc]x
  Mat33 W{};
  W(0,0)=0;      W(0,1)=-Xc.z; W(0,2)= Xc.y;
  W(1,0)=Xc.z;   W(1,1)=0;     W(1,2)=-Xc.x;
  W(2,0)=-Xc.y;  W(2,1)=Xc.x;  W(2,2)=0;
  return W;
}

static inline Vec2 project_K(const Mat33& K, const Vec3& Xc){
  const double x = Xc.x / Xc.z;
  const double y = Xc.y / Xc.z;
  return { K(0,0)*x + K(0,2), K(1,1)*y + K(1,2) };
}

static inline double huber_w(double r_norm, double delta){
  if(r_norm <= delta) return 1.0;
  return delta / (r_norm + 1e-12);
}

static void bundle_adjust_window(const Mat33& K,
                                 std::vector<Keyframe>& kfs,
                                 MapState& map,
                                 const BAConfig& cfg)
{
  const int N = (int)kfs.size();
  if(N < 2) return;
  const int w0 = std::max(0, N - cfg.window);
  const int W = N - w0;
  if(W < 2) return;

  // map kf_id -> local window index
  std::unordered_map<int,int> kf2local;
  kf2local.reserve((size_t)W);
  for(int li=0; li<W; ++li) kf2local.emplace(kfs[w0+li].kf_id, li);

  struct Obs { int li; Vec2 uv; };
  struct LocalPoint { int pid; Vec3* Xw; std::vector<Obs> obs; };

  std::vector<LocalPoint> pts;
  pts.reserve((size_t)cfg.max_points);

  // Collect points that have >=2 observations in the window.
  for(auto& [pid, mp] : map.pts){
    std::vector<Obs> o;
    o.reserve(mp.obs.size());
    for(const auto& [kf_id, uv] : mp.obs){
      auto it = kf2local.find(kf_id);
      if(it == kf2local.end()) continue;
      o.push_back({it->second, uv});
    }
    if((int)o.size() < 2) continue;
    pts.push_back({pid, &mp.Xw, std::move(o)});
    if((int)pts.size() >= cfg.max_points) break;
  }
  if(pts.empty()) return;

  // Helper: extract world->camera pose for each keyframe in the window.
  auto get_wc = [&](int li){
    return kfs[w0+li].pose.inv_wc(); // (Rwc, twc)
  };

  const int P = W;
  const int D = 6*P;

  for(int it=0; it<cfg.iters; ++it){
    sfm::DMat S(D, D, 0.0);
    sfm::DVec b(D, 0.0);

    // Accumulate Schur complement over points.
    for(const auto& lp : pts){
      double Hpp[9] = {0,0,0, 0,0,0, 0,0,0};
      double bp[3]  = {0,0,0};

      // Per-pose accumulators for this point.
      struct PoseAcc {
        int li=-1;
        double Hxx[36] = {0}; // 6x6
        double bx[6]   = {0};
        double Hxp[18] = {0}; // 6x3
      };
      std::array<PoseAcc, 16> accs; // typical obs count per point is small
      int acc_n = 0;

      auto* Xw = lp.Xw;

      auto& obs = lp.obs;
      if(obs.size() > accs.size()){
        // fallback: ignore excessively connected points
        continue;
      }

      for(const auto& ob : obs){
        const int li = ob.li;

        // fetch or create pose accumulator
        int ai = -1;
        for(int k=0;k<acc_n;k++) if(accs[k].li == li){ ai=k; break; }
        if(ai<0){
          ai = acc_n++;
          accs[ai].li = li;
        }

        const auto [Rwc, twc] = get_wc(li);
        const Vec3 Xc = (Rwc * (*Xw)) + twc;
        if(Xc.z <= 1e-6) continue;

        const Vec2 uv_hat = project_K(K, Xc);
        const Vec2 r2 = ob.uv - uv_hat;
        const double rnorm = std::hypot(r2.x, r2.y);
        const double w = huber_w(rnorm, cfg.huber_delta);

        // J_proj: 2x3
        const double fx = K(0,0), fy = K(1,1);
        const double invz = 1.0 / Xc.z;
        const double invz2 = invz*invz;
        const double Jproj[6] = {
          fx*invz, 0.0, -fx*Xc.x*invz2,
          0.0, fy*invz, -fy*Xc.y*invz2
        };

        // J_point = Jproj * Rwc (2x3)
        double Jp[6] = {0,0,0, 0,0,0};
        for(int row=0; row<2; ++row){
          for(int c=0; c<3; ++c){
            const double a0 = Jproj[row*3+0]*Rwc(0,c);
            const double a1 = Jproj[row*3+1]*Rwc(1,c);
            const double a2 = Jproj[row*3+2]*Rwc(2,c);
            Jp[row*3+c] = a0+a1+a2;
          }
        }

        // dXc/dw = -[Xc]x, dXc/dt = I
        const Mat33 Xx = cross_xc(Xc);
        // J_rot = Jproj * ( -Xx )  -> 2x3
        double Jr[6] = {0,0,0, 0,0,0};
        for(int row=0; row<2; ++row){
          for(int c=0; c<3; ++c){
            const double a0 = -Jproj[row*3+0]*Xx(0,c);
            const double a1 = -Jproj[row*3+1]*Xx(1,c);
            const double a2 = -Jproj[row*3+2]*Xx(2,c);
            Jr[row*3+c] = a0+a1+a2;
          }
        }
        // J_pose = [Jr | Jt], where Jt = Jproj (2x3)
        double Jx[12] = {
          Jr[0], Jr[1], Jr[2],  Jproj[0], Jproj[1], Jproj[2],
          Jr[3], Jr[4], Jr[5],  Jproj[3], Jproj[4], Jproj[5]
        };

        // Accumulate Hpp, bp
        for(int a=0;a<3;a++){
          for(int c=0;c<3;c++){
            double s=0;
            for(int k=0;k<2;k++) s += Jp[k*3+a]*Jp[k*3+c];
            Hpp[a*3+c] += w*s;
          }
          double sb=0;
          for(int k=0;k<2;k++) sb += Jp[k*3+a]*((k==0)?r2.x:r2.y);
          bp[a] += w*sb;
        }

        // Hxx (6x6), bx (6), Hxp (6x3)
        auto& A = accs[ai];
        for(int a=0;a<6;a++){
          for(int c=0;c<6;c++){
            double s=0;
            for(int k=0;k<2;k++) s += Jx[k*6+a]*Jx[k*6+c];
            A.Hxx[a*6+c] += w*s;
          }
          double sb=0;
          for(int k=0;k<2;k++) sb += Jx[k*6+a]*((k==0)?r2.x:r2.y);
          A.bx[a] += w*sb;
        }
        for(int a=0;a<6;a++){
          for(int c=0;c<3;c++){
            double s=0;
            for(int k=0;k<2;k++) s += Jx[k*6+a]*Jp[k*3+c];
            A.Hxp[a*3+c] += w*s;
          }
        }
      } // obs

      double invHpp[9];
      if(!sfm::inv3(Hpp, invHpp)) continue;

      // Add direct pose terms (Hxx, bx) to global S and b.
      for(int k=0;k<acc_n;k++){
        const int li = accs[k].li;
        sfm::add_block(S, 6*li, 6*li, accs[k].Hxx, 6, 6);
        sfm::add_vec(b, 6*li, accs[k].bx, 6);
      }

      // Schur elimination: S -= Hxp * inv(Hpp) * Hpx, b -= Hxp * inv(Hpp) * bp
      // Precompute Gi = Hxp_i * invHpp (6x3)
      double G[16][18]; // up to 16 poses per point
      for(int k=0;k<acc_n;k++){
        const double* Hxp = accs[k].Hxp; // 6x3
        double* Gi = G[k];
        for(int r=0;r<6;r++){
          for(int c=0;c<3;c++){
            Gi[r*3+c] = Hxp[r*3+0]*invHpp[0*3+c] + Hxp[r*3+1]*invHpp[1*3+c] + Hxp[r*3+2]*invHpp[2*3+c];
          }
        }
      }

      for(int a=0;a<acc_n;a++){
        const int li = accs[a].li;
        // b term
        double tmp[6] = {0,0,0,0,0,0};
        for(int r=0;r<6;r++){
          tmp[r] = G[a][r*3+0]*bp[0] + G[a][r*3+1]*bp[1] + G[a][r*3+2]*bp[2];
        }
        for(int r=0;r<6;r++) b[6*li + r] -= tmp[r];

        for(int bb=0; bb<acc_n; bb++){
          const int lj = accs[bb].li;
          double blk[36] = {0};
          for(int r=0;r<6;r++){
            for(int c=0;c<6;c++){
              // Gi (6x3) * Hxp_j^T (3x6)
              blk[r*6+c] =
                G[a][r*3+0]*accs[bb].Hxp[c*3+0] +
                G[a][r*3+1]*accs[bb].Hxp[c*3+1] +
                G[a][r*3+2]*accs[bb].Hxp[c*3+2];
            }
          }
          sfm::add_block(S, 6*li, 6*lj, blk, 6, 6);
        }
      }

      // Point update (Gauss-Newton back-substitution) is optional for stability.
      // We update points after solving pose increments below.
    } // points

    // Damping
    for(int i=0;i<D;i++) S(i,i) += cfg.lambda;

    // Fix the first pose in the window (gauge).
    for(int d=0; d<6; d++){
      const int ii = d;
      S(ii,ii) += 1e9;
      b[ii] = 0.0;
    }

    sfm::DVec dx;
    try {
      dx = sfm::solve_gauss(S, b);
    } catch(...) {
      return; // ill-conditioned; skip BA
    }

    // Apply pose updates (left-multiply on world->camera).
    for(int li=1; li<W; ++li){
      Vec3 w = { dx[6*li+0], dx[6*li+1], dx[6*li+2] };
      Vec3 v = { dx[6*li+3], dx[6*li+4], dx[6*li+5] };

      auto [Rwc, twc] = kfs[w0+li].pose.inv_wc();
      const Mat33 dR = sfm::so3_exp(w);
      const Mat33 Rwc2 = dR * Rwc;
      const Vec3  twc2 = twc + v;

      const Mat33 Rcw2 = sfm::transpose(Rwc2);
      const Vec3  Cw2  = -(Rcw2 * twc2);

      kfs[w0+li].pose.R = Rcw2;
      kfs[w0+li].pose.t = Cw2;
    }
  } // iters
}

// Global descriptor for loop candidate search: 32x32 downsample, mean-removed, L2-normalized.
static std::vector<float> global_desc_32(const GrayImage& im){
  GrayImage d = im;
  while(d.w > 32 || d.h > 32) d = downsample2(d);
  // If not exactly 32x32, sample by nearest.
  std::vector<float> v;
  v.reserve(32*32);
  double mean=0.0;
  for(int y=0;y<32;y++){
    for(int x=0;x<32;x++){
      const int sx = std::min(d.w-1, (int)std::round((double)x*(d.w-1)/31.0));
      const int sy = std::min(d.h-1, (int)std::round((double)y*(d.h-1)/31.0));
      const float val = (float)d.at(sx, sy);
      v.push_back(val);
      mean += val;
    }
  }
  mean /= (32.0*32.0);
  double n2=0.0;
  for(auto& x : v){ x = (float)(x - (float)mean); n2 += (double)x*(double)x; }
  const double invn = 1.0 / std::sqrt(n2 + 1e-12);
  for(auto& x : v) x = (float)(x * invn);
  return v;
}

static float dot_desc(const std::vector<float>& a, const std::vector<float>& b){
  float s=0.0f;
  const size_t n = std::min(a.size(), b.size());
  for(size_t i=0;i<n;i++) s += a[i]*b[i];
  return s;
}

static bool posegraph_optimize_centers(std::vector<Keyframe>& kfs, const std::vector<PGEdge>& edges){
  const int N = (int)kfs.size();
  if(N < 2) return false;
  if(edges.empty()) return false;
  const int D = 3*N;
  sfm::DMat H(D, D, 0.0);
  sfm::DVec g(D, 0.0);

  auto add_I = [&](int a, int b, double s){
    // add s*I3 to block (a,b)
    for(int d=0; d<3; d++) H(3*a+d, 3*b+d) += s;
  };

  for(const auto& e : edges){
    if(e.i < 0 || e.j < 0 || e.i >= N || e.j >= N) continue;

    const Vec3 Ci = kfs[e.i].pose.t;
    const Vec3 Cj = kfs[e.j].pose.t;
    const Vec3 d_est = Cj - Ci;

    // Convert measured translation direction into world direction.
    // We use t_delta = -(R_ji^T t_ji) as the i-frame translation direction.
    const Vec3 t_delta = -(sfm::transpose(e.R_ji) * e.t_ji);
    const Vec3 dir_w = unit(kfs[e.i].pose.R * t_delta);

    const double L = std::max(1e-6, norm(d_est));
    const Vec3 d_meas = dir_w * L;

    const Vec3 r = (Cj - Ci) - d_meas; // 3x1 residual
    const double w = e.is_loop ? 2.0 : 1.0; // slightly stronger loop edges

    // J_i = -I, J_j = +I
    add_I(e.i, e.i, w);
    add_I(e.j, e.j, w);
    add_I(e.i, e.j, -w);
    add_I(e.j, e.i, -w);

    // g = J^T r
    g[3*e.i+0] += w * (-r.x);
    g[3*e.i+1] += w * (-r.y);
    g[3*e.i+2] += w * (-r.z);

    g[3*e.j+0] += w * ( r.x);
    g[3*e.j+1] += w * ( r.y);
    g[3*e.j+2] += w * ( r.z);
  }

  // Fix node 0 (gauge)
  for(int d=0; d<3; d++){
    H(d,d) += 1e9;
    g[d] = 0.0;
  }

  sfm::DVec dc;
  try {
    dc = sfm::solve_gauss(H, g);
  } catch(...) {
    return false;
  }

  for(int i=1;i<N;i++){
    kfs[i].pose.t.x += dc[3*i+0];
    kfs[i].pose.t.y += dc[3*i+1];
    kfs[i].pose.t.z += dc[3*i+2];
  }
  return true;
}

static void write_posegraph_edges(const fs::path& path, const std::vector<PGEdge>& edges){
  std::ofstream f(path);
  f << "i,j,rvec_x,rvec_y,rvec_z,t_x,t_y,t_z,inliers,is_loop\n";
  for(const auto& e : edges){
    const Vec3 rvec = sfm::rodrigues_rvec(e.R_ji);
    f << e.i << "," << e.j << ","
      << rvec.x << "," << rvec.y << "," << rvec.z << ","
      << e.t_ji.x << "," << e.t_ji.y << "," << e.t_ji.z << ","
      << e.inliers << "," << (e.is_loop?1:0) << "\n";
  }
}


// ----------------------------
// Pipeline (KLT + E + triangulate + minimal exports)
// ----------------------------
static void write_ply_xyz(const fs::path& path, const std::vector<Vec3>& xyz){
  std::ofstream f(path);
  if(!f) throw std::runtime_error("Failed to write: " + path.string());
  f << "ply\nformat ascii 1.0\n";
  f << "element vertex " << xyz.size() << "\n";
  f << "property float x\nproperty float y\nproperty float z\nend_header\n";
  for(const auto& p : xyz){
    f << p.x << " " << p.y << " " << p.z << "\n";
  }
}

static void write_ply_mesh(const fs::path& path,
                           const std::vector<Vec3>& vertices,
                           const std::vector<std::array<int,3>>& faces){
  std::ofstream f(path);
  if(!f) throw std::runtime_error("Failed to write: " + path.string());
  f << "ply\nformat ascii 1.0\n";
  f << "element vertex " << vertices.size() << "\n";
  f << "property float x\nproperty float y\nproperty float z\n";
  f << "element face " << faces.size() << "\n";
  f << "property list uchar int vertex_indices\n";
  f << "end_header\n";
  for(const auto& p : vertices){
    f << p.x << " " << p.y << " " << p.z << "\n";
  }
  for(const auto& tri : faces){
    f << "3 " << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
  }
}

static inline double orient2d(const Vec2& a, const Vec2& b, const Vec2& c){
  // 2D cross product (b-a) x (c-a)
  return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

static inline bool in_circumcircle(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& p){
  // Robust-enough circumcircle test for Bowyer-Watson.
  // Uses determinant test; sign depends on triangle orientation.
  const double ax = a.x - p.x;
  const double ay = a.y - p.y;
  const double bx = b.x - p.x;
  const double by = b.y - p.y;
  const double cx = c.x - p.x;
  const double cy = c.y - p.y;

  const double a2 = ax*ax + ay*ay;
  const double b2 = bx*bx + by*by;
  const double c2 = cx*cx + cy*cy;

  const double det = a2*(bx*cy - by*cx) - b2*(ax*cy - ay*cx) + c2*(ax*by - ay*bx);
  const double o = orient2d(a,b,c);
  // For CCW triangles, det > 0 implies inside.
  return (o > 0.0) ? (det > 1e-12) : (det < -1e-12);
}

struct EdgeKey {
  int a, b;
  bool operator==(const EdgeKey& o) const { return a==o.a && b==o.b; }
};

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey& e) const {
    // Simple hash for pair of ints
    return (std::size_t)( (std::uint64_t)(std::uint32_t)e.a * 2654435761u ) ^ (std::size_t)(std::uint32_t)e.b;
  }
};

static std::vector<std::array<int,3>> delaunay_triangulate_2d(const std::vector<Vec2>& pts){
  // Bowyer-Watson Delaunay triangulation in 2D.
  // Returns triangles as index triples into pts.
  if(pts.size() < 3) return {};

  // Build working list with super-triangle.
  double minx = pts[0].x, maxx = pts[0].x, miny = pts[0].y, maxy = pts[0].y;
  for(const auto& p : pts){
    minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
    miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
  }
  const double dx = maxx - minx;
  const double dy = maxy - miny;
  const double delta = std::max(dx, dy);
  const double midx = 0.5*(minx + maxx);
  const double midy = 0.5*(miny + maxy);

  std::vector<Vec2> pwork = pts;
  const int i0 = (int)pwork.size();
  const int i1 = i0 + 1;
  const int i2 = i0 + 2;
  pwork.push_back(Vec2{midx - 20.0*delta, midy -  2.0*delta});
  pwork.push_back(Vec2{midx,            midy + 20.0*delta});
  pwork.push_back(Vec2{midx + 20.0*delta, midy -  2.0*delta});

  std::vector<std::array<int,3>> tris;
  // Ensure CCW ordering for the super triangle.
  if(orient2d(pwork[i0], pwork[i1], pwork[i2]) > 0.0) tris.push_back({i0,i1,i2});
  else tris.push_back({i0,i2,i1});

  for(int pi=0; pi<(int)pts.size(); ++pi){
    const Vec2& p = pwork[pi];

    std::vector<int> bad;
    bad.reserve(tris.size());
    for(int ti=0; ti<(int)tris.size(); ++ti){
      const auto& t = tris[ti];
      if(in_circumcircle(pwork[t[0]], pwork[t[1]], pwork[t[2]], p)) bad.push_back(ti);
    }

    // Boundary edges of the polygonal hole.
    std::unordered_map<EdgeKey,int,EdgeKeyHash> edge_count;
    edge_count.reserve((size_t)bad.size()*3);
    auto add_edge = [&](int a, int b){
      // Normalize edge key (undirected)
      EdgeKey ek{std::min(a,b), std::max(a,b)};
      edge_count[ek] += 1;
    };

    for(int idx : bad){
      const auto& t = tris[idx];
      add_edge(t[0], t[1]);
      add_edge(t[1], t[2]);
      add_edge(t[2], t[0]);
    }

    // Remove bad triangles (mark then compact).
    if(!bad.empty()){
      std::vector<char> keep(tris.size(), 1);
      for(int idx : bad) keep[idx] = 0;
      std::vector<std::array<int,3>> kept;
      kept.reserve(tris.size());
      for(size_t ti=0; ti<tris.size(); ++ti){
        if(keep[ti]) kept.push_back(tris[ti]);
      }
      tris.swap(kept);
    }

    // Re-triangulate the hole.
    for(const auto& kv : edge_count){
      if(kv.second != 1) continue; // only boundary edges
      const int a = kv.first.a;
      const int b = kv.first.b;
      // Create triangle (a,b,pi) with CCW ordering.
      if(orient2d(pwork[a], pwork[b], p) > 0.0) tris.push_back({a,b,pi});
      else tris.push_back({b,a,pi});
    }
  }

  // Remove triangles using super triangle vertices.
  std::vector<std::array<int,3>> out;
  out.reserve(tris.size());
  for(const auto& t : tris){
    if(t[0] >= (int)pts.size() || t[1] >= (int)pts.size() || t[2] >= (int)pts.size()) continue;
    out.push_back(t);
  }
  return out;
}

static bool project_world_to_image(const Mat33& K, const Keyframe& kf, const Vec3& Xw,
                                  int w, int h, Vec2& uv){
  auto [Rwc, twc] = kf.pose.inv_wc();
  const Vec3 Xc = (Rwc * Xw) + twc;
  if(!(Xc.z > 1e-8)) return false;
  const double xn = Xc.x / Xc.z;
  const double yn = Xc.y / Xc.z;
  const Vec3 ph = K * Vec3{xn, yn, 1.0};
  uv = Vec2{ph.x, ph.y};
  if(uv.x < 0.0 || uv.y < 0.0 || uv.x >= (double)w || uv.y >= (double)h) return false;
  return std::isfinite(uv.x) && std::isfinite(uv.y);
}

static void build_mesh_from_sparse_points(const Mat33& K,
                                          const Keyframe& kf,
                                          const std::unordered_map<int, MapPoint>& pts,
                                          int img_w,
                                          int img_h,
                                          int max_points,
                                          int grid_px,
                                          double max_edge_px,
                                          std::vector<Vec3>& out_vertices,
                                          std::vector<std::array<int,3>>& out_faces){
  // Project map points into the selected keyframe and run 2D Delaunay on pixel coords.
  // This is a standard, lightweight meshing approach often used on depth-map-like samples;
  // on sparse SfM points it yields a coarse surface suitable for visualization.
  struct Cand { Vec2 uv; Vec3 Xw; };
  std::vector<Cand> cands;
  cands.reserve(pts.size());
  for(const auto& kv : pts){
    Vec2 uv;
    if(!project_world_to_image(K, kf, kv.second.Xw, img_w, img_h, uv)) continue;
    cands.push_back(Cand{uv, kv.second.Xw});
  }

  if((int)cands.size() < 50){
    out_vertices.clear();
    out_faces.clear();
    return;
  }

  // Grid-based subsampling to avoid near-duplicate pixel locations.
  const int cell = std::max(1, grid_px);
  struct CellKey { int cx, cy; bool operator==(const CellKey& o) const { return cx==o.cx && cy==o.cy; } };
  struct CellHash { std::size_t operator()(const CellKey& k) const {
    return ((std::size_t)(std::uint32_t)k.cx * 73856093u) ^ ((std::size_t)(std::uint32_t)k.cy * 19349663u);
  }};
  std::unordered_set<CellKey, CellHash> used;
  used.reserve((size_t)max_points*2);

  std::mt19937 rng(42);
  std::shuffle(cands.begin(), cands.end(), rng);

  out_vertices.clear();
  std::vector<Vec2> uv_sel;
  out_vertices.reserve((size_t)max_points);
  uv_sel.reserve((size_t)max_points);

  for(const auto& c : cands){
    const int cx = (int)std::floor(c.uv.x / (double)cell);
    const int cy = (int)std::floor(c.uv.y / (double)cell);
    const CellKey ck{cx,cy};
    if(used.find(ck) != used.end()) continue;
    used.insert(ck);
    uv_sel.push_back(c.uv);
    out_vertices.push_back(c.Xw);
    if((int)out_vertices.size() >= max_points) break;
  }

  if((int)out_vertices.size() < 50){
    out_faces.clear();
    return;
  }

  auto faces = delaunay_triangulate_2d(uv_sel);

  // Filter triangles with very long edges in pixel space.
  out_faces.clear();
  out_faces.reserve(faces.size());
  for(const auto& t : faces){
    const Vec2& a = uv_sel[t[0]];
    const Vec2& b = uv_sel[t[1]];
    const Vec2& c = uv_sel[t[2]];
    const double dab = std::hypot(a.x-b.x, a.y-b.y);
    const double dbc = std::hypot(b.x-c.x, b.y-c.y);
    const double dca = std::hypot(c.x-a.x, c.y-a.y);
    const double dmax = std::max(dab, std::max(dbc, dca));
    if(dmax > max_edge_px) continue;
    out_faces.push_back(t);
  }
}

static void write_csv_centers(const fs::path& path, const std::vector<Keyframe>& kfs,
                              const std::unordered_map<std::string, MBAngle>& ang){
  std::ofstream f(path);
  f << "kf_id,frame_idx,image,x,y,z,lat,lon\n";
  for(const auto& kf : kfs){
    const auto it = ang.find(kf.img_name);
    const double lat = (it==ang.end()? 0.0 : it->second.lat);
    const double lon = (it==ang.end()? 0.0 : it->second.lon);
    f << kf.kf_id << "," << kf.frame_idx << "," << kf.img_name << ","
      << kf.pose.t.x << "," << kf.pose.t.y << "," << kf.pose.t.z << ","
      << lat << "," << lon << "\n";
  }
}

static Vec3 triangulate_dlt(const Mat33& K, const PoseCW& pose_i, const PoseCW& pose_j, Vec2 ui, Vec2 uj){
  // world->cam for each keyframe
  auto [Rwi, twi] = pose_i.inv_wc();
  auto [Rwj, twj] = pose_j.inv_wc();
  // We'll triangulate in world coordinates using normalized rays.
  // Convert pixels to normalized camera coords using K^-1.
  const Mat33 Kinv = invert_K(K);
  const Vec2 xi = norm_point(Kinv, ui);
  const Vec2 xj = norm_point(Kinv, uj);

  // Build P_i = [Rwi | twi], P_j = [Rwj | twj] in world->cam
  // DLT with 4 equations: x * P3 - P1, y*P3 - P2 for each camera.
  auto row = [&](const Mat33& R, const Vec3& t, double x, double y, int which)->std::array<double,4>{
    // which=0 => x*P3 - P1; which=1 => y*P3 - P2
    const int r0 = (which==0? 0:1);
    const double s = (which==0? x:y);
    return {
      s*R(2,0) - R(r0,0),
      s*R(2,1) - R(r0,1),
      s*R(2,2) - R(r0,2),
      s*t.z    - (r0==0? t.x : t.y)
    };
  };
  std::vector<double> A(16,0.0);
  auto r0 = row(Rwi,twi, xi.x, xi.y, 0);
  auto r1 = row(Rwi,twi, xi.x, xi.y, 1);
  auto r2 = row(Rwj,twj, xj.x, xj.y, 0);
  auto r3 = row(Rwj,twj, xj.x, xj.y, 1);
  auto set = [&](int r, const std::array<double,4>& rr){
    for(int c=0;c<4;c++) A[(size_t)r*4+c]=rr[c];
  };
  set(0,r0); set(1,r1); set(2,r2); set(3,r3);

  const auto AtA = AtA_from_A(A, 4, 4);
  auto eig = sfm::jacobi_eig_sym(AtA, 4, 80);
  std::array<double,4> Xh{};
  for(int r=0;r<4;r++) Xh[r]=eig.V[(size_t)r*4 + 0];
  const double w = Xh[3];
  return {Xh[0]/w, Xh[1]/w, Xh[2]/w};
}

int main(int argc, char** argv){
  try{
    if(argc < 3){
      std::cerr << "Usage: " << argv[0] << " <templering_root> <out_dir> [frames] [options]\n"
                << "Input must be PGM images (P5) in <templering_root>/templeRing_pgm/\n"
                << "and par/ang files in <templering_root>/templeRing/.\n\n"
                << "Options:\n"
                << "  --config <path>           Config JSON (defaults to ./config.json when present)\n"
                << "  --export-geometry <none|pointcloud|mesh|both>\n"
                << "      none: no .ply geometry outputs\n"
                << "      pointcloud: write templeRing_sparse_points.ply\n"
                << "      mesh: write templeRing_mesh_sparse_kf<k>.ply (2D Delaunay on projected sparse points)\n"
                << "      both: write both pointcloud and mesh\n"
                << "  --mesh-kf <k>            Keyframe index used for 2D projection (default 0)\n"
                << "  --mesh-max-points <n>    Max vertices in mesh (default 2500)\n"
                << "  --mesh-grid-px <px>      Pixel grid subsampling cell size (default 4)\n"
                << "  --mesh-max-edge-px <px>  Reject triangles with any edge longer than this (default 80)\n";
      return 2;
    }
    const fs::path root = fs::path(argv[1]);
    const fs::path out  = fs::path(argv[2]);

    int frames = 12;
    bool frames_from_cli = false;

    fs::path config_path;
    bool have_config = false;

    int argi = 3;
    if(argc >= 4){
      const std::string a3 = argv[3];
      if(!a3.empty() && a3[0] != '-'){
        frames = std::stoi(a3);
        frames_from_cli = true;
        argi = 4;
      }
    }

    ExportGeometry export_geom = ExportGeometry::POINTCLOUD;
    bool export_geom_from_cli = false;

    int mesh_kf = 0;
    bool mesh_kf_from_cli = false;

    int mesh_max_points = 2500;
    bool mesh_max_points_from_cli = false;

    int mesh_grid_px = 4;
    bool mesh_grid_px_from_cli = false;

    double mesh_max_edge_px = 80.0;
    bool mesh_max_edge_px_from_cli = false;

    // Algorithm knobs
    LKConfig kcfg{};
    BAConfig bacfg{};
    int kf_min_gap = 1;
    int kf_min_inliers = 200;
    double kf_parallax_px = 18.0;

    while(argi < argc){
      const std::string flag = argv[argi++];
      auto need = [&](const std::string& name)->std::string{
        if(argi >= argc) throw std::runtime_error("Missing value for " + name);
        return std::string(argv[argi++]);
      };
      if(flag == "--config"){
        config_path = fs::path(need(flag));
        have_config = true;
      } else if(flag == "--export-geometry"){
        const std::string v = need(flag);
        const auto eg = parse_export_geometry(v);
        if(!eg) throw std::runtime_error("Invalid --export-geometry value: " + v);
        export_geom = *eg;
        export_geom_from_cli = true;
      } else if(flag == "--mesh-kf"){
        mesh_kf = std::stoi(need(flag));
        mesh_kf_from_cli = true;
      } else if(flag == "--mesh-max-points"){
        mesh_max_points = std::stoi(need(flag));
        mesh_max_points_from_cli = true;
      } else if(flag == "--mesh-grid-px"){
        mesh_grid_px = std::stoi(need(flag));
        mesh_grid_px_from_cli = true;
      } else if(flag == "--mesh-max-edge-px"){
        mesh_max_edge_px = std::stod(need(flag));
        mesh_max_edge_px_from_cli = true;
      } else if(flag == "-h" || flag == "--help"){
        std::cerr << "Run without args to see usage.\n";
        return 0;
      } else {
        throw std::runtime_error("Unknown option: " + flag);
      }
    }

    if(!have_config){
      const fs::path local = fs::path("config.json");
      if(fs::exists(local)){
        config_path = local;
        have_config = true;
      }
    }

    std::optional<minijson::Value> cfg;
    if(have_config){
      try{
        cfg = minijson::parse(read_text_file(config_path));
      } catch(const std::exception& e){
        throw std::runtime_error("Failed to parse config.json: " + config_path.string() + " | " + e.what());
      }
    }

    // Apply config (common + cpp), with CLI taking precedence.
    if(cfg){
      if(!frames_from_cli){
        if(auto v = jint(jpick(*cfg, {"cpp","system","frames"}, {"common","system","frames"}))) frames = std::max(1, *v);
      }

      if(!export_geom_from_cli){
        if(auto s = jstring(jpick(*cfg, {"cpp","outputs","export_geometry"}, {"common","outputs","export_geometry"}))){
          if(const auto eg = parse_export_geometry(*s)) export_geom = *eg;
        }
      }

      if(!mesh_kf_from_cli){
        if(auto v = jint(jpick(*cfg, {"cpp","mesh_sparse","kf"}, {"common","mesh_sparse","kf"}))) mesh_kf = *v;
      }
      if(!mesh_max_points_from_cli){
        if(auto v = jint(jpick(*cfg, {"cpp","mesh_sparse","max_points"}, {"common","mesh_sparse","max_points"}))) mesh_max_points = *v;
      }
      if(!mesh_grid_px_from_cli){
        if(auto v = jint(jpick(*cfg, {"cpp","mesh_sparse","grid_px"}, {"common","mesh_sparse","grid_px"}))) mesh_grid_px = *v;
      }
      if(!mesh_max_edge_px_from_cli){
        if(auto v = jdouble(jpick(*cfg, {"cpp","mesh_sparse","max_edge_px"}, {"common","mesh_sparse","max_edge_px"}))) mesh_max_edge_px = *v;
      }

      // KLT tracking
      if(auto v = jint(jpick(*cfg, {"cpp","klt","max_tracks"}, {"common","klt","max_tracks"}))) kcfg.max_tracks = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","klt","min_tracks"}, {"common","klt","min_tracks"}))) kcfg.min_tracks = *v;
      if(auto v = jdouble(jpick(*cfg, {"cpp","klt","quality"}, {"common","klt","quality"}))) kcfg.quality = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","klt","min_distance"}, {"common","klt","min_distance"}))) kcfg.min_distance = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","klt","pyr_levels"}, {"common","klt","pyr_levels"}))) kcfg.pyr_levels = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","klt","win_radius"}, {"common","klt","win_radius"}))) kcfg.win_radius = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","klt","iters"}, {"common","klt","iters"}))) kcfg.iters = *v;
      if(auto v = jdouble(jpick(*cfg, {"cpp","klt","fb_thresh"}, {"common","klt","fb_thresh"}))) kcfg.fb_thresh = *v;

      // Keyframe selection
      if(auto v = jint(jpick(*cfg, {"cpp","keyframe","min_gap"}, {"common","keyframe","min_gap"}))) kf_min_gap = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","keyframe","min_inliers"}, {"common","keyframe","min_inliers"}))) kf_min_inliers = *v;
      if(auto v = jdouble(jpick(*cfg, {"cpp","keyframe","parallax_px"}, {"common","keyframe","parallax_px"}))) kf_parallax_px = *v;

      // BA
      if(auto v = jint(jpick(*cfg, {"cpp","ba","window"}, {"common","ba","window"}))) bacfg.window = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","ba","iters"}, {"common","ba","iters"}))) bacfg.iters = *v;
      if(auto v = jint(jpick(*cfg, {"cpp","ba","max_points"}, {"common","ba","max_points"}))) bacfg.max_points = *v;
      if(auto v = jdouble(jpick(*cfg, {"cpp","ba","huber_delta"}, {"common","ba","huber_delta"}))) bacfg.huber_delta = *v;
      if(auto v = jdouble(jpick(*cfg, {"cpp","ba","lambda"}, {"common","ba","lambda"}))) bacfg.lambda = *v;
    }

    const fs::path par = root / "templeRing" / "templeR_par.txt";
    const fs::path ang = root / "templeRing" / "templeR_ang.txt";
    const fs::path img_dir = root / "templeRing_pgm"; // created by tool script

    const auto recs = read_par(par);
    const auto angs = read_ang(ang);
    if(recs.empty()) throw std::runtime_error("No records in par file.");

    const Mat33 K = recs.front().K;

    KLTTracker tracker(kcfg);

    PoseCW cur = PoseCW::Identity();
    std::vector<Keyframe> kfs;
    MapState map;
    std::vector<PGEdge> edges;
    std::vector<std::vector<float>> kf_desc;
    std::unordered_map<int, std::vector<std::pair<int, Vec2>>> track_hist;

    std::vector<Vec2> last_prev, last_cur;
    std::vector<int>  last_ids;

    auto should_keyframe = [&](int inliers, double parallax, int frame_idx, int last_kf_frame)->bool{
      if (frame_idx - last_kf_frame < kf_min_gap) return false;
      if (inliers < kf_min_inliers) return true;
      return parallax >= kf_parallax_px;
    };

    int last_kf_frame = -999999;

    for(int fi=0; fi<std::min(frames, (int)recs.size()); ++fi){
      const auto& r = recs[fi];
      const fs::path pgm = img_dir / (fs::path(r.img).replace_extension(".pgm"));
      GrayImage gray = sfm::read_pgm(pgm.string());

      auto step = tracker.step(gray);

      if(step.prev_pts.empty()){
        // first keyframe
        Keyframe kf;
        kf.kf_id = (int)kfs.size();
        kf.frame_idx = fi;
        kf.img_name = r.img;
        kf.pose = cur;
        kf_desc.push_back(global_desc_32(gray));
        for(size_t i=0;i<tracker.tracks().size();++i){
          const auto& tr = tracker.tracks()[i];
          kf.obs.emplace(tr.id, tr.p);
          track_hist[tr.id].push_back({kf.kf_id, tr.p});
        }
        kfs.push_back(std::move(kf));
        last_kf_frame = fi;
        std::cout << "frame " << (fi+1) << "/" << frames << " | keyframes=" << kfs.size()
                  << " | map_points=" << map.pts.size() << "\n";
        continue;
      }

      // compute relative pose (E) from tracked correspondences (pixels)
      std::vector<Vec2> p_i = step.prev_pts;
      std::vector<Vec2> p_j = step.cur_pts;

      auto rel = find_E_ransac(K, p_i, p_j, 2500, 1e-3, 60);
      if(!rel.has_value()){
        // force keyframe if geometry fails
        rel = std::nullopt;
      }

      int inliers = 0;
      double parallax = 0.0;
      if(rel){
        inliers = (int)rel->inliers.size();
        // parallax median in pixels among inliers
        std::vector<double> ds;
        ds.reserve(rel->inliers.size());
        for(int idx : rel->inliers){
          const auto d = p_j[idx] - p_i[idx];
          ds.push_back(std::hypot(d.x, d.y));
        }
        if(!ds.empty()){
          std::nth_element(ds.begin(), ds.begin()+ds.size()/2, ds.end());
          parallax = ds[ds.size()/2];
        }

        // update pose
        cur = compose_right_inv_ij(cur, rel->R_ji, rel->t_ji);
      }

      const bool make_kf = (kfs.empty() || !rel.has_value() || should_keyframe(inliers, parallax, fi, last_kf_frame));
      if(make_kf){
        Keyframe kf;
        kf.kf_id = (int)kfs.size();
        kf.frame_idx = fi;
        kf.img_name = r.img;
        kf.pose = cur;
        const auto new_desc = global_desc_32(gray);

        // store observations for current tracker tracks
        for(const auto& tr : tracker.tracks()){
          kf.obs.emplace(tr.id, tr.p);
          track_hist[tr.id].push_back({kf.kf_id, tr.p});
          if(map.has(tr.id)) map.add_obs(tr.id, kf.kf_id, tr.p);
        }

        // Add sequential pose-graph edge (prev keyframe -> this keyframe) using shared track observations.
        if(!kfs.empty()){
          const auto& prev_kf = kfs.back();
          std::vector<Vec2> ei, ej;
          ei.reserve(1200); ej.reserve(1200);
          for(const auto& [tid, uvj] : kf.obs){
            auto itp = prev_kf.obs.find(tid);
            if(itp == prev_kf.obs.end()) continue;
            ei.push_back(itp->second);
            ej.push_back(uvj);
          }
          if(ei.size() >= 80){
            auto eopt = find_E_ransac(K, ei, ej, 2500, 1e-3, 60);
            if(eopt){
              edges.push_back(PGEdge{prev_kf.kf_id, kf.kf_id, eopt->R_ji, eopt->t_ji, (int)eopt->inliers.size(), false});
            }
          }
        }

        // triangulate new points from first and last obs of each track (simple)
        if(kfs.size() >= 1){
          const auto& kf0 = kfs.front();
          const auto& kfl = kf;
          for(auto& [tid, hist] : track_hist){
            if(map.has(tid) || hist.size() < 2) continue;
            const auto [id0, uv0] = hist.front();
            const auto [idl, uvl] = hist.back();
            if(id0 == idl) continue;
            const Vec3 Xw = triangulate_dlt(K, kfs[id0].pose, kfs[idl].pose, uv0, uvl);
            map.add(tid, Xw);
            for(const auto& [kid, uv] : hist) map.add_obs(tid, kid, uv);
          }
        }

        kfs.push_back(std::move(kf));
        kf_desc.push_back(new_desc);
        last_kf_frame = fi;

        // Local refinement: bundle adjust last W keyframes + active points.
        bundle_adjust_window(K, kfs, map, bacfg);

        // Loop closure: search older keyframes by global descriptor, verify with LK+E, then pose-graph optimize centers.
        const int new_kf_id = kfs.back().kf_id;
        const int min_gap = 6;
        int best_id = -1;
        float best_score = 0.0f;
        for(int kk=0; kk<(int)kfs.size()-min_gap; ++kk){
          const float s = dot_desc(kf_desc[kk], new_desc);
          if(s > best_score){ best_score = s; best_id = kk; }
        }
        if(best_id >= 0 && best_score > 0.94f){
          const auto& old_kf = kfs[best_id];
          const fs::path pgm_old = img_dir / (fs::path(old_kf.img_name).replace_extension(".pgm"));
          GrayImage im_old = sfm::read_pgm(pgm_old.string());

          LKConfig lc = kcfg;
          lc.max_tracks = 1200;
          lc.min_tracks = 600;
          KLTTracker tmp(lc);

          const auto pts0 = shi_tomasi(im_old, lc.max_tracks, lc.quality, lc.min_distance);
          Pyramid pyr0 = build_pyr(im_old, lc.pyr_levels);
          Pyramid pyr1 = build_pyr(gray,   lc.pyr_levels);

          std::vector<Vec2> li, lj;
          li.reserve(pts0.size()); lj.reserve(pts0.size());
          for(const auto& p0 : pts0){
            const Vec2 p1 = tmp.track_one_public(pyr0, pyr1, p0);
            const Vec2 p0b = tmp.track_one_public(pyr1, pyr0, p1);
            const double fb = std::hypot(p0b.x-p0.x, p0b.y-p0.y);
            if(fb >= lc.fb_thresh) continue;
            li.push_back(p0);
            lj.push_back(p1);
          }

          if(li.size() >= 120){
            auto lopt = find_E_ransac(K, li, lj, 4000, 2e-3, 80);
            if(lopt && (int)lopt->inliers.size() >= 100){
              edges.push_back(PGEdge{old_kf.kf_id, new_kf_id, lopt->R_ji, lopt->t_ji, (int)lopt->inliers.size(), true});

              // Drift reduction: translation pose-graph optimize camera centers, then locally re-BA.
              (void)posegraph_optimize_centers(kfs, edges);
              bundle_adjust_window(K, kfs, map, bacfg);
            }
          }
        }
      }

      std::cout << "frame " << (fi+1) << "/" << frames << " | keyframes=" << kfs.size()
                << " | map_points=" << map.pts.size() << "\n";
    }

    fs::create_directories(out);
    write_csv_centers(out / "keyframes_camera_centers.csv", kfs, angs);
    write_posegraph_edges(out / "posegraph_edges.csv", edges);

    // Geometry exports are optional. CSV outputs are always produced.
    if(export_geom == ExportGeometry::POINTCLOUD || export_geom == ExportGeometry::BOTH){
      std::vector<Vec3> xyz;
      xyz.reserve(map.pts.size());
      for(const auto& kv : map.pts) xyz.push_back(kv.second.Xw);
      write_ply_xyz(out / "templeRing_sparse_points.ply", xyz);
    }

    if(export_geom == ExportGeometry::MESH || export_geom == ExportGeometry::BOTH){
      if(kfs.empty()){
        std::cerr << "WARN: mesh export skipped (no keyframes).\n";
      } else {
        const int kidx = std::max(0, std::min(mesh_kf, (int)kfs.size()-1));
        const auto& mkf = kfs[kidx];
        const fs::path pgm_m = img_dir / (fs::path(mkf.img_name).replace_extension(".pgm"));
        GrayImage im_m = sfm::read_pgm(pgm_m.string());

        std::vector<Vec3> verts;
        std::vector<std::array<int,3>> faces;
        build_mesh_from_sparse_points(K, mkf, map.pts, im_m.w, im_m.h,
                                      mesh_max_points, mesh_grid_px, mesh_max_edge_px,
                                      verts, faces);
        if(verts.empty() || faces.empty()){
          std::cerr << "WARN: mesh export skipped (insufficient projected points or no valid triangles).\n";
        } else {
          const std::string fn = std::string("templeRing_mesh_sparse_kf") + std::to_string(kidx) + ".ply";
          write_ply_mesh(out / fn, verts, faces);
        }
      }
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Keyframes: " << kfs.size() << "\n";
    std::cout << "Map points: " << map.pts.size() << "\n";
    std::cout << "Outputs: " << out << "\n";
    return 0;
  } catch(const std::exception& e){
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
