#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <tuple>
#include <vector>

namespace sfm {

constexpr double kEps = 1e-12;

struct Vec2 {
  double x{}, y{};
};
struct Vec3 {
  double x{}, y{}, z{};
};

inline Vec3 operator-(const Vec3& v){ return {-v.x,-v.y,-v.z}; }
inline Vec2 operator-(const Vec2& v){ return {-v.x,-v.y}; }

struct Mat33 {
  // Row-major
  std::array<double, 9> a{};
  static Mat33 I() {
    Mat33 m;
    m(0,0)=1; m(1,1)=1; m(2,2)=1;
    return m;
  }
  double& operator()(int r, int c) { return a[3*r + c]; }
  double  operator()(int r, int c) const { return a[3*r + c]; }
};

inline Vec3 operator+(const Vec3& u, const Vec3& v){ return {u.x+v.x,u.y+v.y,u.z+v.z}; }
inline Vec3 operator-(const Vec3& u, const Vec3& v){ return {u.x-v.x,u.y-v.y,u.z-v.z}; }
inline Vec3 operator*(double s, const Vec3& v){ return {s*v.x,s*v.y,s*v.z}; }
inline Vec3 operator*(const Vec3& v, double s){ return s*v; }
inline Vec3 operator/(const Vec3& v, double s){ return {v.x/s,v.y/s,v.z/s}; }

inline Vec2 operator+(const Vec2& u, const Vec2& v){ return {u.x+v.x,u.y+v.y}; }
inline Vec2 operator-(const Vec2& u, const Vec2& v){ return {u.x-v.x,u.y-v.y}; }
inline Vec2 operator*(double s, const Vec2& v){ return {s*v.x,s*v.y}; }
inline Vec2 operator*(const Vec2& v, double s){ return s*v; }

inline double dot(const Vec3& u, const Vec3& v){ return u.x*v.x+u.y*v.y+u.z*v.z; }
inline Vec3 cross(const Vec3& u, const Vec3& v){
  return {u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x};
}
inline double norm(const Vec3& v){ return std::sqrt(dot(v,v)); }
inline Vec3 unit(const Vec3& v){
  const double n = norm(v);
  if (!std::isfinite(n) || n < kEps) return {0,0,0};
  return v / n;
}

inline Mat33 operator*(const Mat33& A, const Mat33& B){
  Mat33 C{};
  for(int r=0;r<3;r++){
    for(int c=0;c<3;c++){
      double s=0;
      for(int k=0;k<3;k++) s += A(r,k)*B(k,c);
      C(r,c)=s;
    }
  }
  return C;
}
inline Vec3 operator*(const Mat33& A, const Vec3& v){
  return {
    A(0,0)*v.x + A(0,1)*v.y + A(0,2)*v.z,
    A(1,0)*v.x + A(1,1)*v.y + A(1,2)*v.z,
    A(2,0)*v.x + A(2,1)*v.y + A(2,2)*v.z
  };
}
inline Mat33 transpose(const Mat33& A){
  Mat33 T{};
  for(int r=0;r<3;r++) for(int c=0;c<3;c++) T(r,c)=A(c,r);
  return T;
}
inline double det(const Mat33& A){
  return
    A(0,0)*(A(1,1)*A(2,2)-A(1,2)*A(2,1))
  - A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
  + A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
}

// Rodrigues (so(3) exp) for rotation vector w (axis-angle)
inline Mat33 so3_exp(const Vec3& w){
  const double th = norm(w);
  Mat33 R = Mat33::I();
  if (th < 1e-10) {
    // R ≈ I + [w]x
    const double wx=w.x, wy=w.y, wz=w.z;
    R(0,1) = -wz; R(0,2) =  wy;
    R(1,0) =  wz; R(1,2) = -wx;
    R(2,0) = -wy; R(2,1) =  wx;
    return R;
  }
  const Vec3 a = w / th;
  const double ax=a.x, ay=a.y, az=a.z;
  const double c=std::cos(th), s=std::sin(th), C=1-c;
  R(0,0)=c+ax*ax*C;   R(0,1)=ax*ay*C-az*s; R(0,2)=ax*az*C+ay*s;
  R(1,0)=ay*ax*C+az*s;R(1,1)=c+ay*ay*C;    R(1,2)=ay*az*C-ax*s;
  R(2,0)=az*ax*C-ay*s;R(2,1)=az*ay*C+ax*s; R(2,2)=c+az*az*C;
  return R;
}

// Log map (rotation matrix -> rotation vector), using trace-based formula.
inline Vec3 so3_log(const Mat33& R){
  const double tr = R(0,0)+R(1,1)+R(2,2);
  double cos_th = (tr - 1.0) * 0.5;
  cos_th = std::max(-1.0, std::min(1.0, cos_th));
  const double th = std::acos(cos_th);
  if (th < 1e-10) return {0,0,0};
  const double s = std::sin(th);
  const double k = th / (2.0*s);
  return {
    k*(R(2,1)-R(1,2)),
    k*(R(0,2)-R(2,0)),
    k*(R(1,0)-R(0,1))
  };
}

// Symmetric Jacobi eigen-decomposition for small dense matrices (N<=10).
// Returns eigenvalues (ascending) and eigenvectors (columns).
struct SymEig {
  std::vector<double> w;          // size N
  std::vector<double> V;          // NxN, column-major
};

inline SymEig jacobi_eig_sym(std::vector<double> A, int N, int iters=60){
  // A is NxN row-major symmetric
  std::vector<double> V(N*N, 0.0);
  for(int i=0;i<N;i++) V[i*N+i]=1.0;

  auto idx = [N](int r,int c){ return r*N+c; };
  auto vidx = [N](int r,int c){ return r*N+c; };

  for(int it=0; it<iters; ++it){
    // find largest off-diagonal
    int p=0,q=1;
    double maxv = 0;
    for(int i=0;i<N;i++){
      for(int j=i+1;j<N;j++){
        double v = std::fabs(A[idx(i,j)]);
        if (v > maxv){ maxv=v; p=i; q=j; }
      }
    }
    if (maxv < 1e-12) break;

    const double app = A[idx(p,p)];
    const double aqq = A[idx(q,q)];
    const double apq = A[idx(p,q)];
    const double phi = 0.5 * std::atan2(2.0*apq, (aqq-app));
    const double c = std::cos(phi), s = std::sin(phi);

    // rotate A = J^T A J
    for(int k=0;k<N;k++){
      const double aik = A[idx(p,k)];
      const double aqk = A[idx(q,k)];
      A[idx(p,k)] = c*aik - s*aqk;
      A[idx(q,k)] = s*aik + c*aqk;
    }
    for(int k=0;k<N;k++){
      const double aki = A[idx(k,p)];
      const double akq = A[idx(k,q)];
      A[idx(k,p)] = c*aki - s*akq;
      A[idx(k,q)] = s*aki + c*akq;
    }
    A[idx(p,q)] = 0.0;
    A[idx(q,p)] = 0.0;

    // rotate V = V J
    for(int k=0;k<N;k++){
      const double vip = V[vidx(k,p)];
      const double viq = V[vidx(k,q)];
      V[vidx(k,p)] = c*vip - s*viq;
      V[vidx(k,q)] = s*vip + c*viq;
    }
  }

  std::vector<double> w(N);
  for(int i=0;i<N;i++) w[i] = A[i*N+i];

  // sort eigenvalues ascending, permute V columns
  std::vector<int> perm(N);
  for(int i=0;i<N;i++) perm[i]=i;
  std::sort(perm.begin(), perm.end(), [&](int i,int j){ return w[i] < w[j]; });

  std::vector<double> w2(N);
  std::vector<double> V2(N*N);
  for(int ci=0; ci<N; ++ci){
    w2[ci] = w[perm[ci]];
    for(int r=0;r<N;r++){
      V2[r*N+ci] = V[r*N+perm[ci]];
    }
  }
  return {std::move(w2), std::move(V2)};
}

} // namespace sfm
