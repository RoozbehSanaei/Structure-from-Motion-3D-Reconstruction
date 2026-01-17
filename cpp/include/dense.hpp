#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sfm {

// Simple dense matrix/vector utilities for small systems (<= a few hundred unknowns).
// Intended for BA window / translation pose-graph solves; no external dependencies.

struct DVec {
  std::vector<double> v;
  explicit DVec(int n=0, double init=0.0): v((size_t)n, init) {}
  int size() const { return (int)v.size(); }
  double& operator[](int i){ return v[(size_t)i]; }
  double operator[](int i) const { return v[(size_t)i]; }
  void fill(double x){ std::fill(v.begin(), v.end(), x); }
};

struct DMat {
  int r=0, c=0;
  std::vector<double> a;
  DMat() = default;
  DMat(int rows, int cols, double init=0.0): r(rows), c(cols), a((size_t)rows*(size_t)cols, init) {}
  double& operator()(int i, int j){ return a[(size_t)i*(size_t)c + (size_t)j]; }
  double operator()(int i, int j) const { return a[(size_t)i*(size_t)c + (size_t)j]; }
  void fill(double x){ std::fill(a.begin(), a.end(), x); }
};

inline void add_block(DMat& A, int r0, int c0, const double* blk, int br, int bc){
  for(int i=0;i<br;i++){
    for(int j=0;j<bc;j++){
      A(r0+i, c0+j) += blk[i*bc + j];
    }
  }
}

inline void add_block_t(DMat& A, int r0, int c0, const double* blk, int br, int bc){
  // Add transpose of blk (br x bc) into A at (r0,c0) with size (bc x br)
  for(int i=0;i<bc;i++){
    for(int j=0;j<br;j++){
      A(r0+i, c0+j) += blk[j*bc + i];
    }
  }
}

inline void add_vec(DVec& b, int o, const double* x, int n){
  for(int i=0;i<n;i++) b[o+i] += x[i];
}

inline DVec solve_gauss(DMat A, DVec b){
  if(A.r != A.c) throw std::runtime_error("solve_gauss: A must be square.");
  const int n = A.r;
  if(b.size() != n) throw std::runtime_error("solve_gauss: dim mismatch.");

  // Gaussian elimination with partial pivoting.
  for(int k=0;k<n;k++){
    int piv = k;
    double best = std::fabs(A(k,k));
    for(int i=k+1;i<n;i++){
      double v = std::fabs(A(i,k));
      if(v > best){ best=v; piv=i; }
    }
    if(best < 1e-15) throw std::runtime_error("solve_gauss: singular/ill-conditioned.");

    if(piv != k){
      for(int j=k;j<n;j++) std::swap(A(k,j), A(piv,j));
      std::swap(b[k], b[piv]);
    }

    const double akk = A(k,k);
    for(int j=k;j<n;j++) A(k,j) /= akk;
    b[k] /= akk;

    for(int i=k+1;i<n;i++){
      const double f = A(i,k);
      if(std::fabs(f) < 1e-18) continue;
      for(int j=k;j<n;j++) A(i,j) -= f*A(k,j);
      b[i] -= f*b[k];
    }
  }

  DVec x(n, 0.0);
  for(int i=n-1;i>=0;i--){
    double s = b[i];
    for(int j=i+1;j<n;j++) s -= A(i,j)*x[j];
    x[i] = s; // A(i,i) is 1
  }
  return x;
}

// Invert a 3x3 matrix (returns false if near-singular).
inline bool inv3(const double A[9], double invA[9]){
  const double a=A[0], b=A[1], c=A[2];
  const double d=A[3], e=A[4], f=A[5];
  const double g=A[6], h=A[7], i=A[8];

  const double A11 =  (e*i - f*h);
  const double A12 = -(d*i - f*g);
  const double A13 =  (d*h - e*g);
  const double A21 = -(b*i - c*h);
  const double A22 =  (a*i - c*g);
  const double A23 = -(a*h - b*g);
  const double A31 =  (b*f - c*e);
  const double A32 = -(a*f - c*d);
  const double A33 =  (a*e - b*d);

  const double det = a*A11 + b*A12 + c*A13;
  if(std::fabs(det) < 1e-15) return false;

  const double invdet = 1.0/det;
  invA[0]=A11*invdet; invA[1]=A21*invdet; invA[2]=A31*invdet;
  invA[3]=A12*invdet; invA[4]=A22*invdet; invA[5]=A32*invdet;
  invA[6]=A13*invdet; invA[7]=A23*invdet; invA[8]=A33*invdet;
  return true;
}

inline void mat3_mul(const double A[9], const double B[9], double C[9]){
  for(int r=0;r<3;r++){
    for(int c=0;c<3;c++){
      double s=0;
      for(int k=0;k<3;k++) s += A[r*3+k]*B[k*3+c];
      C[r*3+c]=s;
    }
  }
}

inline void mat3_mul_vec(const double A[9], const double x[3], double y[3]){
  for(int r=0;r<3;r++){
    y[r] = A[r*3+0]*x[0] + A[r*3+1]*x[1] + A[r*3+2]*x[2];
  }
}

inline void mat3_t_mul_vec(const double A[9], const double x[3], double y[3]){
  // y = A^T x
  for(int c=0;c<3;c++){
    y[c] = A[0*3+c]*x[0] + A[1*3+c]*x[1] + A[2*3+c]*x[2];
  }
}

} // namespace sfm
