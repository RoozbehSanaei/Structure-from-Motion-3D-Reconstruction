#pragma once
#include "linalg.hpp"

namespace sfm {

// Extra SO(3) helpers not present in linalg.hpp.

inline Mat33 hat(const Vec3& w){
  Mat33 W{};
  W(0,0)=0;   W(0,1)=-w.z; W(0,2)= w.y;
  W(1,0)=w.z; W(1,1)=0;    W(1,2)=-w.x;
  W(2,0)=-w.y;W(2,1)=w.x;  W(2,2)=0;
  return W;
}

inline Vec3 vee(const Mat33& W){
  return { W(2,1), W(0,2), W(1,0) };
}

inline Vec3 rodrigues_rvec(const Mat33& R){
  // Use the log map implemented in linalg.hpp.
  return so3_log(R);
}

} // namespace sfm
