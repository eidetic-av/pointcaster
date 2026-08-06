#pragma once

#include <cmath>
#include <libobsensor/h/ObTypes.h>

namespace pc::devices::orbbec::util {

// The functions from this file come from the official Orbbec v2 SDK
// https://github.com/orbbec/OrbbecSDK_v2/blob/12eef82e7f9e2bdf52d25984cbd99ded8882d316/src/shared/utils/CoordinateUtil.cpp
//
// they are re-implemented in this header with added attributes so they can be
// compiled using nvcc and used on the GPU path
//
// TODO remove validations

constexpr float EPS = 1e-4f;
constexpr float FMAX = 1e4f;

#ifdef __CUDACC__
__host__ __device__
#endif
static bool judgeTransformValid(OBD2CTransform cameraRotParam) {
  // Orthogonality of rotation matrix
  // r1 .*r2 = 0 ;
  float r1r2 = cameraRotParam.rot[0] * cameraRotParam.rot[3] +
               cameraRotParam.rot[1] * cameraRotParam.rot[4] +
               cameraRotParam.rot[2] * cameraRotParam.rot[5];

  float r1r3 = cameraRotParam.rot[0] * cameraRotParam.rot[6] +
               cameraRotParam.rot[1] * cameraRotParam.rot[7] +
               cameraRotParam.rot[2] * cameraRotParam.rot[8];

  if (fabsf(r1r2) < EPS && fabsf(r1r3) < EPS) {
    return true;
  }
  return false;
}

#ifdef __CUDACC__
__host__ __device__
#endif
static bool judgeIntrinsicValid(OBCameraIntrinsic param) {
  if (param.width == 0 || param.height == 0) {
    return false;
  }

  if ((param.fx < 1.0f) || (param.fy < 1.0f) || (param.cx < 1.0f) ||
      (param.cy < 1.0f)) {
    return false;
  }
  if ((param.fx > FMAX) || (param.fy > FMAX) || (param.cx > FMAX) ||
      (param.cy > FMAX)) {
    return false;
  }

  return true;
}

#ifdef __CUDACC__
__host__ __device__
#endif
static bool transformation3dTo3d(const OBPoint3f sourcePoint3f,
                                 OBD2CTransform transSourceToTarget,
                                 OBPoint3f *targetPoint3f) {
  // step 1: parameter validity judgment
  if (!judgeTransformValid(transSourceToTarget)) {
    return false;
  }

  if (fabsf(sourcePoint3f.z) < EPS) {
    return false;
  }

  // step 2: Calculate conversion relationship
  // R *X + t
  float rx = transSourceToTarget.rot[0] * sourcePoint3f.x +
             transSourceToTarget.rot[1] * sourcePoint3f.y +
             transSourceToTarget.rot[2] * sourcePoint3f.z;
  float ry = transSourceToTarget.rot[3] * sourcePoint3f.x +
             transSourceToTarget.rot[4] * sourcePoint3f.y +
             transSourceToTarget.rot[5] * sourcePoint3f.z;
  float rz = transSourceToTarget.rot[6] * sourcePoint3f.x +
             transSourceToTarget.rot[7] * sourcePoint3f.y +
             transSourceToTarget.rot[8] * sourcePoint3f.z;

  (*targetPoint3f).x = rx + transSourceToTarget.trans[0];
  (*targetPoint3f).y = ry + transSourceToTarget.trans[1];
  (*targetPoint3f).z = rz + transSourceToTarget.trans[2];

  return true;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline bool transformation2dTo3d(const OBCameraIntrinsic sourceIntrinsic,
                                 const OBPoint2f sourcePoint2f,
                                 const float sourceDepthPixelValue,
                                 OBD2CTransform transSourceToTarget,
                                 OBPoint3f *targetPoint3f) {
  // step 1: parameter validity judgment
  if (!judgeIntrinsicValid(sourceIntrinsic)) {
    return false;
  }

  if (!judgeTransformValid(transSourceToTarget)) {
    return false;
  }

  // sourcePoint2f is the pixel coordinates of the image, or sub-pixel
  // coordinates
  if (sourcePoint2f.x < 0 || sourcePoint2f.y < 0) {
    return false;
  }

  if (sourcePoint2f.x > (sourceIntrinsic.width - 1) ||
      sourcePoint2f.y > (sourceIntrinsic.height - 1)) {
    return false;
  }

  // step 2: Convert 2D to 3D point
  OBPoint3f source_3f;
  source_3f.z = sourceDepthPixelValue; // Assignment in z direction
  // Convert 2d to 3d (same as converting point cloud)
  source_3f.x = sourceDepthPixelValue * (sourcePoint2f.x - sourceIntrinsic.cx) /
                sourceIntrinsic.fx;
  source_3f.y = sourceDepthPixelValue * (sourcePoint2f.y - sourceIntrinsic.cy) /
                sourceIntrinsic.fy;

  // step 3: Convert the 3D point under the source coordinates to the target
  // camera coordinates
  bool ret =
      transformation3dTo3d(source_3f, transSourceToTarget, targetPoint3f);

  return ret;
}

}; // namespace pc::devices::orbbec::util