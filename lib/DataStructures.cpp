#include "serdp_common/DataStructures.h"

namespace serdp_common {

SonarPoint bearingRange2Cartesian(float bearing, float range) {
  float x = range * sin(bearing);
  float z = range * cos(bearing);

  return SonarPoint(x, z);
}


} // namespace serdp_common