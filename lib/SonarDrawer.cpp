// Copyright 2021 University of Washington Applied Physics Laboratory
//

#include "sonar_image_proc/DrawSonar.h"

#include <iostream>
#include <limits>
#include <opencv2/imgproc/imgproc.hpp>

namespace sonar_image_proc {

using namespace std;
using namespace cv;


SonarDrawer::SonarDrawer()
{;}

void SonarDrawer::drawSonarRectImage(const sonar_image_proc::AbstractSonarInterface &ping,
                        cv::Mat &rect,
                        const SonarColorMap &colorMap) {
    if ((rect.type() == CV_8UC3) || (rect.type() == CV_32FC2)) {
        rect.create(cv::Size(ping.nRanges(), ping.nBearings()),
            rect.type());
    } else {
        rect.create(cv::Size(ping.nRanges(), ping.nBearings()),
                    CV_8UC3);
    }
    rect.setTo(cv::Vec3b(0, 0, 0));

    for (int r = 0; r < ping.nRanges(); r++) {
    for (int b = 0; b < ping.nBearings(); b++) {

        if (rect.type() == CV_8UC3) {
            rect.at<Vec3b>(cv::Point(r, b)) = colorMap.lookup<Vec3b>(ping, b, r);
        } else if (rect.type() == CV_32FC3) {
            rect.at<Vec3f>(cv::Point(r, b)) = colorMap.lookup<Vec3f>(ping, b, r);
        } else {
            assert("Should never get here.");
        }
    }
    }
}

void SonarDrawer::drawSonar(const sonar_image_proc::AbstractSonarInterface &ping,
                    cv::Mat &img,
                    const SonarColorMap &colorMap,
                    const cv::Mat &rect) {

  cv::Mat rectImage(rect);

  if (rect.empty())
      drawSonarRectImage(ping, rectImage, colorMap);

  cv::remap(rect, img, _map(ping), cv::Mat(),
            cv::INTER_CUBIC, cv::BORDER_CONSTANT, 
            cv::Scalar(0, 0, 0));
}



// ==== SonarDrawer::CachedMap ====

cv::Mat SonarDrawer::CachedMap::operator()(const sonar_image_proc::AbstractSonarInterface &ping) {
    if (!isValid(ping)) create(ping);

    return _mapF;
}


// Create **assumes** the structure of the rectImage:
//   ** nBearings cols and nRanges rows
//
void SonarDrawer::CachedMap::create(const sonar_image_proc::AbstractSonarInterface &ping) {
    std::cerr << "Recreating map..." << std::endl;

  // Create map
  cv::Mat newmap;
  //newmap.create(cv::Size(ping.nRanges(), ping.nBearings()), CV_32FC2);

  const int nRanges = ping.nRanges();
  const auto azimuthBounds = ping.azimuthBounds();

  const int minusWidth = floor(nRanges * sin(azimuthBounds.first));
  const int plusWidth = ceil(nRanges * sin(azimuthBounds.second));
  const int width = plusWidth - minusWidth;

  const int originx = abs(minusWidth);

  const cv::Size imgSize(width,nRanges);
  newmap.create(imgSize, CV_32FC2);

  const float db = (azimuthBounds.second - azimuthBounds.first) / ping.nAzimuth();

  for (int x=0; x<newmap.cols; x++) {
    for (int y=0; y<newmap.rows; y++) {
      // Unoptimized version to start

      // Map is
      //
      //  dst = src( mapx(x,y), mapy(x,y) )
      //
      float xp, yp;

      // Calculate range and bearing of this pixel from origin
      const float dx = x-originx;
      const float dy = newmap.rows-y;

      const float range = sqrt(dx*dx + dy*dy);
      const float azimuth = atan2(dx, dy);

      // yp is simply range
      xp = range;
      yp = (azimuth - ping.bearing(0))/db;

      newmap.at<Vec2f>(cv::Point(x, y)) = Vec2f(xp,yp);
    }
  }

  // Save metadata
  _mapF = newmap;

  _numRanges = ping.nRanges();
  _numAzimuth = ping.nBearings();

  _rangeBounds = ping.rangeBounds();
  _azimuthBounds = ping.azimuthBounds();
}

bool SonarDrawer::CachedMap::isValid(const sonar_image_proc::AbstractSonarInterface &ping) {
    if (_mapF.empty()) return false;

    // Check for cache invalidation...
    if ((_numAzimuth != ping.nAzimuth()) ||
        (_numRanges != ping.nRanges()) ||
        (_rangeBounds != ping.rangeBounds() ||
        (_azimuthBounds != ping.azimuthBounds()))) return false;

    return true;
}



}  // namespace sonar_image_proc
