
#include "serdp_common/DrawSonar.h"

#include <opencv2/imgproc.hpp>

#include <opencv2/opencv.hpp>

namespace serdp_common {

  using namespace std;
  using namespace liboculus;
  using namespace cv;

  void drawSonar( const shared_ptr<SimplePingResult> &ping, Mat &mat ) {
    // Ensure mat is 8UC3
    mat.create( mat.size(), CV_8UC3 );
    // mat.setTo( cv::Vec3b(0,0,0) );

    const unsigned int radius = mat.size().width / 2;
    const cv::Point origin( radius, mat.size().height );

    const float binThickness = 3 * ceil( radius / ping->ping()->nRanges );

    //LOG(DEBUG) << "binThickness is " << binThickness;


    // Build vector of start and end angles (in degrees, but still in sonar 0 == straight ahead frame)

    vector< pair<float,float> > angles( ping->ping()->nBeams, make_pair(0.0f, 0.0f) );
    for( unsigned int b = 0; b < ping->ping()->nBeams; ++b) {
      float begin=0.0, end = 0.0;

      //LOG(DEBUG) << "Bearing " << b << " is " << ping->bearings().at(b);
      if( b == 0 ) {

         end = (ping->bearings().at(b+1) + ping->bearings().at(b))/2.0;
         begin = 2*ping->bearings().at(b) - end;

      } else if ( b == ping->ping()->nBeams-1 ) {

         begin = angles[b-1].second;
         end = 2*ping->bearings().at(b) - begin;

      } else {

         begin = angles[b-1].second;
         end = (ping->bearings().at(b+1) + ping->bearings().at(b))/2.0;

      }

      //LOG(DEBUG) << "Bin " << b << " from " << begin << " to " << end;

      angles[b] = make_pair( begin, end );
    }

    for( unsigned int r = 0; r < ping->ping()->nRanges; ++r ) {
      for( unsigned int b = 0; b < ping->ping()->nBeams; ++b ) {

        auto intensity = ping->image().at( b, r );

        // Insert color mapping here
        cv::Scalar color( intensity, intensity, intensity );

        const float begin = angles[b].first+270, end= angles[b].second+270;

        //LOG_IF( DEBUG, r == 128 ) << "From " << begin << " to " << end << "; color " << color;

        const float rad = float(radius * r) / ping->ping()->nRanges;

        const float fudge=0.7;

        // Assume angles are in image frame x-right, y-down
        cv::ellipse( mat, origin, cv::Size(rad, rad), 0,
                    begin-fudge, end+fudge, color, binThickness*1.4 );

      }
    }


   }

   // draw sonar using ImagingSonarMsg rather than a SimplePingResult
   void drawSonar(const imaging_sonar_msgs::ImagingSonarMsg& msg) {

     int nRanges = msg.ranges.size();
     int nBeams = msg.bearings.size();
     int nIntensities = msg.v2intensities.size();
     //int nIntensities = msg.intensities.size();
     // how to recreate bearings() and image()
     // seems like bearings().at(x) is a 1D array, translates to values in msg.bearings
     // image.at(x,y) is a 2D array, maps to values in msg.intensities

     cv::Mat mat( 500, 1000, CV_8UC3 );
     mat.setTo( cv::Vec3b(128,128,128) );
     mat.create( mat.size(), CV_8UC3 );

     const unsigned int radius = mat.size().width / 2;
     const cv::Point origin( radius, mat.size().height );

     const float binThickness = 3 * ceil( radius / nRanges );

     //LOG(DEBUG) << "binThickness is " << binThickness;


     // Build vector of start and end angles (in degrees, but still in sonar 0 == straight ahead frame)

     vector< pair<float,float> > angles( nBeams, make_pair(0.0f, 0.0f) );

     for( unsigned int b = 0; b < nBeams; ++b) {
       float begin=0.0, end = 0.0;

       //LOG(DEBUG) << "Bearing " << b << " is " << ping->bearings().at(b);
       if( b == 0 ) {

          // originally: end = (ping->bearings().at(b+1) + ping->bearings().at(b))/2.0;
          end = (msg.bearings[b+1] + msg.bearings[b])/2.0;
          // originally: begin = 2*ping->bearings().at(b) - end;
          begin = 2*msg.bearings[b] - end;

       } else if ( b == nBeams-1 ) {

          begin = angles[b-1].second;
          // originally: end = 2*ping->bearings().at(b) - begin;
          end = 2*msg.bearings[b] - begin;

       } else {

          begin = angles[b-1].second;
          // originally: end = (ping->bearings().at(b+1) + ping->bearings().at(b))/2.0;
          end = (msg.bearings[b+1] + msg.bearings[b])/2.0;

       }

       //LOG(DEBUG) << "Bin " << b << " from " << begin << " to " << end;

       angles[b] = make_pair( begin, end );
     }

     for( unsigned int r = 0; r < nRanges; ++r ) {
       for( unsigned int b = 0; b < nBeams; ++b ) {

         // originally: auto intensity = ping->image().at( b, r );
         auto intensity = msg.v2intensities[(r * nBeams) + b];
         //auto intensity = msg.intensities[(r * nBeams) + b];
         // Insert color mapping here
         cv::Scalar color( intensity, intensity, intensity );

         const float begin = angles[b].first+270, end= angles[b].second+270;

         //LOG_IF( DEBUG, r == 128 ) << "From " << begin << " to " << end << "; color " << color;

         const float rad = float(radius * r) / nRanges;

         const float fudge=0.7;

         // Assume angles are in image frame x-right, y-down
         cv::ellipse( mat, origin, cv::Size(rad, rad), 0,
                     begin-fudge, end+fudge, color, binThickness*1.4 );
       }
     }

     cv::imshow("ROS sonar", mat);
     cv::waitKey(1);


   }

}