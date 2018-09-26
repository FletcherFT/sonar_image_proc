
// TODO:   Reduce the DRY

#include <string>
#include <thread>
using namespace std;

#include <signal.h>

#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include <CLI/CLI.hpp>

#include <libg3logger/g3logger.h>

#include "liboculus/StatusRx.h"
#include "liboculus/SonarClient.h"
using namespace liboculus;

#include "libblackmagic/DeckLink.h"
#include "libblackmagic/DataTypes.h"
using namespace libblackmagic;

#include "libbmsdi/helpers.h"

#include "libvideoencoder/VideoEncoder.h"
using libvideoencoder::Encoder;

#include "IoServiceThread.h"

#include "serdprecorder/CameraState.h"
#include "serdprecorder/VideoRecorder.h"
using namespace serdprecorder;


using cv::Mat;

bool keepGoing = true;


void signal_handler( int sig )
{
	LOG(INFO) << "Signal handler: " << sig;

	switch( sig ) {
		case SIGINT:
				keepGoing = false;
				break;
		default:
				keepGoing = false;
				break;
	}
}

const int CamNum = 1;

DeckLink deckLink;
shared_ptr<VideoRecorder> recorder(nullptr);



static void processKbInput( char c, DeckLink &decklink, CameraState &camState ) {

	shared_ptr<SharedBMSDIBuffer> sdiBuffer( decklink.output().sdiProtocolBuffer() );

	SDIBufferGuard guard( sdiBuffer );

	switch(c) {
		case 'f':
					// Send absolute focus value
					LOG(INFO) << "Sending instantaneous autofocus to camera";
					guard( []( BMSDIBuffer *buffer ){ bmAddInstantaneousAutofocus( buffer, CamNum ); });
					break;
		 case '[':
					// Send positive focus increment
					LOG(INFO) << "Sending focus increment to camera";
					guard( []( BMSDIBuffer *buffer ){	bmAddFocusOffset( buffer, CamNum, 0.05 ); });
					break;
			case ']':
					// Send negative focus increment
					LOG(INFO) << "Sending focus decrement to camera";
					guard( []( BMSDIBuffer *buffer ){ bmAddFocusOffset( buffer, CamNum, -0.05 ); });
					break;

			//=== Aperture increment/decrement ===
			case ';':
					{
						// Send positive aperture increment
						auto val = camState.apertureInc();
	 					LOG(INFO) << "Sending aperture increment " << val.ord << " , " << val.val << " , " << val.str;
					}
 					// guard( []( BMSDIBuffer *buffer ){	bmAddOrdinalApertureOffset( buffer, CamNum, 1 ); });
 					break;
 			case '\'':
					{
						// Send negative aperture decrement
						auto val = camState.apertureDec();
						LOG(INFO) << "Sending aperture decrement " << val.ord << " , " << val.val << " , " << val.str;
					// guard( []( BMSDIBuffer *buffer ){	bmAddOrdinalApertureOffset( buffer, CamNum, -1 ); });
					}
 					break;

			//=== Shutter increment/decrement ===
			case '.':
 					LOG(INFO) << "Sending shutter increment to camera";
					guard( []( BMSDIBuffer *buffer ){	bmAddOrdinalShutterOffset( buffer, CamNum, 1 ); });
 					break;
 			case '/':
 					LOG(INFO) << "Sending shutter decrement to camera";
					guard( []( BMSDIBuffer *buffer ){	bmAddOrdinalShutterOffset( buffer, CamNum, -1 ); });
 					break;

			//=== Gain increment/decrement ===
			case 'z':
 					LOG(INFO) << "Sending gain increment to camera";
					guard( []( BMSDIBuffer *buffer ){	bmAddSensorGainOffset( buffer, CamNum, 1 ); });
 					break;
 			case 'x':
 					LOG(INFO) << "Sending gain decrement to camera";
					guard( []( BMSDIBuffer *buffer ){	bmAddSensorGainOffset( buffer, CamNum, -1 ); });
 					break;

			//== Increment/decrement white balance
			case 'w':
					LOG(INFO) << "Auto white balance";
					guard( []( BMSDIBuffer *buffer ){	bmAddAutoWhiteBalance( buffer, CamNum ); });
					break;

			case 'e':
					LOG(INFO) << "Restore white balance";
					guard( []( BMSDIBuffer *buffer ){	bmAddRestoreWhiteBalance( buffer, CamNum ); });
					break;

			case 'r':
					LOG(INFO) << "Sending decrement to white balance";
					guard( []( BMSDIBuffer *buffer ){	bmAddWhiteBalanceOffset( buffer, CamNum, -500, 0 ); });
					break;

			case 't':
					LOG(INFO) << "Sending increment to white balance";
					guard( []( BMSDIBuffer *buffer ){	bmAddWhiteBalanceOffset( buffer, CamNum, 500, 0 ); });
					break;

			case '1':
					LOG(INFO) << "Sending Program reference to cameras";
					// guard( [](BMSDIBuffer *buffer ){bmAddReferenceSource( buffer, CamNum, BM_REF_SOURCE_PROGRAM );});
					guard( [](BMSDIBuffer *buffer ){				bmAddVideoMode( buffer, CamNum,bmdModeHD1080p30 );});
					break;

			case '2':
					LOG(INFO) << "Sending 2160p25 reference to cameras";
					guard( [](BMSDIBuffer *buffer ){				bmAddVideoMode( buffer, CamNum,bmdMode4K2160p25 );});
					break;


			case '\\':
					if( recorder->isRecording() ) {
						LOG(INFO) << "Stopping recording";
						recorder->close();
					} else {
						LOG(INFO) << "Starting recording";

						libblackmagic::ModeConfig config( decklink.input().currentConfig() );
						if( !recorder->open( config.width(), config.height(), config.do3D() ) ) {
							LOG(WARNING) << "Unable to start recorder!";
						}
					}

		case 'q':
				keepGoing = false;
				break;
	}

}


int main( int argc, char** argv )
{
	libg3logger::G3Logger logger("bmRecorder");

	signal( SIGINT, signal_handler );

	CLI::App app{"Simple BlackMagic camera recorder"};

	int verbosity = 0;
	app.add_flag("-v", verbosity, "Additional output (use -vv for even more!)");

	bool do3D = false;
	app.add_flag("--do-3d",do3D, "Enable 3D modes");

	bool noDisplay = false;
	app.add_flag("--no-display,-x", noDisplay, "Disable display");

	string desiredModeString = "1080p2997";
	app.add_option("--mode,-m", desiredModeString, "Desired mode");

	bool doConfigCamera = false;
	app.add_flag("--config-camera,-c", doConfigCamera, "If enabled, send initialization info to the cameras");

	bool doListCards = false;
	app.add_flag("--list-cards", doListCards, "List Decklink cards in the system then exit");

	bool doListInputModes = false;
	app.add_flag("--list-input-modes", doListInputModes, "List Input modes then exit");

	int stopAfter = -1;
	app.add_option("--stop-after", stopAfter, "Stop after N frames");

  string sonarIp("auto");
  app.add_option("-s,--sonar-ip", sonarIp, "IP address of sonar or \"auto\" to automatically detect.");

	string outputDir;
	app.add_option("--output,-o", outputDir, "Output dir");

	CLI11_PARSE(app, argc, argv);

	switch(verbosity) {
		case 1:
			logger.stderrHandle->call( &ColorStderrSink::setThreshold, INFO );
			break;
		case 2:
			logger.stderrHandle->call( &ColorStderrSink::setThreshold, DEBUG );
			break;
	}


	// Help string
	cout << "Commands" << endl;
	cout << "    q       quit" << endl;
	cout << "   [ ]     Adjust focus" << endl;
	cout << "    f      Set autofocus" << endl;
	cout << "   ; '     Adjust aperture" << endl;
	cout << "   . /     Adjust shutter speed" << endl;
	cout << "   z x     Adjust sensor gain" << endl;
	cout << "    s      Cycle through reference sources" << endl;

	// Handle the one-off commands
	if( doListCards || doListInputModes ) {
			if(doListCards) deckLink.listCards();
			if(doListInputModes) deckLink.listInputModes();
		return 0;
	}

	BMDDisplayMode mode = stringToDisplayMode( desiredModeString );
	if( mode == bmdModeUnknown ) {
		LOG(FATAL) << "Didn't understand mode \"" << desiredModeString << "\"";
		return -1;
	} else if ( mode == bmdModeDetect ) {
		LOG(WARNING) << "Card will always attempt automatic detection, starting in HD1080p2997 mode";
		mode = bmdModeHD1080p2997;
	} else {
		LOG(WARNING) << "Setting mode " << desiredModeString;
		// if( outputFile.size() > 0 ) {
		// 		videoOutput = MakeVideoEncoder( outputFile, mode, do3D);
		// }
	}

	//  Input should always auto-detect
	deckLink.input().enable( mode, true, do3D );
	deckLink.output().enable( mode );

//	std::shared_ptr<Encoder> videoOutput(nullptr);

	recorder.reset( new VideoRecorder( outputDir ) );

	int count = 0, miss = 0, displayed = 0;

	CameraState cameraState( deckLink.output().sdiProtocolBuffer() );

		LOG(DEBUG) << "Starting streams";
		if( !deckLink.startStreams() ) {
				LOG(WARNING) << "Unable to start streams";
				exit(-1);
		}


		bool once = true;

	while( keepGoing ) {

		std::chrono::steady_clock::time_point loopStart( std::chrono::steady_clock::now() );
		//if( (duration > 0) && (loopStart > end) ) { keepGoing = false;  break; }

		int numImages = 0;
		if( (numImages = deckLink.input().grab()) > 0 ) {

			// Only do these things once good data starts flowing
			if( displayed == 100 ) {
				once = false;

				if ( doConfigCamera ) {
					LOG(INFO) << "Sending configuration to cameras";

				// Be careful not to exceed 255 byte buffer length
				SDIBufferGuard guard( deckLink.output().sdiProtocolBuffer() );
				guard( [mode]( BMSDIBuffer *buffer ) {

					// bmAddOrdinalAperture( buffer, CamNum, 0 );
					// bmAddSensorGain( buffer, CamNum, 8 );
					bmAddReferenceSource( buffer, CamNum, BM_REF_SOURCE_PROGRAM );
					// bmAddAutoWhiteBalance( buffer, CamNum );

					if(mode != bmdModeDetect) {
						LOG(INFO) << "Setting video mode to " << displayModeToString(mode);
						bmAddVideoMode( buffer, CamNum, mode );
					}

				});

				cameraState.updateCamera();
			} else {
				if( mode != bmdModeDetect ) {
					LOG(WARNING) << "Mode " << desiredModeString << " requested, but camera configuration not requested with -c";
				}
			}
			}

			// If output doesn't already exist (might be the case if mdoe = bmdModeDetect)
			// if( (outputFile.size() > 0) && (!videoOutput) ) {
			// 		videoOutput = MakeVideoEncoder( outputFile, deckLink.input().config() );
			// }

			std::array<cv::Mat,2> images;
			for( unsigned int i=0; i < (unsigned int)count && i < images.size(); ++i ) {
				deckLink.input().getRawImage(i, images[i]);

				//if ( videoOutput ) {
					// Convert to AVFrame
					// AVFrame *frame = av_frame_alloc();   ///avcodec_alloc_frame();
					// CHECK( frame != nullptr ) << "Cannot create frame";
					//
				  // frame->width = images[i].size().width;
				  // frame->height = images[i].size().height;
				  // frame->format = AV_PIX_FMT_BGR24;
					//
					// frame->data[0] = images[i].data;
					// frame->linesize[0] = 3*images[i].size().width;
					// frame->extended_data = frame->data;
					//
					// //auto res = av_frame_get_buffer(frame, 0);
					//
					// //videoOutput->AddFrame( frame, count, i );
					//
					// av_frame_free( &frame );
				//}
			}

			if( noDisplay ) {

				//

			} else {

				// Display images

				if( numImages == 1 ) {
					cv::imshow("Image", images[0]);
				} else if ( numImages == 2 ) {

					cv::Mat composite( cv::Size( images[0].size().width + images[0].size().width,
					std::max(images[0].size().height, images[1].size().height )), images[0].type() );

					if( !images[0].empty() ) {
						cv::Mat leftROI( composite, cv::Rect(0,0,images[0].size().width,images[0].size().height) );
						images[0].copyTo( leftROI );
					}

					if( !images[1].empty() ) {
						cv::Mat rightROI( composite, cv::Rect(images[0].size().width, 0, images[1].size().width, images[1].size().height) );
						images[1].copyTo( rightROI );
					}

					cv::imshow("Composite", composite );

					char c = cv::waitKey(1);
					processKbInput( c, deckLink, cameraState );
				}

			}

			LOG_IF(INFO, (displayed % 50) == 0) << "Frame #" << displayed;
			++displayed;


		} else {
			// if grab() fails
			LOG(INFO) << "unable to grab frame";
			++miss;
			std::this_thread::sleep_for(std::chrono::microseconds(1000));
		}

		++count;

		if((stopAfter > 0) && (count >= stopAfter)) keepGoing = false;

	}

	 //std::chrono::duration<float> dur( std::chrono::steady_clock::now()  - start );

	LOG(INFO) << "End of main loop, stopping streams...";

	deckLink.stopStreams();


	// LOG(INFO) << "Recorded " << count << " frames in " <<   dur.count();
	// LOG(INFO) << " Average of " << (float)count / dur.count() << " FPS";
	// LOG(INFO) << "   " << miss << " / " << (miss+count) << " misses";
	// LOG_IF( INFO, displayed > 0 ) << "   Displayed " << displayed << " frames";



		return 0;
	}



	//
	//
	//   try {
	//     IoServiceThread ioSrv;
	//
	//     OsStatusRx statusRx( ioSrv.service() );
	//     std::unique_ptr<SonarClient> sonarClient( nullptr );
	//
	//     if( ipAddr != "auto" ) {
	//       LOG(INFO) << "Connecting to sonar with IP address " << ipAddr;
	//       auto addr( boost::asio::ip::address_v4::from_string( ipAddr ) );
	//
	//       LOG_IF(FATAL,addr.is_unspecified()) << "Hm, couldn't parse IP address";
	//
	//       sonarClient.reset( new SonarClient( ioSrv.service(), addr ) );
	//     }
	//
	//     ioSrv.start();
	//
	//     while(true) {
	//
	//       if( !sonarClient && ipAddr == "auto") {
	//         if( statusRx.status().valid() ) {
	//           auto addr( statusRx.status().ipAddr() );
	//
	//           LOG(INFO) << "Using detected sonar at IP address " << addr;
	//
	//           if( verbosity > 0 ) statusRx.status().dump();
	//
	//           sonarClient.reset( new SonarClient( ioSrv.service(), addr ) );
	//
	//         }
	//       }
	//
	//       if( sonarClient ) {
	//         // Do some stuff
	//         sleep(1);
	//       }
	//
	//     }
	//
	//     ioSrv.stop();
	//
	//   }
	//   catch (std::exception& e)
	//   {
	//     LOG(WARNING) << "Exception: " << e.what();
	//   }
	//
	//
	//
	// }
