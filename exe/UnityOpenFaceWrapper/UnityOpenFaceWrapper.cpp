///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
// UnityOpenFaceWrapper.cpp : Defines the entry point for the console application for detecting landmarks in images.

// dlib
#include <dlib/image_processing/frontal_face_detector.h>

#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>
#include <GazeEstimation.h>

#include <ImageCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>

// For JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;


#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

#define PINVOKE_ENTRY_POINT __declspec(dllexport)

struct CameraCalibrationParams
{
	float fx = -1.0; // focal length X
	float fy = -1.0; // focal length Y
	float cx = -1.0; // camera principal point X
	float cy = -1.0; // camera principal point Y
};

class OpenFaceParams
{
public:
	LandmarkDetector::FaceModelParameters* faceModelParams = nullptr;
	FaceAnalysis::FaceAnalyserParameters* faceAnalyserParams = nullptr;
	FaceAnalysis::FaceAnalyser* faceAnalyser = nullptr;
	LandmarkDetector::CLNF* faceModel = nullptr;
	CameraCalibrationParams cameraCalib;

	cv::CascadeClassifier* classifier;
	dlib::frontal_face_detector face_detector_hog;
	LandmarkDetector::FaceDetectorMTCNN* face_detector_mtcnn;

	bool isClosed() {
		return faceModelParams == nullptr &&
			faceAnalyserParams == nullptr &&
			faceAnalyser == nullptr &&
			faceModel == nullptr &&
			classifier == nullptr &&
			face_detector_mtcnn == nullptr;
	}

	bool Close()
	{
		free(faceAnalyser);
		free(faceAnalyserParams);
		free(faceModel);
		free(faceModelParams);
		free(classifier);
		free(face_detector_mtcnn);
		faceAnalyser = nullptr;
		faceAnalyserParams = nullptr;
		faceModel = nullptr;
		faceModelParams = nullptr;
		classifier = nullptr;
		face_detector_mtcnn = nullptr;

		return true;
	}
};

static OpenFaceParams s_openFaceParams;

byte* matToBytes(cv::Mat image)
{
	int size = image.total() * image.elemSize();
	byte* bytes = new byte[size];  // you will have to delete[] that later
	std::memcpy(bytes, image.data, size * sizeof(byte));
	return bytes;
}

cv::Mat bytesToMat(byte* bytes, int width, int height)
{
	cv::Mat image = cv::Mat(height, width, CV_8UC4, bytes).clone(); // make a copy
	return image;
}

extern "C" {
	PINVOKE_ENTRY_POINT bool __stdcall OpenFaceSetup()
	{
		if (!s_openFaceParams.isClosed())
		{
			std::cout << "ERROR: You must close the OpenFace wrapper before setting up a new one." << std::endl;
			return false;
		}

		// Empty arguments for now
		std::vector<std::string> arguments;
		arguments.push_back("C:\\DEV\\HACKWEEK\\OpenFaceUnity\\x64\\Debug\\FaceLandmarkImg.exe"); // TODO: fix this dirty hack

		// Load the models if images found
		s_openFaceParams.faceModelParams = new LandmarkDetector::FaceModelParameters(arguments);
		auto det_parameters = *s_openFaceParams.faceModelParams;

		// The modules that are being used for tracking
		std::cout << "Loading the model" << std::endl;
		s_openFaceParams.faceModel = new LandmarkDetector::CLNF(det_parameters.model_location);
		auto face_model = *s_openFaceParams.faceModel;

		if (!face_model.loaded_successfully)
		{
			std::cout << "ERROR: Could not load the landmark detector" << std::endl;
			return false;
		}

		std::cout << "Model loaded" << std::endl;

		// Load facial feature extractor and AU analyser (make sure it is static)
		s_openFaceParams.faceAnalyserParams = new FaceAnalysis::FaceAnalyserParameters(arguments);
		auto face_analysis_params = *s_openFaceParams.faceAnalyserParams;

		face_analysis_params.OptimizeForImages();
		s_openFaceParams.faceAnalyser = new FaceAnalysis::FaceAnalyser(face_analysis_params);

		// If bounding boxes not provided, use a face detector
		s_openFaceParams.classifier = new cv::CascadeClassifier(det_parameters.haar_face_detector_location);
		s_openFaceParams.face_detector_hog = dlib::get_frontal_face_detector();
		s_openFaceParams.face_detector_mtcnn = new LandmarkDetector::FaceDetectorMTCNN(det_parameters.mtcnn_face_detector_location);

		// If can't find MTCNN face detector, default to HOG one
		if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR && s_openFaceParams.face_detector_mtcnn->empty())
		{
			std::cout << "INFO: defaulting to HOG-SVM face detector" << std::endl;
			det_parameters.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
		}

		// Ready to analyze
		return true;
	}

	// Return JSON string?
	// The pixels are in ARGB32 format
	PINVOKE_ENTRY_POINT bool __stdcall OpenFaceGetFeatures(const char* pixels, int width, int height, char* jsonData, int jsonDataLength)
	{
		if (s_openFaceParams.isClosed())
		{
			std::cout << "ERROR: Setup() was not called properly" << std::endl;
			return false;
		}
	
		cv::Mat rgba_image = bytesToMat((byte*)pixels, width, height);
		cv::Mat rgb_image = cv::Mat();
		// Convert to RGB24. Strangely enough we expect RGBA input but we get BGRA
		cv::cvtColor(rgba_image, rgb_image, cv::COLOR_BGRA2RGB);
		// Flip vertically?
		cv::flip(rgb_image, rgb_image, 0);

		// Save the image to file for tests
#ifdef false
		auto path = "C:/DEV/HACKWEEK/test.jpg";
		bool okWrite = cv::imwrite(path, rgb_image); // A JPG FILE IS BEING SAVED
#endif

		auto face_model = *s_openFaceParams.faceModel;
		auto face_analyser = *s_openFaceParams.faceAnalyser;
		auto det_parameters = *s_openFaceParams.faceModelParams;

		if (!face_model.eye_model)
		{
			std::cout << "WARNING: no eye model found" << std::endl;
		}

		if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAUClassNames().size() == 0)
		{
			std::cout << "WARNING: no Action Unit models found" << std::endl;
		}

		std::cout << "Starting tracking" << std::endl;
		bool has_bounding_boxes = false;
		float fx = s_openFaceParams.cameraCalib.fx;
		float fy = s_openFaceParams.cameraCalib.fy;
		float cx = s_openFaceParams.cameraCalib.cx;
		float cy = s_openFaceParams.cameraCalib.cy;

		// Making sure the image is in uchar grayscale (some face detectors use RGB, landmark detector uses grayscale)
		cv::Mat_<uchar> grayscale_image; // TODO: convert color to grayscale
		cv::cvtColor(rgb_image, grayscale_image, cv::COLOR_RGB2GRAY);

		// Detect faces in an image
		std::vector<cv::Rect_<float> > face_detections;

		if (has_bounding_boxes)
		{
			//face_detections = image_reader.GetBoundingBoxes();
			std::cout << "Unexpected setup (bounding boxes)" << std::endl;
			return false;
		}
		else
		{
			if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
			{
				std::vector<float> confidences;
				LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, s_openFaceParams.face_detector_hog, confidences);
			}
			else if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HAAR_DETECTOR)
			{
				LandmarkDetector::DetectFaces(face_detections, grayscale_image, *s_openFaceParams.classifier);
			}
			else
			{
				std::vector<float> confidences;
				LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, *s_openFaceParams.face_detector_mtcnn, confidences);
			}
		}

		// Detect landmarks around detected faces
		int face_det = 0;
		// perform landmark detection for every face detected
		json j;
		j["name"] = "openface";
		j["faces"] = {};
		for (size_t face = 0; face < face_detections.size(); ++face)
		{
			// if there are multiple detections go through them
			bool success = LandmarkDetector::DetectLandmarksInImage(rgb_image, face_detections[face], face_model, det_parameters, grayscale_image);

			// Estimate head pose and eye gaze				
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gaze_direction0(0, 0, -1);
			cv::Point3f gaze_direction1(0, 0, -1);
			cv::Vec2f gaze_angle(0, 0);

			if (face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gaze_direction0, fx, fy, cx, cy, true);
				GazeAnalysis::EstimateGaze(face_model, gaze_direction1, fx, fy, cx, cy, false);
				gaze_angle = GazeAnalysis::GetGazeAngle(gaze_direction0, gaze_direction1);
			}

			json jsonFace;
			jsonFace["id"] = face;
			jsonFace["pose"]["x"] = pose_estimate[0];
			jsonFace["pose"]["y"] = pose_estimate[1];
			jsonFace["pose"]["z"] = pose_estimate[2];
			jsonFace["pose"]["rotX"] = pose_estimate[3];
			jsonFace["pose"]["rotY"] = pose_estimate[4];
			jsonFace["pose"]["rotZ"] = pose_estimate[5];
			jsonFace["gaze0"]["x"] = gaze_direction0.x;
			jsonFace["gaze0"]["y"] = gaze_direction0.y;
			jsonFace["gaze0"]["z"] = gaze_direction0.z;
			jsonFace["gaze1"]["x"] = gaze_direction1.x;
			jsonFace["gaze1"]["y"] = gaze_direction1.y;
			jsonFace["gaze1"]["z"] = gaze_direction1.z;

			jsonFace["landmarks"]["success"] = success;
			jsonFace["landmarks"]["landmarks2d"] = {};
			if (success)
			{
				auto numLandmarks = face_model.detected_landmarks.size().height / 2;
				for (int i = 0; i < numLandmarks; i++)
				{
					json jsonCurrLandmark;
					jsonCurrLandmark["x"] = *face_model.detected_landmarks[i];
					jsonCurrLandmark["y"] = *face_model.detected_landmarks[i + numLandmarks];
					jsonFace["landmarks"]["landmarks2d"].push_back(jsonCurrLandmark);
				}
			}

			j["faces"].push_back(jsonFace);
		}

		std::string strResult = j.dump();
		strcpy(jsonData, strResult.c_str());

		return true;
	}

	PINVOKE_ENTRY_POINT bool __stdcall OpenFaceClose()
	{
		return s_openFaceParams.Close();
	}
}
