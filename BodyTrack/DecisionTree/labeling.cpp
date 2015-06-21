#include <iostream>
#include <algorithm>
#include <filesystem> // std::tr2::sys::path etc.
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "main.h"
#include "labeling.h"

using namespace std;
using namespace cv;

KinectSkelton gKinectSkelton;

void ProcessLabeling()
{
	bool change = true;

	// opencv initialization
	Mat depth_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16UC1);
	Mat culustered_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	cv::namedWindow("depth_image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::namedWindow("labeled_image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	
	FILE* fp;

	// add bar to threshold
	const int threshold = 2000; // 2m = 2000mm

	// for each image, we load unsigned short depth and bones
	while (1){
		if (change){
			FILE* fp;
			static int currentIdx = 0;
			static char filepath[256];

			sprintf_s(filepath, "./input/frame%.3d.txt", currentIdx);

			fopen_s(&fp, filepath, "r");

			if (fp == NULL)
				break;

			for (int i = 0; i < IMAGE_HEIGHT; i++){
				for (int j = 0; j < IMAGE_WIDTH; j++){
					int depth;
					fscanf_s(fp, "%d ", &depth);
					depth_image.at<unsigned short>(i, j) = (unsigned short)depth;
				}
			}

			for (int i = 0; i < NUI_SKELETON_POSITION_COUNT; i++){
				int tmp;
				fscanf_s(fp, "%d %f %f\n", &tmp, &gKinectSkelton.positions[i][0], &gKinectSkelton.positions[i][1]);
			}

			// perform NN for each pixel
			for (int i = 0; i < IMAGE_HEIGHT; i++){
				for (int j = 0; j < IMAGE_WIDTH; j++){
					if (depth_image.at<unsigned short>(i, j) < threshold){
						culustered_image.data[(j + i*IMAGE_WIDTH) * 3] = 255;
						culustered_image.data[(j + i*IMAGE_WIDTH) * 3 + 1] = 0;
						culustered_image.data[(j + i*IMAGE_WIDTH) * 3 + 2] = 0;
					}
				}
			}

			change = 0;
			currentIdx++;
		}

		cv::imshow("depth_image", depth_image);
		cv::imshow("labeled_image", culustered_image);
		cv::waitKey(0);
	}

}