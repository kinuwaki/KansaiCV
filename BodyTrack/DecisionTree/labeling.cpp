#include <iostream>
#include <algorithm>
#include <filesystem> // std::tr2::sys::path etc.
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <math.h>
#include "main.h"
#include "labeling.h"

using namespace std;
using namespace cv;

KinectSkelton gKinectSkelton;

enum BodyLabel{
    HEAD,
    BODY,
    RIGHT_HAND,
    LEFT_HAND,
    RIGHT_LEG,
    LEFT_LEG
};

unsigned char colors[6][3] =
{
    { 196, 0, 60 },
    { 60, 196, 0 },
    { 0, 60, 196 },
    { 196, 128, 128 },
    { 128, 196, 196 },
    { 128, 128, 255 },
};

double line_seg_point_distance(float a[2], float b[2], float p[2])
{
    double dx, dy, r2;
    double t, cx, cy;
    dx = b[0] - a[0];
    dy = b[1] - a[1];
    if (dx == 0 && dy == 0)
        return sqrt((p[0] - a[0]) * (p[0] - a[0]) + (p[1] - a[1]) * (p[1] - a[1]));
    r2 = dx * dx + dy * dy;
    t = (dx * (p[0] - a[0]) + dy * (p[1] - a[1])) / (double)r2;
    if (t < 0)
        return sqrt((p[0] - a[0]) * (p[0] - a[0]) + (p[1] - a[1]) * (p[1] - a[1]));

    if (t > 1)
        return sqrt((p[0] - b[0]) * (p[0] - b[0]) + (p[1] - b[1]) * (p[1] - b[1]));
    cx = (1 - t) * a[0] + t * b[0];
    cy = (1 - t) * a[1] + t * b[1];
    return sqrt((p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy));
}

void ProcessLabeling()
{
    bool change = true;

    // opencv initialization
    Mat depth_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16UC1);
    Mat clustered_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    cv::namedWindow("depth_image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
    cv::namedWindow("labeled_image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
    
    // add bar to threshold
    const int threshold1 = 2000; // 2m = 2000mm
    const double threshold2 = 25.0f / 640.0f; // 50 pixels

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

            // begin of NN for each pixel
            for (int i = 0; i < IMAGE_HEIGHT; i++){
                for (int j = 0; j < IMAGE_WIDTH; j++){
                    if (depth_image.at<unsigned short>(i, j) < threshold1){
                        float current[2];
                        current[0] = j / (float)IMAGE_WIDTH;
                        current[1] = i / (float)IMAGE_HEIGHT;

                        {
                            // left hand
                            if ((line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_SHOULDER_RIGHT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_ELBOW_RIGHT], current) < threshold2) ||
                                (line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_ELBOW_RIGHT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_WRIST_RIGHT], current) < threshold2)){
                                memcpy(&clustered_image.data[(j + i*IMAGE_WIDTH) * 3], colors[0], 3);
                                /*
                                line(clustered_image,
                                Point(gKinectSkelton.positions[NUI_SKELETON_POSITION_SHOULDER_RIGHT][0] * IMAGE_WIDTH,
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_SHOULDER_RIGHT][1] * IMAGE_HEIGHT),
                                Point(gKinectSkelton.positions[NUI_SKELETON_POSITION_ELBOW_RIGHT][0] * IMAGE_WIDTH,
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_ELBOW_RIGHT][1] * IMAGE_HEIGHT),
                                Scalar(255, 0, 255));
                                */
                            } // right hand
                            else if ((line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_SHOULDER_LEFT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_ELBOW_LEFT], current) < threshold2) ||
                                (line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_ELBOW_LEFT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_WRIST_LEFT], current) < threshold2)){
                                memcpy(&clustered_image.data[(j + i*IMAGE_WIDTH) * 3], colors[1], 3);
                            } // left foot
                            else if ((line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_HIP_RIGHT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_KNEE_RIGHT], current) < threshold2) ||
                                (line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_KNEE_RIGHT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_ANKLE_RIGHT], current) < threshold2)){
                                memcpy(&clustered_image.data[(j + i*IMAGE_WIDTH) * 3], colors[2], 3);
                            } // right foot
                            else if ((line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_HIP_LEFT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_KNEE_LEFT], current) < threshold2) ||
                                (line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_KNEE_LEFT],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_ANKLE_LEFT], current) < threshold2)){
                                memcpy(&clustered_image.data[(j + i*IMAGE_WIDTH) * 3], colors[3], 3);
                            } // head
                            else if (sqrt((gKinectSkelton.positions[NUI_SKELETON_POSITION_HEAD][0] - current[0]) *
                                (gKinectSkelton.positions[NUI_SKELETON_POSITION_HEAD][0] - current[0]) +
                                (gKinectSkelton.positions[NUI_SKELETON_POSITION_HEAD][1] - current[1]) *
                                (gKinectSkelton.positions[NUI_SKELETON_POSITION_HEAD][1] - current[1])) < threshold2 * 2){
                                memcpy(&clustered_image.data[(j + i*IMAGE_WIDTH) * 3], colors[4], 3);
                            } // body
                            else if (line_seg_point_distance(gKinectSkelton.positions[NUI_SKELETON_POSITION_HIP_CENTER],
                                gKinectSkelton.positions[NUI_SKELETON_POSITION_SHOULDER_CENTER], current) < threshold2 * 3){
                                memcpy(&clustered_image.data[(j + i*IMAGE_WIDTH) * 3], colors[5], 3);
                            }
                        }
                    }
                }
            }
            // end of NN for each pixel

            sprintf_s(filepath, "./labeling/labeled%.3d.ppm", currentIdx);
            imwrite(filepath, clustered_image);

            change = 0;
            currentIdx++;
        }

        cv::imshow("depth_image", depth_image);
        cv::imshow("labeled_image", clustered_image);

        int c = -1;
        cv::waitKey(0);
    }

}