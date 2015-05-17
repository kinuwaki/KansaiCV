#pragma once

using namespace cv;
using namespace std;

// own implementation
namespace SK_FACEINIT{
    const float relative_lefteye_position[2] = {0.3f, 0.35f};
    const float relative_righteye_position[2] = {0.7f, 0.35f};
};

extern String face_cascade_name;
extern CascadeClassifier face_cascade;

// Regression Forest
const int NUM_LANDMARKS = 2;
const int NUM_TREE = 4;

const int RATIO_TRAIN2TEST = 4;
const int MIN_SAMPLE_THRESHOLD = 5;

struct Metric{
    double variance;
    double leftvalue[2], rightvalue[2];
    double leftvariance, rightvariance;
    int u[2];
    int v[2];
    int delta;
};

struct Node{
    int numsample;
    int depth;
    float result[2]; // result

    bool leaf;
    Metric metric;

    Node* leftNode;
    Node* rightNode;

    Node(){
        leaf = 0;
    }
};
extern Node* rootNode[NUM_LANDMARKS][NUM_TREE];

// Image dataset
#define MAX_IMAGE_IDX (1520)
//#define MAX_IMAGE_IDX (100)

#define USE_USB_CAMERA (0)

#define PATCH_SIZEX 129
#define PATCH_SIZEY 65

struct Patch{
    int correct_offset[NUM_LANDMARKS][2];
    unsigned char image[PATCH_SIZEX*PATCH_SIZEY];
};
extern vector<Patch> g_imagePatches;

void MyEyeTrain();

// opencv
void CvEyeInit();
void CvEyeRun(cv::Mat& image, cv::Mat& equalized_image, std::vector<cv::Rect>& faces);
