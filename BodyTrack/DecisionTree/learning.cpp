#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem> // std::tr2::sys::path etc.
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "main.h"
#include "learning.h"

using namespace std;
using namespace cv;

// number of learning images
const int NUM_LEARN_SET = 1;

// search windows
const int SIZE_WINDOW_SIZE = 64;
const float MAX_DELTA = 1.0f;

const int SENSOR_WIDTH = 640;
const int SENSOR_HEIGHT = 480;

struct Image{
    float data[SENSOR_WIDTH*SENSOR_HEIGHT];
};

Image* g_pimages;
Node g_rootNode[NUM_TREE];

float log2f(float antilog) {
    return logf(antilog) / logf(2.0f);
}

void constructNode(Node* curNode, std::vector<Sample> bucket[NUM_LABELING], int depth, float entropy)
{
    int i;
    
    if( entropy == 0.0f ){
        // create leaf node
        curNode->leaf = true;
        curNode->depth = depth;
        for (i = 0; i<NUM_LABELING; i++){
            curNode->histogram[i] = bucket[i].size();
            bucket[i].clear();
        }
        printf("Create leaf node due to entropy == 0.0f\n");
        return;
    }

    if(depth >= MAX_TREE_DEPTH){
        // create leaf node
        curNode->leaf = true;
        curNode->depth = depth;
        for (i = 0; i<NUM_LABELING; i++){
            curNode->histogram[i] = bucket[i].size();
            bucket[i].clear();
        }
        printf("Create leaf node due to max depth\n");
        return;
    }

    printf("start entropy is %f\n", entropy);

    Metric minmetric;
    minmetric.entropy = entropy;

    int u[2], v[2];
    int posu[2], posv[2];
    float delta;

    int numsample[2];
    int numbucket[NUM_LABELING][2];

    bool findbetterentropy = false;
    bool* flag = new bool[curNode->numsample];
    
    // for each u, v, delta,
    for(u[1]=-SIZE_WINDOW_SIZE/2; u[1]<SIZE_WINDOW_SIZE/2; u[1]+=4){
        printf("u[i] = %d", u[1]);
        for(u[0]=-SIZE_WINDOW_SIZE/2; u[0]<SIZE_WINDOW_SIZE/2; u[0]+=4){
            for(v[1]=-SIZE_WINDOW_SIZE/2; v[1]<SIZE_WINDOW_SIZE/2; v[1]+=4){
                for(v[0]=-SIZE_WINDOW_SIZE/2; v[0]<SIZE_WINDOW_SIZE/2; v[0]+=4){
                    if( (v[0] == u[0]) && (v[1] == u[1]) )
                        continue;
                    for(delta=0; delta<MAX_DELTA; delta+=0.05f){ // 1cm
                        // for each sample, calculate true / false
                        int idxbucket, idxbucketsample;
                        numsample[0] = numsample[1] = 0;
                        for (idxbucket = 0; idxbucket<NUM_LABELING; idxbucket++){
                            numbucket[idxbucket][0] = numbucket[idxbucket][1] = 0;
                        }

                        for (idxbucket = 0; idxbucket<NUM_LABELING; idxbucket++){
                            for(idxbucketsample=0; idxbucketsample<bucket[idxbucket].size(); idxbucketsample++ ){
                                int uvalue, vvalue;
                                
                                posu[0] = bucket[idxbucket][idxbucketsample].x + u[0];
                                posu[1] = bucket[idxbucket][idxbucketsample].y + u[1];
                                if( (posu[0] < 0) || (posu[0] >= SENSOR_WIDTH) || (posu[1] < 0) || (posu[1] >= SENSOR_HEIGHT)  ){
                                    uvalue = 0;
                                }else{
                                    uvalue = (int)g_pimages[bucket[idxbucket][idxbucketsample].imgIdx].data[posu[0]+posu[1]*SENSOR_WIDTH];
                                }
                                posv[0] = bucket[idxbucket][idxbucketsample].x + v[0];
                                posv[1] = bucket[idxbucket][idxbucketsample].y + v[1];
                                if( (posv[0] < 0) || (posv[0] >= SENSOR_WIDTH) || (posv[1] < 0) || (posv[1] >= SENSOR_HEIGHT)  ){
                                    vvalue = 0;
                                }else{
                                    vvalue = (int)g_pimages[bucket[idxbucket][idxbucketsample].imgIdx].data[posv[0]+posv[1]*SENSOR_WIDTH];
                                }

                                if(uvalue - vvalue < delta){
                                    numsample[0]++;
                                    numbucket[idxbucket][0]++;
                                }else{
                                    numsample[1]++;
                                    numbucket[idxbucket][1]++;
                                }
                            }
                        }

                        // calculate entropy for left and right
                        float leftentropy = 0.0f, rightentropy = 0.0f;
                        for (idxbucket = 0; idxbucket<NUM_LABELING; idxbucket++){
                            float tmp = 0.0f;
                            if((numsample[0] != 0) && (numbucket[idxbucket][0] != 0)){
                                tmp = numbucket[idxbucket][0] / (float)numsample[0];
                                leftentropy -= tmp * log2f(tmp);
                            }
                            tmp = 0.0f;
                            if((numsample[1] != 0) && (numbucket[idxbucket][1] != 0)){
                                tmp = numbucket[idxbucket][1] / (float)numsample[1];
                                rightentropy -= tmp * log2f(tmp);
                            }
                        }
                        float tmpentropy = leftentropy * numsample[0] / (float)curNode->numsample +
                            rightentropy * numsample[1] / (float)curNode->numsample;

                        if(tmpentropy < minmetric.entropy){
                            findbetterentropy = true;
                            minmetric.u[0] = u[0];
                            minmetric.u[1] = u[1];
                            minmetric.v[0] = v[0];
                            minmetric.v[1] = v[1];
                            minmetric.delta = delta;
                            minmetric.entropy = tmpentropy;
                            minmetric.leftentropy = leftentropy;
                            minmetric.rightentropy = rightentropy;
                            printf("entropy = %f, left = %d, left entropy = %f, right = %d, right entropy=%f\n",
                                tmpentropy, numsample[0], leftentropy, numsample[1], rightentropy);
                        }
                    }
                }
            }
        }
    }

    if(findbetterentropy == false){
        // create leaf node
        curNode->leaf = true;
        curNode->depth = depth;
        for (i = 0; i<NUM_LABELING; i++){
            curNode->histogram[i] = bucket[i].size();
            bucket[i].clear();
        }
        printf("Create leaf node due to no more good split\n");
        return;
    }

    printf("reassign all samples!\n");
    std::vector<Sample> leftbucket[NUM_LABELING];
    std::vector<Sample> rightbucket[NUM_LABELING];
    Node* nextNode = new Node[2];
    nextNode[0].numsample = 0;
    nextNode[1].numsample = 0;
    {
        // for each sample, calculate true / false
        int idxbucket, idxbucketsample;
        numsample[0] = numsample[1] = 0;
        for (i = 0; i<NUM_LABELING; i++){
            numbucket[i][0] = numbucket[i][1] = 0;
        }

        for (idxbucket = 0; idxbucket<NUM_LABELING; idxbucket++){
            for(idxbucketsample=0; idxbucketsample<bucket[idxbucket].size(); idxbucketsample++ ){
                int uvalue, vvalue;

                posu[0] = bucket[idxbucket][idxbucketsample].x + minmetric.u[0];
                posu[1] = bucket[idxbucket][idxbucketsample].y + minmetric.u[1];
                if( (posu[0] < 0) || (posu[0] >= SENSOR_WIDTH) || (posu[1] < 0) || (posu[1] >= SENSOR_HEIGHT)  ){
                    uvalue = 0;
                }else{
                    uvalue = (int)g_pimages[bucket[idxbucket][idxbucketsample].imgIdx].data[posu[0]+posu[1]*SENSOR_WIDTH];
                }
                posv[0] = bucket[idxbucket][idxbucketsample].x + minmetric.v[0];
                posv[1] = bucket[idxbucket][idxbucketsample].y + minmetric.v[1];
                if( (posv[0] < 0) || (posv[0] >= SENSOR_WIDTH) || (posv[1] < 0) || (posv[1] >= SENSOR_HEIGHT)  ){
                    vvalue = 0;
                }else{
                    vvalue = (int)g_pimages[bucket[idxbucket][idxbucketsample].imgIdx].data[posv[0]+posv[1]*SENSOR_WIDTH];
                }

                if(uvalue - vvalue < minmetric.delta){
                    nextNode[0].numsample++;
                    leftbucket[idxbucket].push_back(bucket[idxbucket][idxbucketsample]);
                }else{
                    nextNode[1].numsample++;
                    rightbucket[idxbucket].push_back(bucket[idxbucket][idxbucketsample]);
                }
            }
        }
    }
    
//    printf("done!\n");
    delete[] flag;

    // release all parent bucket item
    for (i = 0; i<NUM_LABELING; i++){
        bucket[i].clear();
    }

    curNode->leaf = false;
    curNode->depth = depth;
    curNode->leftNode = &nextNode[0];
    curNode->rightNode = &nextNode[1];
    curNode->metric = minmetric;
    printf("Left entropy is %f, Right entropy is %f\n", minmetric.leftentropy, minmetric.rightentropy);
    constructNode(&nextNode[0], leftbucket, depth+1, minmetric.leftentropy);
    constructNode(&nextNode[1], rightbucket, depth+1, minmetric.rightentropy);
}

void dumpNode(FILE* fp, Node* node){
    // leaf depth u[2] v[2] delta
    if(node->leaf == true){
        int i;
        fprintf(fp, "1 %d", node->depth);
        for (i = 0; i<NUM_LABELING; i++){
            fprintf(fp, " %d", node->histogram[i]);
        }
            fprintf(fp, "\n");
    }else{
        fprintf(fp, "0 %d %d %d %d %d %f\n", node->depth, node->metric.u[0],
            node->metric.u[1], node->metric.v[0], node->metric.v[1], node->metric.delta);
        dumpNode(fp, node->leftNode);
        dumpNode(fp, node->rightNode);
    }
}

void ProcessLearning()
{
    char filepath[128];
    Mat colored_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    std::vector<Sample> sample_list[NUM_LABELING];

    unsigned char colors[6][3] =
    {
        { 196, 0, 60 },
        { 60, 196, 0 },
        { 0, 60, 196 },
        { 196, 128, 128 },
        { 128, 196, 196 },
        { 128, 128, 255 },
    };

    g_pimages = new Image[NUM_LEARN_SET];
    
    // load samples
    for (int i = 0; i < NUM_LEARN_SET; i++){
        sprintf_s(filepath, "input/frame%.3d.txt", i);

        FILE* fp;
        fopen_s(&fp, filepath, "r");
        for (int j = 0; j < SENSOR_WIDTH * SENSOR_HEIGHT; j++){
            fscanf_s(fp, "%f ", &g_pimages[i].data[j]);
            g_pimages[i].data[j] /= 1000.0f; // milli meter -> meter
        }
        fclose(fp);

        sprintf_s(filepath, "labeling/labeled%.3d.ppm", i);
        colored_image = imread(filepath);
        for (int j = 0; j < SENSOR_WIDTH * SENSOR_HEIGHT; j++){
            for (int k = 0; k < NUM_LABELING; k++){
                if ((colored_image.data[j * 3] == colors[k][0]) &&
                    (colored_image.data[j * 3 + 1] == colors[k][1]) &&
                    (colored_image.data[j * 3 + 2] == colors[k][2]))
                {
                    Sample sample;
                    sample.imgIdx = i;
                    sample.x = j % SENSOR_WIDTH;
                    sample.y = j / SENSOR_WIDTH;
                    sample_list[k].push_back(sample);
                }
            }
        }
    }

    for (int i = 0; i < NUM_TREE; i++){
        std::vector<Sample> tree_list[NUM_LABELING];
        int num_samples = NUM_SAMPLES  / NUM_TREE;
        float interval = 1.0f / num_samples;
        float startup = interval / NUM_TREE * i;

        for (int j = 0; j < NUM_LABELING; j++){
            for (int k = 0; k < num_samples; k++){
                int index = sample_list[j].size() * interval * (k + startup);
                tree_list[j].push_back(sample_list[j][index]);
            }
        }

        float entropy = 0.0f;
        for (int j = 0; j < NUM_LABELING; j++){
            float tmp = 1.0f / (float)NUM_LABELING;
            entropy -= tmp * log2f(tmp);
        }

        g_rootNode[i].numsample = num_samples * NUM_LABELING;
        constructNode(&g_rootNode[i], tree_list, 0, entropy);

        sprintf_s(filepath, "deciontree/dt%.3d.txt", i);
        FILE* fp;
        fopen_s(&fp, filepath, "w");
        dumpNode(fp, &g_rootNode[i]);
        fclose(fp);
    }
}
