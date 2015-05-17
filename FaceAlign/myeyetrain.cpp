#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"

#define LEARNING_SIZEX 384
#define LEARNING_SIZEY 286

const int MAX_TREE_DEPTH = 2;

int NUM_IMAGE = 0;
vector<Patch> g_imagePatches;

void dumpPGMPatch(char* filename, unsigned char* data)
{
    FILE* fp;
    fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%d %d\n255\n", PATCH_SIZEX, PATCH_SIZEY);
    fwrite(data, sizeof(char), PATCH_SIZEX*PATCH_SIZEY, fp);
    fclose(fp);
}

const int MAX_DELTA = 200;
void constructNode(int landmark, Node* curNode, vector<int> bin, int depth, double variance)
{
    int i;

    if( variance == 0.0f ){
        // create leaf node
        curNode->leaf = true;
        curNode->depth = depth;
        bin.clear();
        printf("Create leaf node due to variance == 0.0f\n");
        return;
    }

	if( bin.size() < MIN_SAMPLE_THRESHOLD ){
        // create leaf node
        curNode->leaf = true;
        curNode->depth = depth;
        bin.clear();
        printf("Create leaf node due to small samples\n");
        return;
    }

    if(depth >= MAX_TREE_DEPTH){
        // create leaf node
        curNode->leaf = true;
        curNode->depth = depth;
        bin.clear();
        printf("Create leaf node due to max depth\n");
        return;
    }
    
    Metric minmetric;
    minmetric.variance = variance;

    int u[2], v[2];
    int posu[2], posv[2];
    int delta;

    int numsample[2];
    double valuebucket[2][2];

    bool findbetterentropy = false;
    bool* flag = new bool[curNode->numsample];
    
    // for each u, v, delta,
    for(u[1]=0; u[1]<PATCH_SIZEY; u[1]+=4){
        printf("u[1] = %d", u[1]);
        for(u[0]=0; u[0]<PATCH_SIZEX; u[0]+=4){
            for(v[1]=0; v[1]<PATCH_SIZEY; v[1]+=4){
                for(v[0]=0; v[0]<PATCH_SIZEX; v[0]+=4){
                    if( (v[0] == u[0]) && (v[1] == u[1]) )
                        continue;
                    for(delta=0; delta<MAX_DELTA; delta+=5){
                        // for each sample, calculate true / false
                        vector<int> bucket[2];
                        int idxbucket, idxbucketsample;
                        numsample[0] = numsample[1] = 0;
                        valuebucket[0][0] = valuebucket[0][1] = valuebucket[1][0] = valuebucket[1][1] = 0.0;

                        for(idxbucketsample=0; idxbucketsample<bin.size(); idxbucketsample++ ){
                            int uvalue, vvalue;

                            posu[0] = u[0];
                            posu[1] = u[1];
                            uvalue = (int)g_imagePatches[bin[idxbucketsample]].image[posu[0]+posu[1]*PATCH_SIZEX];

                            posv[0] = v[0];
                            posv[1] = v[1];
                            vvalue = (int)g_imagePatches[bin[idxbucketsample]].image[posv[0]+posv[1]*PATCH_SIZEX];

                            if(uvalue - vvalue < delta){
                                numsample[0]++;
                                valuebucket[0][0]+=g_imagePatches[bin[idxbucketsample]].correct_offset[landmark][0];
                                valuebucket[0][1]+=g_imagePatches[bin[idxbucketsample]].correct_offset[landmark][1];
                                bucket[0].push_back(bin[idxbucketsample]);
                            }else{
                                numsample[1]++;
                                valuebucket[1][0]+=g_imagePatches[bin[idxbucketsample]].correct_offset[landmark][0];
                                valuebucket[1][1]+=g_imagePatches[bin[idxbucketsample]].correct_offset[landmark][1];
                                bucket[1].push_back(bin[idxbucketsample]);
                            }
                        }

                        if(numsample[0] != 0){
                            valuebucket[0][0] /= (double)numsample[0];
                            valuebucket[0][1] /= (double)numsample[0];
                        }
                        if(numsample[1] != 0){
                            valuebucket[1][0] /= (double)numsample[1];
                            valuebucket[1][1] /= (double)numsample[1];
                        }

                        // calculate variance for left and right
                        float leftvariance = 0.0f, rightvariance = 0.0f;
                        if(numsample[0] != 0){
                            for(i=0; i<bucket[0].size(); i++){
                                leftvariance += (g_imagePatches[bucket[0][i]].correct_offset[landmark][0] - valuebucket[0][0]) *
                                    (g_imagePatches[bucket[0][i]].correct_offset[landmark][0] - valuebucket[0][0]) +
                                    (g_imagePatches[bucket[0][i]].correct_offset[landmark][1] - valuebucket[0][1]) *
                                    (g_imagePatches[bucket[0][i]].correct_offset[landmark][1] - valuebucket[0][1]);
                            }
                        }
                        if(numsample[1] != 0){
                            for(i=0; i<bucket[1].size(); i++){
                                rightvariance += (g_imagePatches[bucket[1][i]].correct_offset[landmark][0] - valuebucket[1][0]) *
                                    (g_imagePatches[bucket[1][i]].correct_offset[landmark][0] - valuebucket[1][0]) +
                                    (g_imagePatches[bucket[1][i]].correct_offset[landmark][1] - valuebucket[1][1]) *
                                    (g_imagePatches[bucket[1][i]].correct_offset[landmark][1] - valuebucket[1][1]);
                            }
                        }
                        float tmpvariance = leftvariance + rightvariance;

                        if(tmpvariance < minmetric.variance){
                            findbetterentropy = true;
                            minmetric.u[0] = u[0];
                            minmetric.u[1] = u[1];
                            minmetric.v[0] = v[0];
                            minmetric.v[1] = v[1];
                            minmetric.delta = delta;
                            minmetric.variance = tmpvariance;
                            minmetric.leftvalue[0] = valuebucket[0][0];
                            minmetric.leftvalue[1] = valuebucket[0][1];
                            minmetric.leftvariance = leftvariance;
                            minmetric.rightvalue[0] = valuebucket[1][0];
                            minmetric.rightvalue[1] = valuebucket[1][1];
                            minmetric.rightvariance = rightvariance;
                            printf("variance = %f, left = %d, left variance = %f, right = %d, right variance=%f\n",
                                tmpvariance, numsample[0], leftvariance, numsample[1], rightvariance);
                        }
                        bucket[0].clear();
                        bucket[1].clear();
                    }
                }
            }
        }
    }

    if(findbetterentropy == false){
        // create leaf node
        curNode->leaf = true;
        curNode->depth = depth;
        bin.clear();
        printf("Create leaf node due to no more good split\n");
        return;
    }

    printf("reassign all samples!\n");
    vector<int> leftbin, rightbin;
    Node* nextNode = new Node[2];
    nextNode[0].numsample = 0;
    nextNode[1].numsample = 0;
    {
        // for each sample, calculate true / false
        int idxbucket, idxbucketsample;
        numsample[0] = numsample[1] = 0;

        for(idxbucketsample=0; idxbucketsample<bin.size(); idxbucketsample++ ){
            int uvalue, vvalue;
            posu[0] = minmetric.u[0];
            posu[1] = minmetric.u[1];
            uvalue = (int)g_imagePatches[bin[idxbucketsample]].image[posu[0]+posu[1]*PATCH_SIZEX];

            posv[0] = minmetric.v[0];
            posv[1] = minmetric.v[1];
            vvalue = (int)g_imagePatches[bin[idxbucketsample]].image[posv[0]+posv[1]*PATCH_SIZEX];

            if(uvalue - vvalue < minmetric.delta){
                nextNode[0].numsample++;
                leftbin.push_back(bin[idxbucketsample]);
            }else{
                nextNode[1].numsample++;
                rightbin.push_back(bin[idxbucketsample]);
            }
        }
    }
    
//    printf("done!\n");
    delete[] flag;

    // release all parent bucket item
    bin.clear();

    curNode->leaf = false;
    curNode->depth = depth;
    curNode->leftNode = &nextNode[0];
    curNode->metric = minmetric;
    nextNode[0].result[0] = minmetric.leftvalue[0];
    nextNode[0].result[1] = minmetric.leftvalue[1];
    curNode->rightNode = &nextNode[1];
    nextNode[1].result[0] = minmetric.rightvalue[0];
    nextNode[1].result[1] = minmetric.rightvalue[1];
    printf("Left variance is %f, Right variance is %f\n", minmetric.leftvariance, minmetric.rightvariance);
    constructNode(landmark, &nextNode[0], leftbin, depth+1, minmetric.leftvariance);
    constructNode(landmark, &nextNode[1], rightbin, depth+1, minmetric.rightvariance);
}

void dumpNode(FILE* fp, Node* node){
    // leaf depth u[2] v[2] delta
    if(node->leaf == true){
        int i;
        fprintf(fp, "1 %d %f %f\n", node->depth, node->result[0], node->result[1]);
    }else{
        fprintf(fp, "0 %d %d %d %d %d %d\n", node->depth, node->metric.u[0],
            node->metric.u[1], node->metric.v[0], node->metric.v[1], node->metric.delta);
        dumpNode(fp, node->leftNode);
        dumpNode(fp, node->rightNode);
    }
}

void MyEyeTrain()
{
    int i, j, k, idx;
    g_imagePatches.resize(MAX_IMAGE_IDX);

	// load all image set
    for(i=0; i<MAX_IMAGE_IDX; i++){
		if(i % 100 == 0)
			printf("%dth image is loaded\n", i);
        // We study X % != 4 frame (1000 images)
        if(i % RATIO_TRAIN2TEST != 0){ 
            char filename[256];
            sprintf(filename, "./database/BioID_%.4d.pgm", i);

            // CV face detection
            std::vector<Rect> faces;
            Mat equalized_image;
            Mat image = imread(filename, IMREAD_GRAYSCALE);
            equalizeHist( image, equalized_image );
            face_cascade.detectMultiScale( equalized_image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

            if(faces.size() == 1){
                float ref_pos[NUM_LANDMARKS][2];
                sprintf(filename, "./database/bioid_%.4d.pts", i);
                FILE* fp = fopen(filename, "r");
                fscanf(fp, "version: 1\nn_points: 20\n{\n");
                fscanf(fp, "%f %f\n%f %f\n", &ref_pos[0][0], &ref_pos[0][1], &ref_pos[1][0], &ref_pos[1][1]);
                fclose(fp);

                float init_pos[NUM_LANDMARKS][2];
                init_pos[0][0] = faces[0].x + SK_FACEINIT::relative_lefteye_position[0]*faces[0].width;
                init_pos[0][1] = faces[0].y + SK_FACEINIT::relative_lefteye_position[1]*faces[0].height;
                init_pos[1][0] = faces[0].x + SK_FACEINIT::relative_righteye_position[0]*faces[0].width;
                init_pos[1][1] = faces[0].y + SK_FACEINIT::relative_righteye_position[1]*faces[0].height;

                int init_central[2];
                init_central[0] = (init_pos[0][0] + init_pos[1][0] ) / 2.0f;
                init_central[1] = (init_pos[0][1] + init_pos[1][1] ) / 2.0f;

                idx = 0;
                int half_patch_x = PATCH_SIZEX / 2;
                int half_patch_y = PATCH_SIZEY / 2;
                for(j=-half_patch_y; j<=half_patch_y; j++){
                    for(k=-half_patch_x; k<=half_patch_x; k++){
                        int x = init_central[0] + k;
                        int y = init_central[1] + j;

                        int size = (int)image.dataend - (int)image.datastart;
                        if( (x < 0) || (x >= LEARNING_SIZEX) || (y < 0) || (y >= LEARNING_SIZEY)){
                            g_imagePatches[NUM_IMAGE].image[idx] = 127;
                        }else{
                            g_imagePatches[NUM_IMAGE].image[idx] = image.data[x+y*LEARNING_SIZEX];
                        }
                        idx++;
                    }
                }
//                sprintf(filename, "./patch/patch_%.4d.pgm", NUM_IMAGE);
//                dumpPGMPatch(filename, g_imagePatches[NUM_IMAGE].image);

                // correct offset
                g_imagePatches[NUM_IMAGE].correct_offset[0][0] = ref_pos[0][0] - init_pos[0][0];
                g_imagePatches[NUM_IMAGE].correct_offset[0][1] = ref_pos[0][1] - init_pos[0][1];
                g_imagePatches[NUM_IMAGE].correct_offset[1][0] = ref_pos[1][0] - init_pos[1][0];
                g_imagePatches[NUM_IMAGE].correct_offset[1][1] = ref_pos[1][1] - init_pos[1][1];

                NUM_IMAGE++;
            }
        }
    }
    
	for(j=0; j<NUM_TREE; j++){
		printf("\n----- Constructing %dth Regression Tree -----\n", j);
		for(i=0; i<NUM_LANDMARKS; i++){
			vector<int> index_list; 
			float initialVariance = 0.0f;

			// calculate average offset and compute variance
			int num_image = 0;
			int value;
			float average[2] = {0.0f, 0.0f};
			for(value=0; value<NUM_IMAGE; value++){
				if(value % NUM_TREE == j){
					average[0] += g_imagePatches[value].correct_offset[i][0];
					average[1] += g_imagePatches[value].correct_offset[i][1];
					num_image++;
				}
			}
			average[0] /= (float)num_image;
			average[1] /= (float)num_image;

			// compute init variance
			for(value=0; value<NUM_IMAGE; value++){
				if(value % NUM_TREE == j){
					initialVariance += (average[0] - g_imagePatches[value].correct_offset[i][0]) *
						(average[0] - g_imagePatches[value].correct_offset[i][0]) +
						(average[1] - g_imagePatches[value].correct_offset[i][1]) *
						(average[1] - g_imagePatches[value].correct_offset[i][1]);
					index_list.push_back(value);
				}
			}

			// first experience
			rootNode[i][j] = new Node;
			rootNode[i][j]->leaf = false;
			rootNode[i][j]->depth = 0;
			rootNode[i][j]->numsample = num_image;

			printf("Initial variance is %f\n", initialVariance);

			constructNode(i, rootNode[i][j], index_list, rootNode[i][j]->depth, initialVariance);

			// dump regression forest
			char filename[100];
			sprintf(filename, "./regressiontree/%.3d/rt%.3d.txt",i, j);
			printf("Dumping Random Forest %d's %d th\n", i);
			FILE* fp = fopen(filename, "w");
			dumpNode(fp, rootNode[i][j]);
			fclose(fp);
		}
	}
}
