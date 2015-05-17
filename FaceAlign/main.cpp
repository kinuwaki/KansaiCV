#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"

Node* rootNode[NUM_LANDMARKS][NUM_TREE];

// Global variables
String face_cascade_name = "./haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

void traverseNode(int x, int y, Node* pnode, Mat* pmat, int result[])
{
    int uvalue, vvalue;
    int ux, uy, vx, vy;

    if(pnode->leaf){
        result[0] = pnode->result[0];
        result[1] = pnode->result[1];
    }else{
        ux = x+pnode->metric.u[0];
        uy = y+pnode->metric.u[1];
        if((ux < 0) || (uy < 0) || (ux >= pmat->cols) || (uy >= pmat->rows))
            uvalue = 0;
        else
            uvalue = pmat->data[ux+uy*pmat->cols];
        
        vx = x+pnode->metric.v[0];
        vy = y+pnode->metric.v[1];
        if((vx < 0) || (vy < 0) || (vx >= pmat->cols) || (vy >= pmat->rows))
            vvalue = 0;
        else
            vvalue = pmat->data[vx+vy*pmat->cols];

        if(uvalue - vvalue < pnode->metric.delta){
            traverseNode(x, y, pnode->leftNode, pmat, result);
        }else{
            traverseNode(x, y, pnode->rightNode, pmat, result);
        }
    }
}

void traverseNode(FILE* fp, Node* pnode)
{
    int leaf;
    fscanf(fp, "%d %d ", &leaf, &pnode->depth);

    if(leaf){
        pnode->leaf = leaf;
        fscanf(fp, "%f %f\n",  &pnode->result[0], &pnode->result[1]);
    }else{
        pnode->leaf = leaf;
        Node* nextnode = new Node[2];
        fscanf(fp, "%d %d %d %d %d\n",  &pnode->metric.u[0],  &pnode->metric.u[1],
            &pnode->metric.v[0],  &pnode->metric.v[1],  &pnode->metric.delta);
        pnode->leftNode = &nextnode[0];
        pnode->rightNode = &nextnode[1];
        traverseNode(fp, &nextnode[0]);
        traverseNode(fp, &nextnode[1]);
    }
}

int main(int argc, char *argv[])
{
	int i, j;

	//-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){
        printf("--(!)Error loading\n"); return -1;
    };
    CvEyeInit();

#if USE_USB_CAMERA
	CvCapture *capture = cvCreateCameraCapture( 0 );
    if( capture == NULL )
    {
        return -1;
    }
#endif

    if(0){
        MyEyeTrain();
    }

    char filename[256];
    int image_idx = 0;
    sprintf(filename, "./database/BioID_%.4d.pgm", image_idx);

    std::vector<Rect> faces;
    Mat equalized_image;
	
	for(j=0; j<NUM_TREE; j++){
		for(i=0; i<NUM_LANDMARKS; i++){
			rootNode[i][j] = new Node;
			char filename[100];
			sprintf(filename, "./regressiontree/%.3d/rt%.3d.txt", i, j);
			FILE* fp = fopen(filename, "r");
			traverseNode(fp, rootNode[i][j]);
			fclose(fp);
		}
	}

    while(1)
    {
#if USE_USB_CAMERA
		Mat image;
		Size size(384, 286);
        Mat camera_rawimage = cvQueryFrame( capture );
        Mat camera_grayimage;
		cvtColor(camera_rawimage, camera_grayimage, CV_RGB2GRAY);
		resize(camera_grayimage, image, size, cv::INTER_CUBIC);
#else
        Mat image = imread(filename, 0);
#endif

        equalizeHist( image, equalized_image );

        //-- Detect faces
        face_cascade.detectMultiScale( equalized_image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        if(1){
            if(faces.size() == 1){
                Point lefteye( faces[0].x + SK_FACEINIT::relative_lefteye_position[0]*faces[0].width, faces[0].y + SK_FACEINIT::relative_lefteye_position[1]*faces[0].height );
                Point righteye( faces[0].x + SK_FACEINIT::relative_righteye_position[0]*faces[0].width, faces[0].y + SK_FACEINIT::relative_righteye_position[1]*faces[0].height );

                if(1){
					// -----------------
                    // compute relative coordinate
                    int local_coordinate[2];
                    local_coordinate[0] = (lefteye.x + righteye.x) / 2.0f - PATCH_SIZEX / 2;
                    local_coordinate[1] = (lefteye.y + righteye.y) / 2.0f - PATCH_SIZEY / 2;
                    
					// -----------------
					// derive next offset
					int tmp[2];
					int modified_left_eyeoffset[2] = {0, 0};
					int modified_right_eyeoffset[2] = {0, 0};

					for(j=0; j<NUM_TREE; j++){
						traverseNode(local_coordinate[0], local_coordinate[1], rootNode[0][j], &image, tmp);
						modified_left_eyeoffset[0] += tmp[0];
						modified_left_eyeoffset[1] += tmp[1];
					}
					modified_left_eyeoffset[0] /= NUM_TREE;
					modified_left_eyeoffset[1] /= NUM_TREE;

					for(j=0; j<NUM_TREE; j++){
						traverseNode(local_coordinate[0], local_coordinate[1], rootNode[1][j], &image, tmp);
						modified_right_eyeoffset[0] += tmp[0];
						modified_right_eyeoffset[1] += tmp[1];
					}
					modified_right_eyeoffset[0] /= NUM_TREE;
					modified_right_eyeoffset[1] /= NUM_TREE;

					// render modified eye location
                    Point modified_lefteye( lefteye.x + modified_left_eyeoffset[0], lefteye.y + modified_left_eyeoffset[1] );
                    circle( image, modified_lefteye, 2, Scalar( 255, 0, 0 ), 2, 8, 0 );

					Point modified_righteye( righteye.x + modified_left_eyeoffset[0], righteye.y + modified_right_eyeoffset[1] );                    
                    circle( image, modified_righteye, 2, Scalar( 255, 0, 0 ), 2, 8, 0 );
                }
                
                // initial estimation
//                circle( image, lefteye, 3, Scalar( 128, 0, 0 ), 2, 8, 0 );
//                circle( image, righteye, 3, Scalar( 128, 0, 0 ), 2, 8, 0 );
                
                Point center( faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5 );
                ellipse( image, center, Size( faces[0].width*0.5, faces[0].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
            }
        }else{
            // use opencv eye
            CvEyeRun(image, equalized_image, faces);
        }

        //-- Show what you got
        imshow( "SourceImage", image );

        int c = waitKey(10);
        switch(c){
        case 'n':
            image_idx+=RATIO_TRAIN2TEST;
            sprintf(filename, "./database/BioID_%.4d.pgm", image_idx);
            break;
        default:
            break;
        }
    }
}