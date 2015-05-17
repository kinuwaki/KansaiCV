#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"

String eyes_cascade_name  = "./haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier eyes_cascade;

void CvEyeInit()
{
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        printf("--(!)Error loading\n"); 
    };
}

void CvEyeRun(Mat& image, Mat& equalized_image, std::vector<Rect>& faces)
{
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = equalized_image( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( image, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
}