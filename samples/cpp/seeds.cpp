#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates SEEDS superpixels using OpenCV class SuperpixelSEEDS\n"
            "It captures either from the camera of your choice: 0, 1, ... default 0\n"
            "Or from an input image\n"
            "Call:\n"
            "./seeds [camera #, default 0]\n"
            "./seeds [input image file]\n" << endl;
}

static const char* window_name = "SEEDS Superpixels";

int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat input_image;
    bool use_video_capture = false;
    help();

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])) )
    {
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
        use_video_capture = true;
    }
    else if( argc >= 2 )
    {
        input_image = imread(argv[1]);
    }

    if( use_video_capture )
    {
        if( !cap.isOpened() )
        {
            cout << "Could not initialize capturing...\n";
            return -1;
        }
    }
    else if( input_image.empty() )
    {
        cout << "Could not open image...\n";
        return -1;
    }

    namedWindow(window_name, 0);
    int num_iterations = 4;
    createTrackbar("Iterations", window_name, &num_iterations, 12, 0);

    Mat result;
    bool init = false;
    Ptr<SuperpixelSEEDS> seeds;
    int width, height;

    for (;;)
    {
        Mat frame;
        if( use_video_capture )
            cap >> frame;
        else
            input_image.copyTo(frame);

        if( frame.empty() )
            break;

        if( !init )
        {
            width = frame.size().width;
            height = frame.size().height;
            seeds = createSuperpixelSEEDS(width, height, frame.channels());
            seeds->initialize(3, 4, 4);
            init = true;
        }
        Mat converted;
        cvtColor(frame, converted, COLOR_BGR2HSV);
        /*
         //16 bit depth:
         Mat img16;
         converted.convertTo(img16, CV_16U);
         img16 *= 256;
         //*/
        /*
         //float
         Mat imgfloat;
         converted.convertTo(imgfloat, CV_32F);
         imgfloat /= 256;
         //*/

        double t = (double) getTickCount();

        seeds->iterate(converted, num_iterations);
        result = frame;

        t = ((double) getTickCount() - t) / getTickFrequency();
        printf("SEEDS segmentation took %i ms\n", (int) (t * 1000));

        seeds->drawContoursAroundLabels(result, 0xff0000, false);

        imshow(window_name, result);

        int c = waitKey(1);
        if( c == 'q' || c == 'Q' || (c & 255) == 27 )
            break;
    }

    return 0;
}
