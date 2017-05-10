// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

This example program shows how to find frontal human faces in an image and
estimate their pose.  The pose takes the form of 68 landmarks.  These are
points on the face such as the corners of the mouth, along the eyebrows, on
the eyes, and so forth.


This example is essentially just a version of the face_landmark_detection_ex.cpp
example modified to use OpenCV's VideoCapture object to read from a camera instead
of files.


Finally, note that the face detector is fastest when compiled with at least
SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
chip then you should enable at least SSE2 instructions.  If you are using
cmake to compile this program you can enable them by using one of the
following commands when you create the build project:
cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
This will set the appropriate compiler options for GCC, clang, Visual
Studio, or the Intel compiler.  If you are using another compiler then you
need to consult your compiler's manual to determine how to enable these
instructions.  Note that AVX is the fastest but requires a CPU from at least
2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv\ml.h>
#include <opencv2\ml\ml.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <opencv\cv.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include"svm.h"

using namespace dlib;
using namespace std;

//cv::Point cPoint[68];
cv::vector<cv::Point> cPoints;
double featureScaler;
struct svm_node *node;
struct svm_model *svmModel;
char*mdFile = "face2.model";




double innerBrowRaiser;     //AU 1
double outerBrowRaiser;     //AU 2
double browLower;           //AU 4
double upperLidRaiser;      //AU 5
double lidTightener;        //AU 7
double noseWrinkler;        //AU 9
double lipCornerPull;       //AU 12
double lipCornerDepress;    //AU 15
double lowerLipDepress;     //AU 16
double lipStretch;          //AU 20
double lipTightener;        //AU 23
double jawDrop;             //AU 26

double getDist(cv::Point p1, cv::Point p2)
{
	double r = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	return r;
}

double getDistX(cv::Point p1, cv::Point p2)
{
	double r = abs(p1.x - p2.x);
	return r;
}

double getDistY(cv::Point p1, cv::Point p2)
{
	double r = abs(p1.y - p2.y);
	return r;
}


//绘制脸部的一些特征轮廓
void my_drawEdge(cv::Mat img, std::vector<cv::Point> points)
{

	//assign feature scaler as the width of the face, which does not change in response to different expression
	//featureScaler = (getDistX(cPoints[0], cPoints[16]) + getDistX(cPoints[1], cPoints[15]) + getDistX(cPoints[2], cPoints[14])) / 3;
	//cv::line(img, points[0], points[16], cv::Scalar(255, 0, 0), 3);
	//cv::line(img, points[1], points[15], cv::Scalar(255, 0, 0), 3);
	//cv::line(img, points[2], points[14], cv::Scalar(255, 0, 0), 3);

	int line_width = 1;
	cv::line(img, points[21], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[22], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[21], points[22], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[17], points[18], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[18], points[19], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[19], points[20], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[20], points[21], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[22], points[23], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[23], points[24], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[24], points[25], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[25], points[26], cv::Scalar(255, 0, 0), line_width);


	
	cv::line(img, points[17], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[26], points[27], cv::Scalar(255, 0, 0), line_width);

	cv::line(img, points[17], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[18], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[19], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[21], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[22], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[23], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[24], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[25], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[26], points[27], cv::Scalar(255, 0, 0), line_width);

	cv::line(img, points[37], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[44], points[27], cv::Scalar(255, 0, 0), line_width);

	cv::line(img, points[37], points[41], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[38], points[40], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[43], points[47], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[44], points[46], cv::Scalar(255, 0, 0), line_width);

	cv::line(img, points[29], points[27], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[30], points[27], cv::Scalar(255, 0, 0), line_width);

	cv::line(img, points[48], points[33], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[54], points[33], cv::Scalar(255, 0, 0), line_width);

	cv::line(img, points[57], points[33], cv::Scalar(255, 0, 0), line_width);
	
	cv::line(img, points[48], points[54], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[48], points[17], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[54], points[26], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[30], points[17], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[30], points[26], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[48], points[59], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[59], points[58], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[58], points[57], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[57], points[56], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[56], points[55], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[55], points[54], cv::Scalar(255, 0, 0), line_width);


	cv::line(img, points[49], points[59], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[50], points[58], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[51], points[57], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[52], points[56], cv::Scalar(255, 0, 0), line_width);
	cv::line(img, points[53], points[55], cv::Scalar(255, 0, 0), line_width);


}


frontal_face_detector detector = get_frontal_face_detector();
shape_predictor pose_model;


/*int main___test____________test__________test()
{
	
	CvSVM svm;
	CvSVMParams param;
	param.svm_type = 100;
	param.kernel_type = 1;
	param.degree = 4;
	param.gamma = 4;
	param.coef0 = 1;
	CvMat *dataMat = cvCreateMat(10, 12, CV_32FC1);
	CvMat *labelMat = cvCreateMat(10, 1, CV_32SC1);
	for (int i = 0; i<10; i++)
	{
		for (int j = 0; j<12; j++)
		{
			cvSetReal2D(dataMat, i, j, data_org[i][j + 1]);
		}
		cvSetReal2D(labelMat, i, 0, data_org[i][0]);

	}
	svm.train(dataMat, labelMat, NULL, NULL, param);
	svm.save("svmResult.txt");
	CvMat *testMat = cvCreateMat(1, 12, CV_32FC1);

	for (int i = 0; i<12; i++)
	{
		cvSetReal2D(testMat, 0, i, test_data[i]);
	}

	float flag = 0;
	flag = svm.predict(testMat);
	cout << "testMat, flag = " << flag << endl;
	system("pause");
	cvReleaseMat(&dataMat);
	cvReleaseMat(&labelMat);
	cvReleaseMat(&testMat);
	return 0;
}*/


int main(){

	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	//double duration;
	//duration = static_cast<double>(cv::getTickCount());
	char* faceImgPath = "happy10.jpg";
	cv::Mat img = cv::imread(faceImgPath);
	//cv::resize(img, img, cv::Size(512, 512));
	//cv::Mat img;
	cv::Mat friFrame;
	img.copyTo(friFrame);
	cv_image<bgr_pixel> cimg(friFrame);
	std::vector<rectangle> faces = detector(cimg);
	if (faces.size() == 0)
	{
		std::cout << "can't detect the face" << endl;
		exit(0);
	}
	std::vector<full_object_detection> shapes;
	for (int i = 0; i < faces.size(); i++)
	{
		full_object_detection shape = pose_model(cimg, faces[i]);
		shapes.push_back(shape);
	}
	for (int i = 0; i < shapes.size(); i++)
	{
		full_object_detection shape = shapes[i];
		//std::cout << shape.part(1).size();
		for (int i = 0; i < 68; i++)
		{
			//std::cout << shape.part(i).size();
			cv::Point2f temp;
			temp.x = shape.part(i)(0);
			temp.y = shape.part(i)(1);
			cPoints.push_back(temp);
			//cv::line(img, cv::Point(shape.part(67)(0), shape.part(67)(1)), temp, cv::Scalar(0, 0, 255));
		}
		cv::imshow("org_face", img);
		my_drawEdge(img, cPoints);
		cv::imshow("face_edge", img);
		cv::waitKey(200);
		//assign feature scaler as the width of the face, which does not change in response to different expression
		featureScaler = (getDistX(cPoints[0], cPoints[16]) + getDistX(cPoints[1], cPoints[15]) + getDistX(cPoints[2], cPoints[14])) / 3;
		//assign action unit 1
		innerBrowRaiser = ((getDistY(cPoints[21], cPoints[27]) + getDistY(cPoints[22], cPoints[27])) / 2) / featureScaler;
		//assign action unit 2
		outerBrowRaiser = ((getDistY(cPoints[17], cPoints[27]) + getDistY(cPoints[26], cPoints[27])) / 2) / featureScaler;
		//assign action unit 4
		browLower = (((getDistY(cPoints[17], cPoints[27]) + getDistY(cPoints[18], cPoints[27]) +
			getDistY(cPoints[19], cPoints[27]) + getDistY(cPoints[20], cPoints[27]) +
			getDistY(cPoints[21], cPoints[27])) / 5 +
			(getDistY(cPoints[22], cPoints[27]) + getDistY(cPoints[23], cPoints[27]) +
			getDistY(cPoints[24], cPoints[27]) + getDistY(cPoints[25], cPoints[27]) +
			getDistY(cPoints[26], cPoints[27])) / 5) / 2) / featureScaler;
		//assign action unit 5
		upperLidRaiser = ((getDistY(cPoints[37], cPoints[27]) + getDistY(cPoints[44], cPoints[27])) / 2) / featureScaler;
		//assign action unit 7
		lidTightener = ((getDistY(cPoints[37], cPoints[41]) + getDistY(cPoints[38], cPoints[40])) / 2 +
			(getDistY(cPoints[43], cPoints[47]) + getDistY(cPoints[44], cPoints[46])) / 2) / featureScaler;
		//assign action unit 9
		noseWrinkler = (getDistY(cPoints[29], cPoints[27]) + getDistY(cPoints[30], cPoints[27])) / featureScaler;
		//assign action unit 12
		lipCornerPull = ((getDistY(cPoints[48], cPoints[33]) + getDistY(cPoints[54], cPoints[33])) / 2) / featureScaler;
		//assign action unit 16
		lowerLipDepress = getDistY(cPoints[57], cPoints[33]) / featureScaler;
		//assign action unit 20
		lipStretch = getDistX(cPoints[48], cPoints[54]) / featureScaler;
		//assign action unit 23
		lipTightener = (getDistY(cPoints[49], cPoints[59]) +
			getDistY(cPoints[50], cPoints[58]) +
			getDistY(cPoints[51], cPoints[57]) +
			getDistY(cPoints[52], cPoints[56]) +
			getDistY(cPoints[53], cPoints[55])) / featureScaler;
		//assign action unit 26
		jawDrop = getDistY(cPoints[8], cPoints[27]) / featureScaler;

		if ((svmModel = svm_load_model(mdFile)) == 0)
		{
			std::cout << "can not open model file!!" << std::endl;
			exit(0);
		}

		//allocate memory from svm node
		node = (struct svm_node *)malloc(64 * sizeof(struct svm_node));

		double class_nr = 0;
		int class_nr_int = 0;
		int i = 0;
		for (i = 0; i < 11; i++)
		{
			node[i].index = i;
		}
		node[11].index = -1;

		//assign value of nodes
		node[0].value = innerBrowRaiser;
		node[1].value = outerBrowRaiser;
		node[2].value = browLower;
		node[3].value = upperLidRaiser;
		node[4].value = lidTightener;
		node[5].value = noseWrinkler;
		node[6].value = lipCornerPull;
		node[7].value = lowerLipDepress;
		node[8].value = lipStretch;
		node[9].value = lipTightener;
		node[10].value = jawDrop;
		std::cout << innerBrowRaiser << endl;
		cout << outerBrowRaiser << endl;
		cout << browLower << endl;
		cout << upperLidRaiser << endl;
		cout << lidTightener << endl;
		cout << noseWrinkler << endl;
		cout << lipCornerDepress << endl;
		cout << lowerLipDepress << endl;
		cout << lipStretch << endl;
		cout << lipTightener << endl;
		cout << jawDrop << endl;
		//predict the class
		//0: neutral face
		//1: happy
		//2: angry
		//3: disgust
		//-1 sad
		//-2 suprise
		//-3 fear
		class_nr = svm_predict(svmModel, node);
		class_nr_int = (int)class_nr;

		if (class_nr_int == 2){
			std::cout << "angry" << std::endl;
		}
		else if (class_nr_int == 1)
		{
			std::cout << "happy" << std::endl;
		}
		else if (class_nr_int == 3)
		{
			std::cout << "disgust" << std::endl;
		}
		else if (class_nr_int == -1){
			std::cout << "sad" << std::endl;
		}
		else if (class_nr_int == -2)
		{
			std::cout << "surprise" << std::endl;
		}
		else if (class_nr_int == -3)
		{
			std::cout << "fear" << std::endl;
		}
		else
		{
			std::cout << "nature" << std::endl;
		}

	}
	return 0;
}


