#include<stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mxnet/c_predict_api.h>
#include <math.h>

#include "mxnet_mtcnn.hpp"

void test_mtcnn()
{
	std::string model_dir = "model";
	MxNetMtcnn mtcnn;
	mtcnn.LoadModule(model_dir);

	cv::Mat img = cv::imread("image/stephanie-sun.jpg");
	std::vector<face_box> face_info;
	mtcnn.Detect(img, face_info);
	for (int i = 0; i<face_info.size();i++)
	{
		auto face = face_info[i];
		std::cout << "face location: x0=" << face.x0 << " y0=" << face.y0 << "x1=" << face.x1 << "y1=" << face.y1 << std::endl;
		std::cout << "face landmark: x0=" << face.landmark.x[0] << " x1=" << face.landmark.x[1] << "x2=" << face.landmark.x[2] << "x3=" << face.landmark.x[3] <<"x4="<<face.landmark.x[4]<< std::endl;
		std::cout << "face landmark: y0=" << face.landmark.y[0] << " y1=" << face.landmark.y[1] << "y2=" << face.landmark.y[2] << "y3=" << face.landmark.y[3] << "y4=" << face.landmark.y[4] << std::endl;

		for (int j = 0; j < 5; j++)
		{
			cv::Point p(face.landmark.x[j], face.landmark.y[j]);
			cv::circle(img, p, 2, cv::Scalar(0, 0, 255), -1);
		}

		cv::Point pt1(face.x0, face.y0);
		cv::Point pt2(face.x1, face.y1);
		cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0),2);
		cv::imshow("img", img);
		cv::waitKey(0);
	}

}

int main(int argc, char* argv[]) {

	test_mtcnn();
	return 0;
}