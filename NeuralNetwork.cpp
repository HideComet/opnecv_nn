#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream>  
#include <string> 
class OR_mnist;
using namespace std;
using namespace cv;

int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int main(){
	CvANN_MLP bp;
	CvANN_MLP*nnetwork;
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.1;
	params.bp_moment_scale = 0.1;

	float labels[10][2] = {{0.9,0.1}, {0.9,0.1}, {0.9,0.1}, {0.9,0.1}, {0.9,0.1}, {0.1,0.9}, {0.1,0.9}, {0.1,0.9}, {0.1,0.9}, {0.1,0.9}};
	Mat labelsMat(10, 2, CV_32FC1, labels);
	
	ifstream file(fileName, ios::binary);

	float trainingData[10][2] = { { 10, 10 }, { 20, 20 }, { 30, 30 }, { 40, 40 }, { 50, 50 }, { 100, 100 }, { 200, 200 }, { 300, 300 }, { 400, 400 }, {500,500}};
	Mat trainingDataMat(10, 2, CV_32FC1, trainingData);
	
	Mat layerSizes = (Mat_<int>(1, 5) << 2,2,2,2,2);
	bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);

	bp.train(trainingDataMat, labelsMat, Mat(), Mat(), params);	

	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	Vec3b green(0, 255, 0), blue(255, 0, 0);
	
	// Show the decision regions
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			Mat responseMat;
			bp.predict(sampleMat, responseMat);
			float* p = responseMat.ptr<float>(0);
			//cout << responseMat.ptr << endl;
			if (p[0] > p[1])
			{
				image.at<Vec3b>(j, i) = green;
			}
			else
			{
				image.at<Vec3b>(j, i) = blue;
			}
		}
	} 
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(10, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(20, 20), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(30, 30), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(40, 40), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(50, 50), 5, Scalar(0,0,0), thickness, lineType);
	circle(image, Point(100, 100), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(200, 200), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(300, 300), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(400, 400), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(500, 500), 5, Scalar(255, 255, 255), thickness, lineType);	
	imshow("BP Simple Example", image); 
	
	bp.save("bp.xml");

	waitKey(0);
	return 0;
	
}