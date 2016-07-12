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

Mat read_mnist_image(const string fileName) {
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	Mat DataMat;

	ifstream in(fileName, ios::binary);
	if (file.is_open())
	{
		cout << "成功打开图像集 ... \n";

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		//cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << endl;

		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);
		cout << "MAGIC NUMBER = " << magic_number
			<< " ;NUMBER OF IMAGES = " << number_of_images
			<< " ; NUMBER OF ROWS = " << n_rows
			<< " ; NUMBER OF COLS = " << n_cols << endl;

		//-test-
		//number_of_images = testNum;
		//输出第一张和最后一张图，检测读取数据无误
		Mat s = Mat::zeros(n_rows, n_rows * n_cols, CV_32FC1);
		Mat e = Mat::zeros(n_rows, n_rows * n_cols, CV_32FC1);

		cout << "开始读取Image数据......\n";
		start_time = clock();
		DataMat = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
		for (int i = 0; i < number_of_images; i++) {
			for (int j = 0; j < n_rows * n_cols; j++) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				float pixel_value = float((temp + 0.0) / 255.0);
				DataMat.at<float>(i, j) = pixel_value;

				//打印第一张和最后一张图像数据
				if (i == 0) {
					s.at<float>(j / n_cols, j % n_cols) = pixel_value;
				}
				else if (i == number_of_images - 1) {
					e.at<float>(j / n_cols, j % n_cols) = pixel_value;
				}
			}
		}
		end_time = clock();
		cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
		cout << "读取Image数据完毕......" << cost_time << "s\n";

		imshow("first image", s);
		imshow("last image", e);
		waitKey(0);
	}
	file.close();
	return DataMat;
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
	
	ifstream file("test", ios::binary);

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