﻿#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream>  
#include <string> 
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;

void writeMatToFile(cv::Mat& m, const char* filename)
{
	ofstream fout(filename);
	for (int i = 0; i<m.rows; i++){
		for (int j = 0; j<m.cols; j++){
			fout << m.at<float>(i, j) << "\t";
		}
		fout << endl;
	}
	fout.close();
}
void determine(int number, Mat data, int number_of_items){
	float val = 0.9;
	if (number == 0){ data.at<float>(number_of_items, 0) = val; }
	if (number == 1){ data.at<float>(number_of_items, 1) = val; }
	if (number == 2){ data.at<float>(number_of_items, 2) = val; }
	if (number == 3){ data.at<float>(number_of_items, 3) = val; }
	if (number == 4){ data.at<float>(number_of_items, 4) = val; }
	if (number == 5){ data.at<float>(number_of_items, 5) = val; }
	if (number == 6){ data.at<float>(number_of_items, 6) = val; }
	if (number == 7){ data.at<float>(number_of_items, 7) = val; }
	if (number == 8){ data.at<float>(number_of_items, 8) = val; }
	if (number == 9){ data.at<float>(number_of_items, 9) = val; }

}

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

	ifstream file(fileName,ios::binary);
	if (file.is_open())
	{

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
	
		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);

		DataMat = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
		for (int i = 0; i < number_of_images; i++) {
			for (int j = 0; j < n_rows * n_cols; j++) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				float pixel_value = float((temp + 0.0) / 255.0);
				DataMat.at<float>(i, j) = pixel_value;
			}
		}
	}
	file.close();
	return DataMat;
}

Mat read_Mnist_Label(string filename)
{
	int magic_number;
	int number_of_items;

	Mat LabelMat;

	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_items, sizeof(number_of_items));
		magic_number = reverseInt(magic_number);
		number_of_items = reverseInt(number_of_items);
		
		unsigned int s = 0, e = 0;
		LabelMat = Mat::zeros(number_of_items, 10, CV_32FC1);
		for (int i = 0; i < number_of_items; i++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			determine(int(temp), LabelMat, i);
			//LabelMat.at<int>(i, 0) = temp;

		}
	}
	file.close();
	return LabelMat;
}

void NeuralNetwork_Train(int hidelayer){
	CvANN_MLP bp;
	CvANN_MLP*nnetwork;
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.0001;
	params.bp_moment_scale = 0;

	CvTermCriteria TermCrlt;
	TermCrlt.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	TermCrlt.epsilon = 0.0001f;
	TermCrlt.max_iter = 2;
	params.term_crit = TermCrlt;

	Mat layerSizes = (Mat_<int>(1, 3) << 784, hidelayer, 10);
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1);

	Mat mnist_image_data = read_mnist_image("MNIST/train-images.idx3-ubyte");
	cout << mnist_image_data.rows << " :" << mnist_image_data.cols << endl;
	Mat mnist_label_data = read_Mnist_Label("MNIST/train-labels.idx1-ubyte");
	cout << mnist_label_data.rows << " :" << mnist_label_data.cols << endl;

	cout << "train..." << endl;
	time_t nStart = time(NULL);
	bp.train(mnist_image_data, mnist_label_data, Mat(), Mat(), params);
	time_t nEnd = time(NULL);
	cout << "endl ; " << nEnd - nStart << "總訓練秒數" << endl;
	
	char xml_name[50];
	sprintf(xml_name, "NeuralNetwork_", hidelayer, "_hidelayer.xml");
	bp.save(xml_name);
}

void NeuralNetwork_test(int hidelayer){
	CvANN_MLP bp;

	Mat mnist_image_data = read_mnist_image("MNIST/t10k-images.idx3-ubyte");
	cout << mnist_image_data.rows << " :" << mnist_image_data.cols << endl;
	Mat mnist_label_data = read_Mnist_Label("MNIST/t10k-labels.idx1-ubyte");
	cout << mnist_label_data.rows << " :" << mnist_label_data.cols << endl;

	/*float sample[1][784] = {
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.329412, 0.72549, 0.623529, 0.592157, 0.235294, 0.141176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.870588, 0.996078, 0.996078, 0.996078, 0.996078, 0.945098, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.666667, 0.203922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.262745, 0.447059, 0.282353, 0.447059, 0.639216, 0.890196, 0.996078, 0.882353, 0.996078, 0.996078, 0.996078, 0.980392, 0.898039, 0.996078, 0.996078, 0.54902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0666667, 0.258824, 0.054902, 0.262745, 0.262745, 0.262745, 0.231373, 0.0823529, 0.92549, 0.996078, 0.415686, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.32549, 0.992157, 0.819608, 0.07 } };
	Mat sampleMat(1, 784, CV_32FC1, sample);*/
	char xml_name[50];
	sprintf(xml_name, "NeuralNetwork_%d_hidelayer.xml", hidelayer);
	cout << xml_name << endl;
	bp.load(xml_name);

	Mat responseMat;
	bp.predict(mnist_image_data, responseMat);
	writeMatToFile(responseMat, "responseMat/responseMat.txt");

}
int main(){	
	//NeuralNetwork_Train(1000);

	NeuralNetwork_test(1000);

	waitKey(0);
	while (true){if (waitKey(10) == 27)break;
	}
	return 0;
}