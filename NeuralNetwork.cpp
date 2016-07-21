#include <opencv2/core/core.hpp>  
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

	if (number == 0){ data.at<float>(number_of_items, 0) = 1.0; }
	if (number == 1){ data.at<float>(number_of_items, 1) = 1.0; }
	if (number == 2){ data.at<float>(number_of_items, 2) = 1.0; }
	if (number == 3){ data.at<float>(number_of_items, 3) = 1.0; }
	if (number == 4){ data.at<float>(number_of_items, 4) = 1.0; }
	if (number == 5){ data.at<float>(number_of_items, 5) = 1.0; }
	if (number == 6){ data.at<float>(number_of_items, 6) = 1.0; }
	if (number == 7){ data.at<float>(number_of_items, 7) = 1.0; }
	if (number == 8){ data.at<float>(number_of_items, 8) = 1.0; }
	if (number == 9){ data.at<float>(number_of_items, 9) = 1.0; }

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

void NeuralNetwork(Mat layerSizes){
	CvANN_MLP bp;
	CvANN_MLP*nnetwork;
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.1;
	params.bp_moment_scale = 0.1;

	Mat mnist_image_data = read_mnist_image("MNIST/train-images.idx3-ubyte");
	cout << mnist_image_data.rows << " :" << mnist_image_data.cols << endl;
	Mat mnist_label_data = read_Mnist_Label("MNIST/train-labels.idx1-ubyte");
	cout << mnist_label_data.rows << " :" << mnist_label_data.cols << endl;

	//writeMatToFile(mnist_image_data, "mnist_image_data.txt");
	//writeMatToFile(mnist_label_data, "mnist_label_data.txt");

	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM);
	bp.train(mnist_image_data, mnist_label_data, Mat(), Mat(), params);
	bp.save("bp.xml");
}
int main(){
/*	time_t nStart = time(NULL);
	Mat layerSizes = (Mat_<int>(1, 3) << 784, 300, 10);
	NeuralNetwork(layerSizes);
	time_t nEnd = time(NULL);
	cout << nEnd - nStart << "總訓練秒數" << endl;*/

/*	Mat mnist_image_data = read_mnist_image("MNIST/train-images.idx3-ubyte");
	cout << mnist_image_data.rows << " :" << mnist_image_data.cols << endl;
	
	Mat mnist_label_data=read_Mnist_Label("MNIST/train-labels.idx1-ubyte");
	cout << mnist_label_data.rows << " :" << mnist_label_data.cols << endl;*/

//	writeMatToFile(mnist_image_data, "mnist_image_data.txt");
//	writeMatToFile(mnist_label_data,"mnist_label_data.txt");

/*	Mat layerSizes = (Mat_<int>(1,3) << 784,100,10);
	cout << layerSizes;
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1);
	
	cout << "train..." << endl;
	bp.train(mnist_image_data, mnist_label_data, Mat(), Mat(), params);
	cout << "end" << endl;
	*/
	/*Mat mnist_image_data = read_mnist_image("MNIST/t10k-images.idx3-ubyte");
	cout << mnist_image_data.rows << " :" << mnist_image_data.cols << endl;

	Mat mnist_label_data = read_Mnist_Label("MNIST/t10k-labels.idx1-ubyte");
	cout << mnist_label_data.rows << " :" << mnist_label_data.cols << endl;*/
	CvANN_MLP bp;
	bp.load("bp.xml");

	float sample[1][784] = { 
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.329412, 0.72549, 0.623529, 0.592157, 0.235294, 0.141176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.870588, 0.996078, 0.996078, 0.996078, 0.996078, 0.945098, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.666667, 0.203922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.262745, 0.447059, 0.282353, 0.447059, 0.639216, 0.890196, 0.996078, 0.882353, 0.996078, 0.996078, 0.996078, 0.980392, 0.898039, 0.996078, 0.996078, 0.54902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0666667, 0.258824, 0.054902, 0.262745, 0.262745, 0.262745, 0.231373, 0.0823529, 0.92549, 0.996078, 0.415686, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.32549, 0.992157, 0.819608, 0.07 } };
	Mat sampleMat(1, 784, CV_32FC1, sample);
	Mat responseMat;

	bp.predict(sampleMat, responseMat);
	float* p = responseMat.ptr<float>(0);
	cout << responseMat << endl;
	cout<< p <<endl;

	/*int width = 512, height = 512;
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
				image.at<Vec3b>(j, i) = p[0] * green;
			}
			else
			{
				image.at<Vec3b>(j, i) = p[1] * blue;
			}
		}
	} 
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(100, 100), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(20, 20), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(30, 30), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(40, 40), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(50, 50), 5, Scalar(0,0,0), thickness, lineType);
	circle(image, Point(200, 50), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(20, 200), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(300, 300), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(400, 400), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(500, 500), 5, Scalar(255, 255, 255), thickness, lineType);	
	imshow("BP", image); */
	
	
	
	waitKey(0);
	while (true){if (waitKey(10) == 27)break;
	}
	return 0;
}