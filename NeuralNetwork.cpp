#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream>  
#include <string> 
#include <fstream>
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

Mat read_mnist_label(const string fileName) {
	int magic_number;
	int number_of_items;

	Mat LabelMat;

	ifstream file(fileName, ios::binary);
	if (file.is_open())
	{
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_items, sizeof(number_of_items));
		magic_number = reverseInt(magic_number);
		number_of_items = reverseInt(number_of_items);

		cout << "MAGIC NUMBER = " << magic_number << "  ; NUMBER OF ITEMS = " << number_of_items << endl;
		unsigned int s = 0, e = 0;

		LabelMat = Mat::zeros(number_of_items, 1, CV_32SC1);
		for (int i = 0; i < number_of_items; i++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (i == 0) s = (unsigned int)temp;
			else if (i == number_of_items - 1) e = (unsigned int)temp;
		}
		cout << "first label = " << s << endl;
		cout << "last label = " << e << endl;
	}
	file.close();
	return LabelMat;
}
void writeMatToFile(cv::Mat& m, const char* filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			fout << m.at<float>(i, j) << "\t";
		}
		fout << endl;
	}

	fout.close();
}

int main(){
	CvANN_MLP bp;
	CvANN_MLP*nnetwork;
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.1;
	params.bp_moment_scale = 0.1;

	//float labels[10][2] = {{0.9,0.1}, {0.9,0.1}, {0.9,0.1}, {0.9,0.1}, {0.9,0.1}, {0.1,0.9}, {0.1,0.9}, {0.1,0.9}, {0.1,0.9}, {0.1,0.9}};
	//Mat labelsMat(10, 2, CV_32FC1, labels);
	
	Mat mnist_image_data = read_mnist_image("MNIST/t10k-images.idx3-ubyte");
	cout << mnist_image_data.rows << " :" << mnist_image_data.cols << endl;
	
	writeMatToFile(mnist_image_data, "mnist_image_data.txt");

	Mat mnist_label_data = read_mnist_label("MNIST/t10k-labels.idx1-ubyte");
	

	//float trainingData[10][2] = { { 10, 10 }, { 20, 20 }, { 30, 30 }, { 40, 40 }, { 50, 50 }, { 100, 100 }, { 200, 200 }, { 300, 300 }, { 400, 400 }, {500,500}};
	//Mat trainingDataMat(10, 2, CV_32FC1, trainingData);

	Mat layerSizes = (Mat_<int>(1,3) << 784,100,10);
	bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);
	

	//bp.train(mnist_image_data, mnist_label_data, Mat(), Mat(), params);

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
	*/
	bp.save("bp.xml");

	waitKey(0);
	return 0;
	
}