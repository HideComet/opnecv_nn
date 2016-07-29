#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
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
	float val = 1;
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

Mat read_Mnist_Label(string filename, int return_choose)
{
	int magic_number;
	int number_of_items;
	Mat LabelMat;
	Mat temp_Label;
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
			temp_Label.push_back(int(temp));
		}
	}
	file.close();
	if (return_choose == 1){
		return LabelMat;
	}
	else{
		return temp_Label;
	}

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
	TermCrlt.epsilon = 0.000001f; 
	TermCrlt.max_iter = 50; 
	params.term_crit = TermCrlt;

	Mat mnist_image_data = read_mnist_image("MNIST/train-images.idx3-ubyte");
	//writeMatToFile(mnist_image_data, "mnist_image_data");
	Mat mnist_label_data = read_Mnist_Label("MNIST/train-labels.idx1-ubyte", 1);
	//writeMatToFile(mnist_label_data, "mnist_label_data");
	Mat layerSizes = (Mat_<int>(1, 3) << mnist_image_data.cols,hidelayer, mnist_label_data.cols);
	cout << "train layerSizes=" << layerSizes << endl;
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1);

	cout << "train..." ;
	time_t nStart = time(NULL);
	bp.train(mnist_image_data, mnist_label_data, Mat(), Mat(), params);
	time_t nEnd = time(NULL);
	cout << "--"<< nEnd - nStart << "秒" << endl;
	
	char xml_name[50];
	sprintf(xml_name, "xml/NeuralNetwork_%d_hidelayer.xml", hidelayer);
	bp.save(xml_name);
}

void NeuralNetwork_test(int hidelayer){
	CvANN_MLP bp;

	Mat mnist_image_data = read_mnist_image("MNIST/t10k-images.idx3-ubyte");
	Mat mnist_label_data = read_Mnist_Label("MNIST/t10k-labels.idx1-ubyte", 2);

	char xml_name[50], responseMat_name[50];
	sprintf(xml_name, "xml/NeuralNetwork_%d_hidelayer.xml", hidelayer);
	bp.load(xml_name);

	Mat layerSizes = bp.get_layer_sizes();
	cout << "test layerSizes=" << layerSizes<<endl;
	
	Mat responseMat;
	cout << "test..." ;
	time_t nStart = time(NULL);
	bp.predict(mnist_image_data, responseMat);
	time_t nEnd = time(NULL);
	cout <<"--"<< nEnd - nStart << "秒" << endl;

	//sprintf(responseMat_name, "responseMat/responseMat_%d_hidelayer.txt", hidelayer);
	//writeMatToFile(responseMat,responseMat_name);
	
	Point maxPoint, maxLoc;
	Mat DataMat;
	for (int i = 0; i<responseMat.rows; i++){
		Mat temp_max = responseMat.rowRange(i, i + 1);
		Point maxLoc;
		minMaxLoc(temp_max, NULL, NULL, NULL, &maxLoc);
		DataMat.push_back(maxLoc.x);
	}
	Mat sum = DataMat - mnist_label_data;
	double error = countNonZero(sum), total = responseMat.rows;
	double Correct_rate = ((total - error) / total) * 100;

	cout << "測試總數量 :" << total << " 測試錯誤數量 :" << error << endl;
	cout <<"正確率 :" <<Correct_rate <<"%"<< endl;

}
void test(int hidelayer){
	CvANN_MLP bp; 
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.1;
	params.bp_moment_scale = 0.1;
	
	float labels[3][5] = { { 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 } };
	Mat labelsMat(3, 5, CV_32FC1, labels);

	float trainingData[3][5] = { { 1, 2, 3, 4, 5 }, { 111, 112, 113, 114, 115 }, { 21, 22, 23, 24, 25 } };
	Mat trainingDataMat(3, 5, CV_32FC1, trainingData);
	Mat layerSizes = (Mat_<int>(1, 3) << 5, hidelayer, 5);
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM); 

	bp.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	bp.save("xml/test.xml");
}
int main(int argc, char**argv){
	switch (*argv[1]){
	case'1':
		NeuralNetwork_Train((int)atoi(argv[2]));
		break;
	case'2':
		NeuralNetwork_test(atoi(argv[2]));
		break;
	case'3':
		NeuralNetwork_Train(atoi(argv[2]));
		NeuralNetwork_test(atoi(argv[2]));
		break;
	case'4':
		cout << "debug" << endl;
		test(atoi(argv[2]));
		break;
	}

	return 0;
}