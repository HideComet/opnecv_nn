#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream>  
#include <string> 
#include <fstream>
#include <time.h>

using namespace cv;
using namespace std;
#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
const char *pstrWindowsMouseDrawTitle = "鼠標繪圖(http://blog.csdn.net/MoreWindows)";
// 鼠標消息的回調函數
void on_mouse(int event, int x, int y, int flags, void* param)
{
	static bool s_bMouseLButtonDown = false;
	static CvPoint s_cvPrePoint = cvPoint(0, 0);

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		s_bMouseLButtonDown = true;
		s_cvPrePoint = cvPoint(x, y);
		break;

	case  CV_EVENT_LBUTTONUP:
		s_bMouseLButtonDown = false;
		break;

	case CV_EVENT_MOUSEMOVE:
		if (s_bMouseLButtonDown)
		{
			CvPoint cvCurrPoint = cvPoint(x, y);
			cvLine((IplImage*)param, s_cvPrePoint, cvCurrPoint, CV_RGB(0, 0, 20), 3);
			s_cvPrePoint = cvCurrPoint;
			cvShowImage(pstrWindowsMouseDrawTitle, (IplImage*)param);
		}
		break;
	}
}
int main()
{
	const int MAX_WIDTH = 500, MAX_HEIGHT = 400;
	const char *pstrSaveImageName = "MouseDraw.jpg";

	IplImage *pSrcImage = cvCreateImage(cvSize(MAX_WIDTH, MAX_HEIGHT), IPL_DEPTH_8U, 3);
	cvSet(pSrcImage, CV_RGB(255, 255, 255)); //可以用cvSet()將圖像填充成白色
	cvNamedWindow(pstrWindowsMouseDrawTitle, CV_WINDOW_AUTOSIZE);
	cvShowImage(pstrWindowsMouseDrawTitle, pSrcImage);

	cvSetMouseCallback(pstrWindowsMouseDrawTitle, on_mouse, (void*)pSrcImage);

	int c;
	do{
		c = cvWaitKey(0);
		switch ((char)c)
		{
		case 'r':
			cvSet(pSrcImage, CV_RGB(255, 255, 255));
			cvShowImage(pstrWindowsMouseDrawTitle, pSrcImage);
			break;

		case 's':
			cvSaveImage(pstrSaveImageName, pSrcImage);
			break;
		}
	} while (c > 0 && c != 27);

	cvDestroyWindow(pstrWindowsMouseDrawTitle);
	cvReleaseImage(&pSrcImage);
	return 0;
}