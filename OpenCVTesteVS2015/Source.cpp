#include <iostream> 

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <iostream>
#include <fstream>
#include <stdlib.h>

void colorReduce(cv::Mat& image, int div = 128)
{
	int nl = image.rows;
	int nc = image.cols * image.channels();

	for (int j = 0; j < nl; j++)
	{
		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i < nc; i++)
		{
			data[i] = data[i] / div * div + div / 2;
		}
	}

	/*
	CORES Remanescentes:

	amarelo 192 192 64
	cinza 64 64 64
	vermelho 192 64 64
	azul 64 64 192
	verde 64 192 64
	rosa 192 64 192
	branco 192 192 192
	ciano 64 192 192

	*/
}


int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
cv::Mat src, src_gray, dst;
char* window_name = "Threshold Demo";
char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

void Threshold_Demo(int, void*)
{
	/* 0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/

	cv::threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);

	imshow(window_name, dst);
}

int main()
{
	/*IplImage* pImg;

	if ((pImg = cvLoadImage("c://1.jpg", 1)) != 0)
	{
		cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
		cvShowImage("Image", pImg);
		cvWaitKey(0);
		cvDestroyWindow("Image");
		return 0;
	}*/
	
		string imagem("C://1.jpg");
		//IplImage* pImg;
		cv::Mat x = cv::imread(imagem.c_str(),CV_LOAD_IMAGE_COLOR);
		cv::Mat aux;

		cv::imshow("Imagem Original", x);
		cv::waitKey(0);

	//	//clareando a imagem
	//	for (int i = 0; i < x.rows; i++)
	//	{
	//		for (int j = 0; j < x.cols; j++)
	//		{
	//cv::Vec3b bgrPixel = x.at<cv::Vec3b>(i, j);
	//
	//			try
	//			{
	//if (bgrPixel.val[0] < 200)
	//	bgrPixel.val[0] = bgrPixel.val[0] + 50; // B
	//if (bgrPixel.val[1] < 200)
	//	bgrPixel.val[1] = bgrPixel.val[1] + 50; // G
	//if (bgrPixel.val[2] < 200)
	//	bgrPixel.val[2] = bgrPixel.val[2] + 50; // R
	//
	//	x.at<cv::Vec3b>(i, j) = bgrPixel;
	//			}
	//			catch (int e)
	//			{
	//				continue;
	//			}
	//		}
	//	}
	//	
	//	cv::imshow("Imagem Original", x);
	//	cv::waitKey(0);
	//
		//trocando canais
		/*for (int i = 0; i < x.rows; i++)
		{
			for (int j = 0; j < x.cols; j++)
			{
				cv::Vec3b bgrPixel = x.at<cv::Vec3b>(i, j);

				try
				{
					int b = bgrPixel.val[0];
					int g = bgrPixel.val[1];
					int r = bgrPixel.val[2];

					bgrPixel.val[0] = g;
					bgrPixel.val[1] = r;
					bgrPixel.val[2] = b;

			x.at<cv::Vec3b>(i, j) = bgrPixel;
				}
				catch (int e)
				{
					continue;
				}
			}
		}

		cv::imshow("Imagem Final", x);
		cv::waitKey(0);
		cv::imwrite("2.jpg", x);
*/
		//operação morfologica de abertura
		/*int erosion_size = 2;
		cv::Mat element = getStructuringElement(cv::MORPH_OPEN,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));

		cv::erode(x, x, element);

		cv::dilate(x, x, (5, 5), cv::Point(-1, -1), 1, 1, 1);
		cv::morphologyEx(x, x, cv::MORPH_OPEN, (15, 15));
		cv::imshow("Imagem Final", x);
		cv::waitKey(0);*/
		//cv::imwrite("3.jpg", x);

		//trransformando para HSV
	//	cv::Mat hsv;
	//	cv::cvtColor(x, hsv, CV_BGR2HSV);
	////	cv::Mat aux = hsv.clone();
	//	cv::imshow("Imagem Final", hsv);
	//	cv::waitKey(0);
		//cv::imwrite("4.jpg", aux);

		//reduzindo cores
		colorReduce(x);
		cv::imshow("Imagem Final", x);
		cv::waitKey(0);
		//cv::imwrite("5.jpg", x);

		//threshold
		/*src = cv::imread("1.jpg", 1);
		cvtColor(src, src_gray, CV_RGB2GRAY);
		cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
		cv::createTrackbar(trackbar_type, window_name, &threshold_type,	max_type, Threshold_Demo);
		cv::createTrackbar(trackbar_value,	window_name, &threshold_value,	max_value, Threshold_Demo);
		Threshold_Demo(0, 0);
		while (true)
		{
			int c;
			c = cv::waitKey(20);
			if ((char)c == 27)
			{
				break;
			}
		}*/

		//bordas
		/*cv::Mat gray, edge, draw;
		cv::cvtColor(x, gray, CV_BGR2GRAY);
		cv::Canny(gray, edge, 10, 250, 3);
		edge.convertTo(draw, CV_8U);
		cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
		cv::imshow("image", draw);
		cv::imwrite("6.jpg", draw);

		cv::imshow("Imagem Final", x);
		cv::waitKey(0);*/

		//circulos
		//cv::Mat moedas = cv::imread("moedas.jpg", CV_LOAD_IMAGE_COLOR);
		//cv::imshow("Imagem Final", moedas);
		//cv::waitKey(0);

		//cv::cvtColor(moedas, aux, CV_BGR2GRAY);
		//cv::GaussianBlur(aux, aux, cv::Size(9, 9), 2, 2);
		//vector<cv::Vec3f> circles;
		//cv::HoughCircles(aux, circles, CV_HOUGH_GRADIENT, 2, moedas.rows / 4, 200, 100);
		//for (size_t i = 0; i < circles.size(); i++)
		//{
		//	cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		//	int radius = cvRound(circles[i][2]);
		//	circle(moedas, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		//	circle(moedas, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
		//}
		//cv::namedWindow("circles", 1);
		//cv::imshow("circles", moedas);
		//cv::waitKey(0);

	//	//linhas - retas
		//cv::Mat predio = cv::imread("xadrez.jpg", CV_LOAD_IMAGE_COLOR);
		/*cv::Mat predio = cv::imread("predio.jpg", CV_LOAD_IMAGE_COLOR);
		cv::Mat dst;
		cv::Canny(predio, dst, 50, 200, 3);
		cv::cvtColor(dst, aux, CV_GRAY2BGR);
	#if 0
		vector<cv::Vec2f> lines;
		cv::HoughLines(dst, lines, 1, CV_PI / 180, 100);

		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0];
			float theta = lines[i][1];
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			cv::Point pt1(cvRound(x0 + 1000 * (-b)), cv::cvRound(y0 + 1000 * (a)));
			cv::Point pt2(cvRound(x0 - 1000 * (-b)), cv::cvRound(y0 - 1000 * (a)));
			line(color_dst, pt1, pt2, cv::Scalar(0, 0, 255), 3, 8);
		}
	#else
		vector<cv::Vec4i> lines;
		cv::HoughLines(dst, lines, 1, CV_PI / 180, 80, 50, 4);
		for (size_t i = 0; i < lines.size(); i++)
		{
			line(predio, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
		}
	#endif
		cv::namedWindow("Detected Lines", 1);
		cv::imshow("Detected Lines", predio);
		cv::waitKey(0);*/


		//mostrar vídeo
		//cv::VideoCapture cap("teste.mp4");

		//if (!cap.isOpened()) 
		//{
		//	cout << "Error opening video stream or file" << endl;
		//	return -1;
		//}

		//while (1) 
		//{
		//	cv::Mat frame;
		//	cap >> frame;

		//	if (frame.empty())
		//		break;
		//	
		//	cv::imshow("Frame", frame);
		//	
		//	// Press ESC on keyboard to exit
		//	char c = (char)cv::waitKey(25);
		//	if (c == 27)
		//		break;
		//}
		//
		//cap.release();
		//cv::destroyAllWindows();


		//escreve video
		//VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));


		//ler e mostrar webcam
		//cv::VideoCapture cap(0);

		//// Check if camera opened successfully
		//if (!cap.isOpened())
		//{
		//	cout << "Error opening video stream" << endl;
		//	return -1;
		//}

		//while (1)
		//{
		//	cv::Mat frame;
		//	cap >> frame;
		//	if (frame.empty())
		//		break;
		//	imshow("Frame", frame);
		//	// Press ESC on keyboard to exit
		//	char c = (char)cv::waitKey(1);

		//	if (c == 27)
		//		break;
		//}
		//// When everything done, release the video capture and write object
		//cap.release();
		//// Closes all the windows
		//cv::destroyAllWindows();

		//gravar da webcam
		//cv::VideoCapture cap(0);

		//// Check if camera opened successfully
		//if (!cap.isOpened())
		//{
		//	cout << "Error opening video stream" << endl;
		//	return -1;
		//}

		//// Default resolution of the frame is obtained.The default resolution is system dependent.
		//int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
		//int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		//// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		//cv::VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));

		//while (1)
		//{
		//	cv::Mat frame;
		//	// Capture frame-by-frame
		//	cap >> frame;
		//	// If the frame is empty, break immediately
		//	if (frame.empty())
		//		break;
		//	// Write the frame into the file 'outcpp.avi'
		//	video.write(frame);
		//	// Display the resulting frame
		//	imshow("Frame", frame);
		//	// Press ESC on keyboard to exit
		//	char c = (char)cv::waitKey(1);

		//	if (c == 27)
		//		break;
		//}
		//// When everything done, release the video capture and write object
		//cap.release();
		//video.release();
		//// Closes all the windows
		//cv::destroyAllWindows();

		////webcam com canais de cores invertidos
		//cv::VideoCapture cap(0);

		//if (!cap.isOpened())
		//{
		//	cout << "Error opening video stream" << endl;
		//	return -1;
		//}

		//while (1)
		//{
		//	cv::Mat frame;
		//	cap >> frame;

		//	if (frame.empty())
		//		break;

		//	for (int i = 0; i < frame.rows; i++)
		//	{
		//		for (int j = 0; j < frame.cols; j++)
		//		{
		//			cv::Vec3b bgrPixel = frame.at<cv::Vec3b>(i, j);

		//			try
		//			{
		//				int b = bgrPixel.val[0];
		//				int g = bgrPixel.val[1];
		//				int r = bgrPixel.val[2];

		//				bgrPixel.val[0] = r;
		//				bgrPixel.val[1] = g;
		//				bgrPixel.val[2] = b;

		//				frame.at<cv::Vec3b>(i, j) = bgrPixel;
		//			}
		//			catch (int e)
		//			{
		//				continue;
		//			}
		//		}
		//	}

		//	imshow("Frame", frame);
		//	// Press ESC on keyboard to exit
		//	char c = (char)cv::waitKey(1);

		//	if (c == 27)
		//		break;
		//}
		//// When everything done, release the video capture and write object
		//cap.release();
		//// Closes all the windows
		//cv::destroyAllWindows();

		////webcam em HSV
		//cv::VideoCapture cap(0);

		//if (!cap.isOpened())
		//{
		//	cout << "Error opening video stream" << endl;
		//	return -1;
		//}

		//while (1)
		//{
		//	cv::Mat frame;
		//	cap >> frame;

		//	if (frame.empty())
		//		break;

		//	//trransformando para HSV
		//	cv::Mat hsv;
		//	cv::cvtColor(frame, hsv, CV_BGR2HSV);

		//	imshow("Frame", hsv);
		//	// Press ESC on keyboard to exit
		//	char c = (char)cv::waitKey(1);

		//	if (c == 27)
		//		break;
		//}

		//cap.release();
		//cv::destroyAllWindows();

		////webcam com redução de cores
		//cv::VideoCapture cap(0);

		//if (!cap.isOpened())
		//{
		//	cout << "Error opening video stream" << endl;
		//	return -1;
		//}

		//while (1)
		//{
		//	cv::Mat frame;
		//	cap >> frame;

		//	if (frame.empty())
		//		break;

		//	colorReduce(frame);

		//	imshow("Frame", frame);

		//	char c = (char)cv::waitKey(1);

		//	if (c == 27)
		//		break;
		//}

		//cap.release();
		//cv::destroyAllWindows();

		////webcam destacando bordas
		//cv::VideoCapture cap(0);

		//if (!cap.isOpened())
		//{
		//	cout << "Error opening video stream" << endl;
		//	return -1;
		//}

		//while (1)
		//{
		//	cv::Mat frame;
		//	cap >> frame;

		//	if (frame.empty())
		//		break;

		//	cv::Mat gray, edge, draw;
		//	cv::cvtColor(frame, gray, CV_BGR2GRAY);
		//	cv::Canny(gray, edge, 10, 250, 3);
		//	edge.convertTo(draw, CV_8U);

		//	imshow("Frame", edge);

		//	char c = (char)cv::waitKey(1);

		//	if (c == 27)
		//		break;
		//}

		//cap.release();
		//cv::destroyAllWindows();

		////webcam destacando circulos
		//cv::VideoCapture cap(0);

		//if (!cap.isOpened())
		//{
		//	cout << "Error opening video stream" << endl;
		//	return -1;
		//}

		//while (1)
		//{
		//	cv::Mat frame;
		//	cap >> frame;

		//	if (frame.empty())
		//		break;

		//	cv::cvtColor(frame, aux, CV_BGR2GRAY);
		//	cv::GaussianBlur(aux, aux, cv::Size(9, 9), 2, 2);
		//	vector<cv::Vec3f> circles;
		//	cv::HoughCircles(aux, circles, CV_HOUGH_GRADIENT, 2, frame.rows / 4, 200, 100);
		//	for (size_t i = 0; i < circles.size(); i++)
		//	{
		//		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		//		int radius = cvRound(circles[i][2]);
		//		circle(frame, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		//		circle(frame, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
		//	}

		//	imshow("Frame", frame);

		//	char c = (char)cv::waitKey(1);

		//	if (c == 27)
		//		break;
		//}

		//cap.release();
		//cv::destroyAllWindows();

		//webcam destacando circulos
	//	cv::VideoCapture cap(0);
	//
	//	if (!cap.isOpened())
	//	{
	//		cout << "Error opening video stream" << endl;
	//		return -1;
	//	}
	//
	//	while (1)
	//	{
	//		cv::Mat frame;
	//		cap >> frame;
	//
	//		if (frame.empty())
	//			break;
	//
	//		imshow("Frame", frame);
	//
	//		char c = (char)cv::waitKey(1);
	//
	//		if (c == 27)
	//			break;
	//	}
	//
	//	cap.release();
	//	cv::destroyAllWindows();
	//
	//	return -1;


	//circulo em vídeo
	//cv::VideoCapture cap("teste.mp4");
	//
	//if (!cap.isOpened())
	//{
	//	cout << "Error opening video stream" << endl;
	//	return -1;
	//}
	//
	//while (1)
	//{
	//	cv::Mat frame;
	//	cap >> frame;
	//
	//	if (frame.empty())
	//		break;
	//
	//	cv::cvtColor(frame, aux, CV_BGR2GRAY);
	//	cv::GaussianBlur(aux, aux, cv::Size(9, 9), 2, 2);
	//	vector<cv::Vec3f> circles;
	//	cv::HoughCircles(aux, circles, CV_HOUGH_GRADIENT, 2, frame.rows / 4, 200, 100);
	//	for (size_t i = 0; i < circles.size(); i++)
	//	{
	//		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	//		int radius = cvRound(circles[i][2]);
	//		circle(frame, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
	//		circle(frame, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
	//	}
	//
	//	imshow("Frame", frame);
	//
	//	char c = (char)cv::waitKey(1);
	//
	//	if (c == 27)
	//		break;
	//}
	//
	//cap.release();
	//cv::destroyAllWindows();

}