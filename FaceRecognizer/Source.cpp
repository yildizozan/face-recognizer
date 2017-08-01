#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

#define IMAGE_SIZE 200
#define MAX_PIC 7

void detectAndDisplay(Mat &frame, Ptr<FaceRecognizer> &model);
Ptr<FaceRecognizer> trainer();

string face_cascade_name = "haarcascade_frontalface_default.xml";

CascadeClassifier face_cascade;

int main(int argc, char** argv)
{

	if (!face_cascade.load(face_cascade_name))
	{
		cout << "Error face_cascade" << endl;
		system("pause");
		return -1;
	}


	// Read the video stream
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Video stream not read!" << endl;
		system("pause");
		return -1;
	}

	// Train
	Ptr<FaceRecognizer> model = trainer();

	// Starting reading frame
	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "Empty frame!" << endl;
			break;
		}

		detectAndDisplay(frame, model);

		int c = waitKey(30);
		if ((char)c == 27)
		{
			break;
		}
	}

	return 0;
}

void detectAndDisplay(Mat &frame, Ptr<FaceRecognizer> &model)
{
	vector<Rect> faces;
	Mat frame_gray;

	string username;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.6);

	// All faces draw rectangle
	for (size_t i = 0; i < faces.size(); ++i)
	{
		Rect face_i = faces[i];
		Mat face = frame_gray(face_i);

		//rectangle(frame, Point(face_i.x, face_i.y), Point(face_i.x + face_i.width, face_i.y + face_i.height), Scalar(255, 0, 255), 2);
		rectangle(frame, Point(face_i.x, face_i.y), Point(face_i.x + face_i.width, face_i.y + face_i.height), Scalar(255, 0, 255), 2);

		// Who I am
		Mat face_resize;
		resize(face, face_resize, Size(IMAGE_SIZE, IMAGE_SIZE));
		int predictLabel;
		double predictVal;
		model->predict(face_resize, predictLabel, predictVal);
		switch (predictLabel)
		{
		case 0:
			username = "Ozan";
			break;
		case 1:
			username = "Alkan";
			break;
		case 2:
			username = "Obama";
			break;
		case 3:
			username = "angelinajolie";
			break;
		case 4:
			username = "arnoldschwarzenegger";
			break;
		case 5:
			username = "bradpitt";
			break;
		case 6:
			username = "georgeclooney";
			break;
		case 7:
			username = "johnnydepp";
			break;
		case 8:
			username = "katyperry";
			break;
		default:
			username = "Taniyamadim ki! :(";
			break;
		}
		putText(frame, username, cvPoint(face_i.x, face_i.y + face_i.height + 30), CV_FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 2, CV_AA);

		// Predict result
		string box_text = format("Predict value = %f", predictVal);
		putText(frame, box_text, Point(face_i.tl().x - 10.0, face_i.tl().y - 10.0), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
	}

	// Showing
	imshow("Pencere", frame);
}

Ptr<FaceRecognizer> trainer()
{
	/*in this two vector we put the images and labes for training*/
	vector<Mat> images;
	vector<int> labels;

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\ozan\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(0);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\alkan\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(1);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\obama\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(2);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\angelinajolie\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(3);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\arnoldschwarzenegger\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(4);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\bradpitt\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(5);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\georgeclooney\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(6);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\johnnydepp\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(7);
	}

	for (size_t i = 1; i <= MAX_PIC; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\katyperry\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(8);
	}

	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	return model;
}