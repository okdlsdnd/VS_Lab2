<!--21700150 김인웅-->

Lab2
======

목표 : 얼굴인식 후 체온 측정, 평균 체온이 38.0C를 넘으면 경고문 출력  

패러미터 참조를 위해 주어진 이미지

![rgbnew](https://user-images.githubusercontent.com/80805040/161732103-73dd3293-0daa-468a-be67-7028b3f36f3e.jpg)

이 이미지를 이용해 얼굴 인식과 체온 인식을 위한 패러미터를 얻을 것이다. 

### Parameter

##### Thresholding  

체온 측정을 위해선 우선 얼굴 부분만을 가져와야 한다. 이를 위해 cvtColor를 이용하여 BGR 이미지를 HSV 이미지로 바꾼다. 이후 inRange를 이용해 체온이 측정 되는 부분만 따온다. 이때 사용한 범위는 Scalar(0, 180, 138) 부터, Scalar(85, 255, 205)이다.(위 이미지에서 따온 hsv의 최소 범위는 Scalar(0, 205, 138) 이었지만 영상처리를 위해 Scalar(0, 180, 138)로 수정하였다.)

inRange를 이용하여 처리한 이미지

![thresholding](https://user-images.githubusercontent.com/80805040/161735083-0b0aaf24-dfea-44e8-8679-67a8fc3384e5.png)

이 중 얼굴만을 사용해야 한다. 따라서 findContours를 이용해서 최대 크기의 contour를 찾는다.

drawContours를 이용해서 그려낸 윤곽선은 다음과 같다.

![contours](https://user-images.githubusercontent.com/80805040/161737214-0e20b4e9-4535-43a7-9d82-e721a96a44a0.png)

그려진 Contour를 토대로 사각형을 그린다.

![rectangle](https://user-images.githubusercontent.com/80805040/161742380-9dfdd9bd-378d-4753-9ecb-1f149010d45f.png)

##### Intensity 계산  

제공된 이미지에서 Intensity는 0에서 25.0C, 255에서 40.0C를 가진다. 이를 위해 split을 이용하여 hsv 이미지에서 grayscale 이미지를 얻는다. 얻어진 grayscale 이미지는 아래와 같다.

![gray](https://user-images.githubusercontent.com/80805040/161745274-b00c839a-a4d5-4b53-b075-9b3e8426e1fc.jpg) 

이 때 contour로 얻어진 영역에 대해 Mask를 만들고 위 이미지를 bitwise_and를 이용하여 얼굴 부분의 이미지만 얻어낸다.

![gray](https://user-images.githubusercontent.com/80805040/161745976-f64452bf-bd07-443d-b333-159a94a5cd04.png)

이것으로 얻어진 온도의 계산식은 다음과 같다.

temperature = 15 * (intensity) / 255 + 25

이를 putText를 이용하여 원본 이미지에 넣는다. 이때 평균 온도는 각 픽셀의 intensity 중 상위 5%만 이용하여 얻어낸다. 이를 위해 sort를 이용해 얻어진 값을 내림차 순으로 정렬하여 계산한다.


##### Final Image  

이제 위에서 얻어낸 Contour, Rectangle, Text를 모두 원본 이미지에 덮어 씌우고 출력한다.
최종 이미지

![final](https://user-images.githubusercontent.com/80805040/161746704-7d3b54d2-f774-425f-8b41-62ac3c39c99b.png)



이 값을 이용해서 영상을 출력한다. 아래에 영상 링크를 첨부한다.  

https://youtu.be/PRp-TuiLFnM

### Appendix

##### 코드

    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace std;
    using namespace cv;
    
    Mat src, img;
    
    int main()
    {
        Mat src_disp, hsv, dst, gray;
        vector<vector<Point> > contours;
    
        VideoCapture cap("IR_DEMO_cut.avi");
    
        /*src = cv::imread("rgbnew.jpg");
    
        src.copyTo(src_disp);
    
        Mat dst_track = Mat::zeros(src.size(), CV_8UC3);
    
        cvtColor(src, hsv, COLOR_BGR2HSV);*/
    
        while (true)
        {
        cap >> img;
            if (img.empty()) {
                return 0;
            }
    
            bool bSuccess = cap.read(src);
    
            Mat dst_track = Mat::zeros(src.size(), CV_8UC3);
    
            src.copyTo(src_disp);
    
            cvtColor(src, hsv, COLOR_BGR2HSV);
    
            inRange(hsv, Scalar(0, 180, 138), Scalar(85, 255, 205), dst);
    
            namedWindow("dst", CV_WINDOW_NORMAL);
            cv::imshow("dst", dst);
    
            findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
            if (contours.size() > 0)
            {
                /// Find the Contour with the largest area ///
                double maxArea = 0;
                int maxArea_idx = 0;
    
                for (int i = 0; i < contours.size(); i++)
                    if (contourArea(contours[i]) > maxArea) {
                        maxArea = contourArea(contours[i]);
                        maxArea_idx = i;
                    }
    
                ///  Draw the max Contour on Black-background  Image ///
                Mat dst_out = Mat::zeros(dst.size(), CV_8UC3);
                if (maxArea > 10000) {
                    drawContours(dst_out, contours, maxArea_idx, Scalar(0, 0, 255), 2, 8);
                }
                namedWindow("Contour", 0);
                imshow("Contour", dst_out);
    
                /// Draw the Contour Box on Original Image ///
                Rect boxPoint = boundingRect(contours[maxArea_idx]);
                if (maxArea > 10000) {
                    drawContours(src_disp, contours, maxArea_idx, Scalar(255, 255, 255), 2, 8);
                    rectangle(src_disp, boxPoint, Scalar(255, 0, 255), 3);
                }
    
                /// Continue Drawing the Contour Box  ///
                if (maxArea > 10000) {
                    rectangle(dst_track, boxPoint, Scalar(255, 0, 255), 3);
                }
                namedWindow("Contour_Track", 0);
                imshow("Contour_Track", dst_track);
    
                /// Draw the Mask for the Gray Image
                Mat gray = Mat::zeros(dst.size(), CV_8UC1);
                if (maxArea > 10000) {
                    vector<vector<Point> > contours2;
                    contours2.push_back(contours[maxArea_idx]);
                    Mat mask = Mat::zeros(dst.size(), CV_8UC1);
                    drawContours(mask, contours2, -1, 255, -1);
    
                    int height = src.rows;
                    int width = src.cols;
    
                    Mat roi(src, Rect(0, 0, width, height));
    
                    Mat result;
                    bitwise_and(roi, roi, result, mask);
    
                    vector<Mat> channels;
                    split(result, channels);
                    gray = channels[2];
    
                    /// Put texts on Original Image
                    double avgVal = 0;
                    double maxVal = 0;
                    int zero = 0;
                    std::vector<int> Val;
                    std::vector<int> Sort;
                    for (int v = 0; v < gray.rows; v++)
                        for (int u = 0; u < gray.cols; u++)
                            if (gray.at<uchar>(v, u) == 0)
                            {
                                continue;
                            }
                            else {
    
                                Val.push_back(gray.at<uchar>(v, u));
    
                                if (maxVal < gray.at<uchar>(v, u)) {
                                    maxVal = gray.at<uchar>(v, u);
                                }
                            }
    
                    cv::sort(Val, Sort, SORT_DESCENDING);
    
                    for (int i = 0; i < Sort.size() * 0.05; i++) {
                        avgVal += Sort[i];
                    }
                    avgVal /= (Sort.size() * 0.05);
    
                    double avg = 15 * avgVal / 255 + 25;
                    double max = 15 * maxVal / 255 + 25;
    
                    char mystr_1[40];
                    char mystr_2[40];
    
                    sprintf_s(mystr_1, "Average Temperature : %f", avg);
                    sprintf_s(mystr_2, "Max Temperature : %f", max);
    
                    putText(src_disp, mystr_1, Point(20, 20), 5, 1, Scalar(255, 255, 255), 1);
                    putText(src_disp, mystr_2, Point(20, 40), 5, 1, Scalar(255, 255, 255), 1);
                    if (avg > 38) {
                        putText(src_disp, "Warning", Point(20, 60), 5, 1, Scalar(0, 0, 255), 1);
                    }
    
                    cout << "average temperature = " << avg << endl;
                    cout << "mamximum temperature = " << max << endl;
                }
                namedWindow("Contour_Box", 0);
                imshow("Contour_Box", src_disp);
    
                namedWindow("gray", 0);
                imshow("gray", gray);
            }
    
            if (waitKey(30) == 27)
                break;
    
            char c = (char)waitKey(10);
            if (c == 27)
                break;
    
        }
    
    }


##### Flow Chart

![flowchart](https://user-images.githubusercontent.com/80805040/161751699-2f91a0fd-970a-4654-894a-32e68c9e0fa2.png)
