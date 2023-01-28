#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "myheader.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main() 
{
    VideoCapture cap(0); 
    if (!cap.isOpened()) { cout << "video error\n"; }
    Mat origin_frame; // 원본영상
    Rect roi_frame_size = Rect(Point(ROI_W_LEFT_HIGH_X, ROI_H_LEFT_LOW_Y),Point(ROI_W_RIGHT_LOW_X, ROI_H_RIGHT_LOW_Y)); //Size(340x240)
    Mat roi_frame; // 자른 영상
    Mat previous_roi_frame, present_roi_frame; // 이전영상(previous) 현재영상(present)
    extern int header_x, header_y; 
    int dest = 0; // 방향
    while(true) 
    {
        cap >> origin_frame;
        //resize(origin_frame, origin_frame, Size(VIDEO_W, VIDEO_H), 0, 0, INTER_LINEAR); //영상의 사이즈가 640x480이 아닐 때 활성화
        roi_frame = RoiImage(origin_frame, roi_frame, roi_frame_size); //Roi 이미지
        previous_roi_frame = roi_frame; //이전영상 입력
        cvtColor(previous_roi_frame, previous_roi_frame, COLOR_BGR2GRAY); //흑백영상으로 변환

        if (present_roi_frame.data) 
        {
            //SURF매칭 시작
            Ptr<SURF> obstacle = SURF::create(500, 4, 3, false, true); //SURF 매칭위한 이미지 변환
            vector<KeyPoint> previous_kPoint, present_kPoint; //SURF 특징점 저장
            Mat surf_previous, surf_present; //특징점을 이미지화 시킨 곳을 저장
            obstacle->detectAndCompute(previous_roi_frame, noArray(), previous_kPoint, surf_previous); // 특징점을 찾고 저장
            obstacle->detectAndCompute(origin_frame, noArray(), present_kPoint, surf_present); 
            //키포인트가 있을 때 매칭
            if (previous_kPoint.size() > 0 && present_kPoint.size() > 0) 
            {
                Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED); //매칭 FLANN 최소이웃근접
                vector<vector<DMatch> > knn_matches; //knn(k-클러스터링)을 이용하여 매칭
                matcher->knnMatch(surf_previous, surf_present, knn_matches, 2); //매칭
                sort(knn_matches.begin(), knn_matches.end()); //매칭된 데이터를 정렬
                const float ratio_thresh = 0.28f; //증가비율?? 자세히 모르겠음......
                vector<DMatch> good_matches; //증가비율에 따른 매칭 결과 값
                vector<vector<Point>> contours_present; //윤곽을 저장할 벡터
                vector<Point> element_present; //위 벡터에 입력할 벡터
                Point fpt_present; //필터링 된 매칭 포인트
                for (size_t i = 0; i < knn_matches.size(); i++)
                {
                    //if (previous_kPoint[knn_matches[i][0].queryIdx].size < present_kPoint[knn_matches[i][0].trainIdx].size) {
                    if (knn_matches[i][0].distance <= ratio_thresh * knn_matches[i][1].distance)
                    {
                        good_matches.push_back(knn_matches[i][0]); //위 조건으로 필터링 된 좌표 입력
                        //현재 영상에서 나온 좌표
                        fpt_present = Point(present_kPoint[knn_matches[i][0].trainIdx].pt.x, present_kPoint[knn_matches[i][0].trainIdx].pt.y);
                        element_present.push_back(fpt_present);
                        contours_present.push_back(element_present);
                    }
                    //}
                }
                //서프매칭 끝
                //매칭된 좌표의 윤곽선 그리기 시작
                vector<vector<Point>> hull_present(element_present.size());
                for (size_t i = 0; i < contours_present.size(); i++)
                {
                    convexHull(contours_present[i], hull_present[i]);
                }
                drawContours(origin_frame, hull_present, hull_present.size() - 1, Scalar(255, 255, 255));
                //매칭된 좌표의 윤곽선 그리기 끝
                //hull의 면적
                float present_area = 0;
                for (int i = 0, j = 0; i < hull_present.size() - 1; i++) 
                {
                    j = i + 1;
                    present_area += (hull_present[i][0].x * hull_present[j][0].y) - (hull_present[j][0].x * hull_present[i][0].y);
                }
                present_area = fabs(present_area) / 2.0;
                //hull의 좌표
                Moments present_mu;
                present_mu = moments(hull_present[hull_present.size() - 1]); //현재
                Point2f present_mc_pt;
                present_mc_pt = Point2f(static_cast<float>(present_mu.m10 / (present_mu.m00 + 1e-5)), static_cast<float>(present_mu.m01 / (present_mu.m00 + 1e-5))); //현재
                circle(origin_frame, present_mc_pt, 4, Scalar(255, 255, 255), -1); //현재
                //면적과 좌표 출력
                cout << "Contour area : " << present_area << " Centor Point : " << present_mc_pt << endl;
                //rectangle_site(drawing2, origin_frame, 8, CV_32S, 150, 120); //인식된 영역 그리기
                if ((present_area == 0) || ((present_mc_pt.x == 0) && (present_mc_pt.y == 0))) 
                {
                    dest = 0;
                    header_x = VIDEO_W / 2;
                    header_y = VIDEO_H / 2;
                }
                else 
                {
                    header_x = present_mc_pt.x;
                    header_y = present_mc_pt.y;
                }
                //-- 왼쪽 오른쪽 구분하기
                dest = RnL();
            }
            else 
            {
                dest = 0;
                header_x = VIDEO_W / 2;
                header_y = VIDEO_H / 2;
            }
            draw_roi(roi_frame_size, origin_frame);
            if (dest == -1) cout << "left\n";
            else if (dest == 0) cout << "centor\n";
            else if (dest == 1) cout << "right\n";
            else cout << "None\n";

            imshow("main", origin_frame); //원본영상
            //imshow("result", res); 
            //imshow("roi_main", roi_frame); //영역 지정 영상
            //imshow("previous", previous_roi_frame); //이전 영상
            //imshow("present", present_roi_frame); //현재 영상
            //imshow("drawing", drawing); //contour
        }
        swap(present_roi_frame, previous_roi_frame); //이미지 전환
        if (waitKey(20) == 27) break;
    }
}
//Colored by Color Scripter