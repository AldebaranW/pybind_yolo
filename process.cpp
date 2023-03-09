#include <iostream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace py = pybind11;

int myapproxPolyDP(std::vector<cv::Point> contours,std::vector<cv::Point> &rects, double minepsilon,
                double maxepsilon, int sides);
std::vector<cv::Point2f> matchPoints(std::vector<cv::Point> poly);

int dataprocess(int class_num, Eigen::MatrixXi mask) {
    // class_num: 类别
    // mask: 长度不定， 形式为 [[点1坐标]， [点2坐标]， ...]
    cv::Mat mask_;
    std::vector<cv::Point> poly, contours;
    cv::eigen2cv(mask, mask_);
    // cv::approxPolyDP(mask_, poly, 10, true);
    for (int i = 0; i < mask.rows(); i++) {
        cv::Point pt = mask_.at<cv::Point2i>(i);
        cv::Point2f pt_;
        pt_= {float(pt.x), float(pt.y)};
        contours.push_back(pt_);
    }
    // std::cout << contours << std::endl;
    int res = myapproxPolyDP(contours, poly, 0, 10, 4);
    if (res == 2) {
        return false;
    }

    cv::Mat rvec, tvec, rotMat;
    Eigen::Matrix3d R_T;
    Eigen::Vector3d tvec_;
    cv::Mat F_MAT = (cv::Mat_<double>(3, 3) << 2.8782842692538566e+02, 0., 2.8926852144861482e+02, 0.,
        2.8780772641312882e+02, 2.4934996600204786e+02, 0., 0., 1.);
    cv::Mat C_MAT = (cv::Mat_<double>(1, 5) << -6.8732455025061909e-02, 1.7584447291315711e-01,
        1.0621261625698190e-03, -2.1403059368057149e-03,
        -1.3665333157303680e-01);
    std::vector<cv::Point3d> pw = { {0., -75., -75.}, 
                                    {0., 75., -75.}, 
                                    {0., 75., 75.}, 
                                    {0, -75., 75.}};
    
    std::vector<cv::Point2f> poly_ = matchPoints(poly);

    cv::solvePnP(pw, poly_, F_MAT, C_MAT, rvec, tvec);
    cv::Rodrigues(rvec, rotMat);
    cv::cv2eigen(rotMat, R_T);
    cv::cv2eigen(tvec, tvec_);

    float box_transMat[12];
    box_transMat[9] = tvec_(0, 0);
    box_transMat[10] = tvec_(1, 0);
    box_transMat[11] = tvec_(2, 0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            box_transMat[i * 3 + j] = R_T(i, j);
    for (int i = 0; i < 12; i++) 
        std::cout << box_transMat[i] << std::endl;

    return true;

}


int myapproxPolyDP( std::vector<cv::Point> contours,
                    std::vector<cv::Point> &rects, double minepsilon,
                    double maxepsilon, int sides) {
    int count = 0;
    while (true) {
        std::vector<cv::Point> rect1;
        std::vector<cv::Point> rect2;
        std::vector<cv::Point> rect3; //空间回收问题
        cv::approxPolyDP(contours, rect1, minepsilon, true);
        cv::approxPolyDP(contours, rect2, maxepsilon, true);

        if (rect1.size() > sides && rect2.size() > sides) {
            rects = contours;
            return 2;
        }

        if (rect1.size() < sides && rect2.size() < sides) {
            rects = contours;
            return 2;
        }

        if (rect1.size() == sides) {
            rects.resize(sides);
            for (int i = 0; i < sides; i++) {
                rects[i] = rect1[i];
            }
            return true;
        }

        if (rect2.size() == sides) {
            rects.resize(sides);
            for (int i = 0; i < sides; i++) {
                rects[i] = rect2[i];
            }
            return true;
        } else {
            double midepsilon = (minepsilon + maxepsilon) / 2.0;
            if ((midepsilon - minepsilon) < 1e-5)
                return 0;
            cv::approxPolyDP(contours, rect3, midepsilon, true);
            if (rect3.size() < sides) {
                maxepsilon = midepsilon;
                continue;
            }
            if (rect3.size() > sides) {
                minepsilon = midepsilon;
                continue;
            }
            if (rect3.size() == sides) {
                for (int i = 0; i < sides; i++) {
                rects.resize(sides);
                rects[i] = rect3[i];
                // std::cout << "第" << count << "次循环" << std::endl;
                }
                return true;
            }
        }
    }
    return true;
}

std::vector<cv::Point2f> matchPoints(std::vector<cv::Point> poly) {
    cv::Point2f center;
    cv::Moments mu = cv::moments(poly);
    center = {mu.m10 / mu.m00, mu.m01 / mu.m00};
    std::vector<double> tg;
    double tg_[4] = {std::atan2(poly[0].y - center.y, poly[0].x - center.x),
                    std::atan2(poly[1].y - center.y, poly[1].x - center.x),
                    std::atan2(poly[2].y - center.y, poly[2].x - center.x),
                    std::atan2(poly[3].y - center.y, poly[3].x - center.x)};
    for (int i = 0; i < 4; i++) {
        tg.push_back(tg_[i]);
    }
    std::sort(tg.begin(), tg.end(), 
        [](double a, double b ){ return a < b; });

    std::vector<cv::Point2f> poly_;
    for (int i = 0; i < 4; i++) {
        if (tg[0] == tg_[i]) {
            poly_ = {
            {float(poly[i].x), float(poly[i].y)},
            {float(poly[(i + 1) % 4].x), float(poly[(i + 1) % 4].y)},
            {float(poly[(i + 2) % 4].x), float(poly[(i + 2) % 4].y)},
            {float(poly[(i + 3) % 4].x), float(poly[(i + 3) % 4].y)}};
            break;
        }
    }
    return poly_;
}

PYBIND11_MODULE(_process, m) {
    m.def("dataprocess", &dataprocess);
}

