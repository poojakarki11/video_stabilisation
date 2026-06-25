// Project 3 - Video Stabilisation
//
// Notes:
//   * For each pair of consecutive frames, estimate the rigid (translation +
//     rotation) motion between them using Shi-Tomasi corners tracked with
//     Lucas-Kanade optical flow.
//   * Accumulate that motion into a trajectory (cumulative dx, dy, da).
//   * Smooth the trajectory with a Gaussian-weighted sliding window of size N.
//   * For the central frame in the window, build a transform that moves it
//     from its actual trajectory position to the smoothed position & apply
//     it with warpAffine().
//   * Display the result with a green border.
//
// Smoothing scalar trajectories (dx, dy, da) avoids the instability of averaging full projective matrices, which was causing the frame to sweep.

#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <cmath>
#include <iostream>

// Tunable parameters
// Smoothing window size N (odd, so there is a well defined central frame)
static const int    N     = 19;
// Standard deviation of the Gaussian used for the weighted average
static const double SIGMA = 8.0;
// Crop/zoom factor: scaling the output up slightly so the shifted frame edges stay outside the visible window, hiding the black borders. 1.05 = 5% zoom.
static const double ZOOM  = 1.05;


// One step of inter-frame motion: translation (dx, dy) and rotation (da).
struct Motion {
    double dx = 0.0;
    double dy = 0.0;
    double da = 0.0;   // rotation angle, radians
};

// Estimate the rigid motion between two frames using optical flow.
// Returns zero motion if estimation fails so a bad frame does no harm
static Motion estimateMotion(const cv::Mat &prevGray, const cv::Mat &currGray)
{
    Motion m;

    std::vector<cv::Point2f> prevPts, currPts;
    cv::goodFeaturesToTrack(prevGray, prevPts, 400, 0.01, 10);
    if (prevPts.size() < 10)
        return m;

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray, currGray, prevPts, currPts, status, err);

    std::vector<cv::Point2f> goodPrev, goodCurr;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            goodPrev.push_back(prevPts[i]);
            goodCurr.push_back(currPts[i]);
        }
    }
    if (goodPrev.size() < 10)
        return m;

    // Rigid transform (translation + rotation, no scale/shear): robust and
    // exactly the degrees of freedom that matter for hand-shake.
    cv::Mat T = cv::estimateAffinePartial2D(goodPrev, goodCurr);
    if (T.empty())
        return m;

    m.dx = T.at<double>(0, 2);
    m.dy = T.at<double>(1, 2);
    m.da = std::atan2(T.at<double>(1, 0), T.at<double>(0, 0));

    // Rejecting implausibly large single-frame motion
    if (std::abs(m.dx) > 100.0 || std::abs(m.dy) > 100.0 || std::abs(m.da) > 0.3)
        return Motion();   // zero

    return m;
}

int main(int argc, char **argv)
{
    cv::VideoCapture cap;
    if (argc > 1)
        cap.open(argv[1]);
    else
        cap.open(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video source." << std::endl;
        return -1;
    }

    // Pre-computing the Gaussian weights for the smoothing window.
    const int centre = N / 2;

    cv::namedWindow("Input",  cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);

    std::deque<cv::Mat>     frameBuf;   // buffer of N frames
    std::deque<cv::Point3d> trajBuf;    // cumulative trajectory (x, y, angle)

    cv::Mat prevGray;
    cv::Point3d cumTraj(0, 0, 0);       // running cumulative trajectory

    cv::Mat frame;
    bool finished = false;

    // Render the frame at buffer index 'c' using a smoothing window over the first 'count' entries, then showing it with a green border.
    auto renderCentral = [&](int c, int count) {
        double sx = 0, sy = 0, sa = 0, wsum = 0;
        for (int i = 0; i < count; ++i) {
            double d = i - c;
            double w = std::exp(-(d * d) / (2.0 * SIGMA * SIGMA));
            sx += w * trajBuf[i].x;
            sy += w * trajBuf[i].y;
            sa += w * trajBuf[i].z;
            wsum += w;
        }
        sx /= wsum; sy /= wsum; sa /= wsum;

        // Correction = smoothed - actual: how much to move the central frame to place it on the smooth path.
        double cdx = sx - trajBuf[c].x;
        double cdy = sy - trajBuf[c].y;
        double cda = sa - trajBuf[c].z;

        // Build the affine warp: rotation + translation to stabilise the frame.
        // small zoom about the image centre to hide the exposed black borders.
        double cx = frameBuf[c].cols / 2.0;
        double cy = frameBuf[c].rows / 2.0;

        double cosA = std::cos(cda) * ZOOM;
        double sinA = std::sin(cda) * ZOOM;

        // applying the stabilising shift. Folded into a single 2x3 matrix
        cv::Mat warp = (cv::Mat_<double>(2, 3) <<
            cosA, -sinA, cx - cosA * cx + sinA * cy + cdx,
            sinA,  cosA, cy - sinA * cx - cosA * cy + cdy);

        cv::Mat output;
        cv::warpAffine(frameBuf[c], output, warp, frameBuf[c].size());

        cv::rectangle(output, cv::Point(0, 0),
                      cv::Point(output.cols - 1, output.rows - 1),
                      cv::Scalar(0, 255, 0), 3);

        cv::imshow("Output", output);
    };

    while (!finished) {
        bool grabbed = cap.read(frame);

        if (grabbed && !frame.empty()) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            if (!prevGray.empty()) {
                Motion m = estimateMotion(prevGray, gray);
                cumTraj.x += m.dx;
                cumTraj.y += m.dy;
                cumTraj.z += m.da;
            }
            // (first frame: cumTraj stays at 0,0,0)

            frameBuf.push_back(frame.clone());
            trajBuf.push_back(cumTraj);
            prevGray = gray;

            cv::imshow("Input", frame);
        } else {
            finished = true;
        }

        // Once N frames are buffered, emitting the central one; at end & drain.
        while ((int)frameBuf.size() >= N || (finished && !frameBuf.empty())) {
            int n = (int)frameBuf.size();

            if (n >= N) {
                renderCentral(centre, N);
            } else {
                int c = std::min(centre, n - 1);
                renderCentral(c, n);
            }
            frameBuf.pop_front();
            trajBuf.pop_front();

            int key = cv::waitKey(30);
            if (key == 27) {           // ESC quits
                finished = true;
                frameBuf.clear();
                trajBuf.clear();
                break;
            }
        }

        if (!finished) {
            int key = cv::waitKey(1);
            if (key == 27)
                finished = true;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}