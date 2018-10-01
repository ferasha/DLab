
#ifndef DLAB_H_
#define DLAB_H_


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

struct cameraFrame{
	cv::Mat color_img;
	cv::Mat depth_img_float;
};

class DLab: public cv::DescriptorExtractor {
public:
	DLab();
	virtual ~DLab();
	void computeImpl( const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints,
	                  cv::Mat& descriptors ) const;
	int descriptorType() const;
	int descriptorSize() const;

	cameraFrame currentFrame;

private:

    int smoothedSum(const cv::Mat& sum, const cv::Point2f& pt) const;
    double smoothedSumDepth(const cv::Mat& sum, const cv::Point2f& pt) const;
    void compute_orientation(const cv::Mat &image, const cv::Mat& depth_img, std::vector<cv::KeyPoint>& keypoints ) const;
    void getPixelPairs(int index, const cv::KeyPoint& kpt, cv::Point2f& p1, cv::Point2f& p2,
    		float c, float s) const;
    void computeDescriptors(const cv::Mat& gray, const cv::Mat& sum_depth, const std::vector<cv::Mat>& sum_color,
            std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    cv::Ptr<cv::Feature2D> surf;

    static const int patch_size = 48;
    static const int half_kernel_size = 4;
};

#endif /* DLAB_H_ */
