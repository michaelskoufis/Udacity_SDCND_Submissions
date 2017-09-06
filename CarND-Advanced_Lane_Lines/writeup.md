## Advanced Lane Finding Project - Report

### This document summarizes the methodology used to solve this challenge and provides a visual demonstration of the implemented solution.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Camera Calibration"
[image2]: ./output_images/straight_lines_1.png "Camera Calibration ex1"
[image3]: ./output_images/straight_lines_2.png "Camera Calibration ex1"
[image4]: ./output_images/test1.png "Camera Calibration ex.1"
[image5]: ./output_images/test2.png "Camera Calibration ex.2"
[image6]: ./output_images/test3.png "Camera Calibration ex.3"
[image7]: ./output_images/test4.png "Camera Calibration ex.4"
[image8]: ./output_images/test5.png "Camera Calibration ex.5"
[image9]: ./output_images/test6.png "Camera Calibration ex.6"
[image10]: ./output_images/color_binary.png "Color Binary"
[image11]: ./output_images/grad_mag_binary.png "Gradient Magnitude Binary"
[image12]: ./output_images/grad_dir_binary.png "Gradient Direction Binary"
[image13]: ./output_images/combined_binary.png "Combined Binary"
[image14]: ./output_images/masked_binary.png "Masked Binary"
[image15]: ./output_images/perspect_before.png "Before Perspective Transformation"
[image16]: ./output_images/perspect_after.png "After Perspective Transformation"
[image17]: ./output_images/tile_tracing.png "Rectangle tracing"
[image18]: ./output_images/polynom_fit_1.png "Polynomial Fit"
[image19]: ./output_images/polynom_fit_2.png "Polynomial Fit"
[image20]: ./output_images/final.png "Final Output Image"
[image21]: ./output_images/perspect_before_short.png "Before Perspective Transformation"
[image22]: ./output_images/error.png "Error Before Termination"

[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration takes place in the `CalibrateCamera()` function.  In there, a mapping between object points in the 3D world and 2D image points on the camera image plane is implemented.  The regular image of a chessboard from different angles is used due to its regular and repeating patterns to identify camera discrepancies.  OpenCV already offers modules that address this process.  These functions are the following: `cv2.findChessboardCorners()` and `cv2.drawChessboardCorners()`.  Upon deriving the camera matrix and the distortion coefficients, a call to `cv2.undistort()` applies the distortion coeffcients, producing thus an undistorted image.  See chessboard image below.  The top is the original image.  The bottom is the undistorted image.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Additional examples of calibration follow below.  The original images were provided and reside in the `test_examples/` directory. 

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Three separate binaries were generated: color, gradient magnitude and gradient direction.  

```python
    color_binary    = ColorThresholdingHLS(undistorted, color_thresh)
    grad_mag_binary = GradientMagThreshold(undistorted, 3, grad_mag_thresh)
    grad_dir_binary = GradientDirThreshold(undistorted, 3, grad_dir_thresh)
```

The min/max thresholds were the following respectively.

```python
   color_thresh = (180, 255)
   grad_mag_thresh = (85, 255)
   grad_dir_thresh = (0.9, 1.1)
```

The color thresholding is implemented in the function `ColorThresholdingHLS()`.  The color space is converted from RGB to HLS and the image is subsequently thresholded into a binary image.

![alt text][image10]

The gradient magnitude thresholding is implemented in the `GradientMagThreshold()` function.  A Sobel operator in the x direction is implemented, scaled and thresholded.

![alt text][image11]

The gradient direction thresholding in `GradientDirThreshold()` didn't pay off as anticipated.  This relates most likely to the chosen thresholds.

![alt text][image12]

Given that the color and gradient magnitude binaries have had the most useful results, those two were used to produce a combined binary image.

```python
combined[((color_binary == 1) | (grad_mag_binary == 1))] = 1
```

![alt text][image13]

Subsequently, the combined binary image is fed into a masking function to remove landscape and other undesired details.

![alt text][image14]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The two basic functions that prepare the data before the perspective transformation are the following.

```python
    # Apply a perspective transform to rectify binary image ("birds-eye view")
    src  = GetPerspectiveSourcePoints(masked)
    dest = GetPerspectiveDestinationPoints(masked)
```

The first one gets four source points in the original image and the second one chooses four points in the destination image where the previous points will map to.  An example of selected source points in the original image is shown below.

![alt text][image21]

The selected top and bottom points that form the trapezoidal are extrapolated to the top and bottom of the visible binary image.

![alt text][image15]

The destination points are selected as the corners of a rectange that includes most of the pixel space in the destination image.

![alt text][image16]

Then, by computing the transformation matrix, the image is warped successfully using the OpenCV `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` functions.

```python
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(np.float32([src]), np.float32([dest]))
    
    # Compute the inverse perspective transform:
    Minv = cv2.getPerspectiveTransform(np.float32([dest]), np.float32([src]))

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(masked, M, masked.shape[::-1])
```

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is done in the `FindLaneLineStart()` and `TrackSlidingWindows()` functions.  A historgram is used on the warped image to identify the x pixel coordinates of the line starting points.  Then, a method was implemented same as the one in the lectures to use sliding tiles or rectangles or windows to track the lines to the top end.

![alt text][image17]

Then, given the points, polynomials of the form Ay<sup>2</sup>+By+C are fitted to the lines.  The fitting takes place in the `FitPolynomialToSlidingWindows()` function.

![alt text][image18]
![alt text][image19]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This was done in the same way as in the lectures.  Assuming that the previous data fitting works well for known A, B and C, then the curvature is computed as R = (1+(2Ay+B<sup>2</sup>))<sup>3/2</sup>/|2A|.  The notation `y` corresponds to the y value at the bottom of the image, that is the number of rows in the image.  The above is computed in the function `ComputeLaneCurvature()`.

The car offset from the middle of the lane is calculated in `ComputeOffsetFromCenter()`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

At the end of the pipeline and following an inverse perspective transformation, this is the output image with the full extent of the lane showing on the road.

![alt text][image20]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Besides the wobbly lane boundary lines in some cases, the main issue is that the algorithm process the first 30 seconds of the video and then terminates abnormally.  So far, this is not rersolved.  It is expected that the cause is a data type conflict.  Fixing it is an on-going effort, although no progress has been made recently.  See below for error message.

![alt text][image22]


