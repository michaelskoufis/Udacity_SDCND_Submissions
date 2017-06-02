# **Finding Lane Lines on the Road** 

## Project 1 Writeup

### This report contains a brief overview of Project 1, discussess the approach used along with suggested improvements.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following

* Make a pipeline that finds lane lines on the road
* Reflect on the work in a written report

[//]: # (Image References)

[image1]: ./test_images/solidWhiteCurve.jpg
[image2]: ./examples/solid_white_curve_grayscale.png
[image3]: ./examples/solid_white_curve_gaussian.png
[image4]: ./examples/solid_white_curve_edges.png
[image5]: ./examples/solid_white_curve_masked_edges.png
[image6]: ./examples/solid_white_curve_hough.png
[image7]: ./examples/solid_white_curve_extrapolated.png
[image8]: ./examples/solid_white_curve_overlaid.png


### Reflection

### 1. Pipeline description pipeline.

My pipeline consisted of 7 steps. 

* Convert to grayscale
* Apply Gaussian filtering
* Generate the Canny edges
* Apply mask to region of interest
* Generate the Hough image
* Extrapolate the full extent of the lane lines
* Superimpose the extrapolated lane lines onto the original image

Converting to grayscale, applying Gaussian filtering, generating the Canny edges, masking-in the area of interest and generating the Hough transformation followed closely the lecture material.  See images below.

![alt text][image1] Original

![alt text][image2] Grayscale

![alt text][image3] Gaussian

![alt text][image4] Canny edges

![alt text][image5] Masked Canny edges

![alt text][image6] Hough


One thing I decided to do differently was NOT to modify the draw_lines function, but generate the required functionality in a different Python function that I called *__extrapolate_lane_lines()__*.

The _extrapolate_lane_lines()_ function reads in the Hough image and identifies points at the far and at the close ends of each of the lanes.  To do this, I decided not to resort to slope calculation, but look at the coordinates.  Specifically, the algorithm attempts to find those coordinates that have:

* lowest  x coordinate AND lowest  y coordinate (left top)
* lowest  x coordinate AND highest y coordinate (left bottom)
* highest x coordinate AND lowest  y coordinate (right top)
* highest x coordinate AND highest y coordinate (right bottom)

Once those points are determined, the algorithm further extrapolates to the bottom and to the top of the masked region (when possible).  Once we have determined the four endpoints --two for each line-- we simply draw the lane lines and return the result.  Below is the result of the extrapolated image before and after it is overlaid to the original image.

![alt text][image7] Extrapolated lines

![alt text][image8] Final


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be the cases in which the extrapolation of the lane lines does not prove to be as stable as one would expect.  


### 3. Suggest possible improvements to your pipeline

This could probably be improved if additional filtering is applied to the image before running the Hough transformation. Alternatively, the Hough threshold could be tweaked to generate a less noisy image, with the risk of losing a few lines in the output.
