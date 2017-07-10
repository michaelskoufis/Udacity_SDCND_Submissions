# **Traffic Sign Recognition** 

## Project Description

### The following presents a solution to the street sign classification problem.  It discusses data preprocessing, the neural network architecture used to implement the classification model, as well as results such as validation and testing accuracies.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the German Traffic Signs dataset
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/dataset_barchart_initial.png "Initial Dataset"
[image2]: ./images/code_gen_dist_imgs.png "Generate Distorted Images"
[image3]: ./images/dataset_sizes_25pc.png "Enhanced Dataset 25%"
[image4]: ./images/dataset_sizes_50pc.png "Enhanced Dataset 50%"
[image5]: ./images/dataset_barchart_25.png "Enhanced Dataset 25%"
[image6]: ./images/dataset_barchart_50.png "Enhanced Dataset 50%"
[image7]: ./images/dataset_barchart_75.png "Enhanced Dataset 75%"

[image8]:  ./images/sign1.jpg "Traffic Sign 1"
[image9]:  ./images/sign2.jpg "Traffic Sign 2"
[image10]: ./images/sign3.jpg "Traffic Sign 3"
[image11]: ./images/sign4.jpg "Traffic Sign 4"
[image12]: ./images/sign5.jpg "Traffic Sign 5"

[image13]: ./images/original1.png "Case 1 - Original"
[image14]: ./images/affline1.png  "Case 1 - Affline Transformation"
[image15]: ./images/original2.png "Case 2 - Original"
[image16]: ./images/blur2.png     "Case 2 - Gaussian Blur"
[image17]: ./images/original3.png "Case 3 - Original"
[image18]: ./images/translation3.png "Case 3 - Translation"
[image19]: ./images/original4.png "Case 4 - Original"
[image20]: ./images/rotation4.png "Case 4 - Rotation"
[image21]: ./images/original5.png "Case 5 - Original"
[image22]: ./images/perspective5.png "Case 5 - Perspective"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This is the present document.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Standard python and matplotlib's' pyplot library were used to calculate and plot summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data differs from class to class.  In particular, the number of sample images varries considerably.  See below for the distribution of example images per class.

![alt text][image1]

To avoid training an inefficient model that would predict well some classes and poorly others, new data were generated.  See the next section for the distribution of examples for each class after the data enhancement.

### Design and Test a Model Architecture

#### 1.  Data Preprocessing

The original images were converted to grayscale from RGB and subsequently were normalized to a 0-mean and equal variance.  This was done to assist with training the network and improve accuracy.  Then, the grayscale normalized data were further processed in various ways.  For each class or else a street sign example, a representative image was picked from the initial dataset randomly.  On that image, various distortions were applied: Gaussian blur, contrast enhancement, brightness enhancement, transformations such as affine & perspective, rotations and lastly translations.  See the code for how it was done.  A screenshot of the snipped of the code has also been included below.

![alt text][image2]

A series of examples are given below showcasing some of the applied distortions.

Case 1 - Original
![alt text][image13]

Case 1 - Affline Transformation
![alt text][image14]

Case 2 - Original
![alt text][image15]

Case 2 - Gaussian Blur
![alt text][image16]

Case 3 - Original
![alt text][image17]

Case 3 - Translation
![alt text][image18]

Case 4 - Original
![alt text][image19]

Case 4 - Rotation
![alt text][image20]

Case 5 - Original
![alt text][image21]

Case 5 - Perspective
![alt text][image22]

Also, find a chart showing the per-class distribution of the examples after the data enhancement for the three main sets: the training, validation and test sets.  Several runs were performed to cover various percentages of the gap in the examples between each class and the class with the maximum number of examples.  This was repeated for each of the three main sets.  See below for the plots for the 25%, 50% and 75% coverage of the gap respectively.

Dataset sizes at 25% gap coverage
![alt text][image3]

Dataset sizes at 50% gap coverage
![alt text][image4]

Dataset barchart at 25% gap coverage 
![alt text][image5]

Dataset barchart at 50% gap coverage 
![alt text][image6]

Dataset barchart at 75% gap coverage 
![alt text][image7]

The final size of the sets post-enhancement were (25% case):

* The size of training set is 47695
* The size of the validation set is 5875
* The size of test set is 17526


#### 2. Model Architecture

My final model was largely based on the LeNet network topology and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x36 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Fully connected		| Input = 1600, Output = 120        			|
| RELU					|												|
| Fully connected		| Input = 120, Output = 84        				|
| RELU					|												|
| Fully connected		| Input = 84, Output = 43        		        |
| Softmax				|         									    |
|				        |         									    

#### 3. Training the Model

To train the model, a batch size of 128, a learning rate of 0.001 and an iterative approach of 12 epochs were used.  The network is trained by running an Adam optimizer on the cross-entropy cost function.

#### 4.  Validation set accuracy 

In the data pre-processing section, it was mentioned that distortion methods were introduced in order to enhance the datasets and ultimately improve the set accuracy (training, validation or test) by preventing overfitting for some classes and underfitting for others.  What could have been also added was a dropout mechanism to prevent the model from following closely specific samples from the dataset.

Choosing the parameters for the LeNet architecture was an iterative approach.  Essentially, various number of filters --following each of the two convolution layers-- were tried to produce the feature maps.  In the end, filters of size 36 and 64 respectively for the two convolution layers gave satisfying results compared to other options.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.940 (max of 0.957 in Epoch 10)
* test set accuracy of 0.919

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] 
![alt text][image9] 
![alt text][image10] 
![alt text][image11] 
![alt text][image12]

The images were resized to a 32x32 resolution anticipated as input by LeNet.  Also, the images were grayscaled and normalized to a 0-mean.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

The accuracy for those signs was not good.  Perhaps during resampling, certain features were lost leading to a confusion in the LeNet prediction.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		|    	Children crossing						| 
| Roundabout     		|  		Pedestrians								|
| Speed Limit (50)		| 		Children crossing						|
| No Entry	      		| 		Pedestrians			 				    |
| Speed Limit (60)		|       Children crossing						|


The model was not able to guess correctly any of the street signs.  This could be due to the resampling of the original image and hence loss in the quality of the image or because of lack of a sufficient number of representative examples for this sign in the original training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top five soft max probabilities were (although pretty low):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.09829857        	|    Children crossing							| 
| 0.1619356     		| 	 Pedestrians								|
| 0.12122423 			| 	 Children crossing							|
| 0.10702604	      	| 	 Pedestrians				 				|
| 0.13054362			|    Children crossing   						|



