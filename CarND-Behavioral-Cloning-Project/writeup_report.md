# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track-one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "NVIDIA Model Visualization"
[image2]: ./examples/model_implementation.png "NVIDIA Model Implementation"

[image3]: ./examples/hist_before_proc.png "Histogram before data processing"
[image4]: ./examples/hist_after_proc.png "Histogram after data processing"
[image5]: ./examples/original_center.png "Original center image"
[image6]: ./examples/original_center_flipped.png "Flipped center image"
[image7]: ./examples/original_right.png "Original right image"
[image8]: ./examples/original_left.png "Original left image"
[image9]: ./examples/resized_center.png "Resized center image"
[image10]: ./examples/training_output.png "Training output"


## Rubric Points
### In the following, it is described how the requirements resulting from the [rubric points](https://review.udacity.com/#!/rubrics/432/view) are individually addressed in my implementation of the beavioral cloning project. 

---
### Files Submitted & Code Quality

#### 1. The submitted packet includes all required files and can be used to run the simulator in autonomous mode

My submission includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network 
* writeup_report.md summarizing the results
* video file with the car driving autonomously in track one for two laps

#### 2. The submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model follows closely the NVIDIA ConvNet model.  The latter consists of a normalization layer (using a Keras lambda layer), five convolutional layers and four fully connected layers.  The images were resized to match the NVIDIA paper's original size (i.e. 66 x 200) and converted to the YUV colorspace.

![alt text][image1] NVIDIA Model Visualization

The model includes ELU layers to introduce nonlinearity.  Lastly, one cropping and three dropout layers were utilized to remove unwanted features from the image and to reduce the likelihood of overfitting the training data.  See below for the implementation of NVIDIA's model.

![alt text][image2] NVIDIA Model Implementation

#### 2. Attempts to reduce overfitting in the model

As mentioned before, the model contains dropout layers in order to reduce overfitting between the fully connected layers.  Three dropout layers were used that implement 50%, 25% and 10% dropout rates respectively.  Furthermore, the initial set of input data was manipulated to remove biases resulting from the large number of near-zero steering angles.

The model was trained and validated on the Udacity-provided and other collected data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and by ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of the Udacity-provided data along with collected data running the simulator in training mode for center lane driving, recovering from the left and right sides of the road.  The model performed well with the Udacity data alone, so in the final model, all other obtained data were removed.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Once the initial data set and the model were determined, my focus was to implement the training in an efficient manner, given that it would run on the local CPU.  For this reason, I used a generator for both training and validating sets.  

In the generator function, the data was shuffled for each epoch and was separated into batches of 16 frames.  For each frame in a batch, the center, left and right images were resized from 160x320 to 161x200.  Each of the left, right and center images was flipped horizontally and added to an augmented set of images.  For the flipped images, the steering angles' sign was also flipped.  For the left and right images' angles, a shift was added and subtracted respectively from/to the steering angle of the center image.  Here, the steering delta was selected to be 0.2 degrees.

![alt text][image5] Original center image
![alt text][image6] Flipped center image
![alt text][image7] Original right image
![alt text][image8] Original left image
![alt text][image9] Resized center image

Once the augmented set was generated, training and validation generators were defined for the training and validation sets respectively.  As mentioned earlier, the implemented model was based on the NVIDIA model.  The modifications included the addition of a cropping layer to rescale the images from 161x200 to 66x200, ELU non-linearity layers and dropout layers between the fully connected layers.  

In the cropping layer, the top 70 and the bottom 26 rows of pixels were discared that contained landscape details --such as trees, hills-- and the hood of the car, which could ultimately interfere with the learning process and affect performance dramatically.  This resulted to a resized input image of dimensions 66 x 200, as proposed by NVIDIA's paper.  A mean square error loss function along with an adam optimizer were used in the training.

Overfitting was addressed with the introduction of dropout layers using dropout rates of 50%, 25% and 10% respectively (note: the closer to the output node, the lower the dropout rate).  Also, to avoid biasing the model drive on a straight line, the number of very small steering angles in the initial data set was limited.  This lead up to removing a large number of samples from the set.  Despite this, the model seemed to perform better.

![alt text][image3] Histogram before data processing
![alt text][image4] Histogram after data processing

The model was trained for three epochs, each processing 3059 images.  At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road for multiple laps.

#### 2. Final Model Architecture

The final model architecture consisted of the convolutional neural network proposed by NVIDIA.  The latter consists of a normalization layer, five convolutional layers, four fully connected layers.  Between the fully connected layers, dropout layers have been added with dropout rates of 50%, 25% and 10% respectively.  ELU layers were added for non-linear modeling, along with dropout layers to reduce overfitting.

The model fails to drive the car in the most challenging track (track-two).  What is required to accomplish this is obtain additional driving data (images and steering angles) from track-two and add them to the training set.  Also, it is likely that more small angles should be removed from the initial set, as several autonomous runs have indicated that the car tends to go on a straight line or not turn sufficiently on sharp turns in track-two.  Lastly, more epochs than three could be used.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving.  When tried to train the model using this data, the predicted steering angles were causing the car to drive off the track immediately.  Several dropout layers were added and removed, but there seemed to be an issue with the amount and quality of the data, along with perhaps some biasing.  In the end, I decided to keep the Udacity data only, since the car was able to complete several laps in the first track without going off road.

The frames with corresponding small angles appeared to be disproportionately more compared to the rest of the angles.  For that reason, the number of frames with angles close to zero was limited to a smaller number.  Thus, this reduced the initial set from 6428 images to 3059.

![alt text][image10] Training output

To augment the original Udacity data set, the images were flipped horizontally along with flipping the sign of the corresponding steering angles.  Also, the images from the left and right cameras were used with an appropriate shift in the steering angle (add shift for the left camera, subtract for the right camera).  After the collection process, I had 3059 number of data points.  I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
