# **Traffic Sign Recognition** 

## Writeup



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary of Dataset

I used the numpy, pandas and matplotlib libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43
* Maximum data in one class is 2010
* Minimum data in one class is 180

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

First let's look at what a random image in the set looks like. Here is a set of 45 random images drawn from the training set.

![Random images](./examples/download1.png)

Now, since these images are random and repititive let's look at one image from each class or sign labels.

![Class images](./examples/download2.png)

Pretty good! Since we know that there are 43 classes and each class has a different kind of image lets look at first 15 images from each class so that we get to visualize the difference in images  in each class.

![15 images](./examples/download3.png)

We can see that the data has same images repeated with a slight variation in each one. As mentioned, it can be due to the fact that they are sampled from a 1 second video feed on each sign. This creates a strong need to shuffle the data while training.

Let's look at the histogram of training dataset.

![histogram](./examples/download4.png)


### Design and Test Model Architecture

#### 1. Data Preprocessing

As a first step, I decided to convert the images to grayscale because thats one of the starting points in image conversion. Since the brightness in images is a big factor, the grayscale image fails to capture features in some cases. 

Here is an example of a traffic sign image after grayscaling.

![grayscale](./examples/download5.png)

I read through a paper on AlexNet one of the first high accuracy classifiers for traffic signs, and found a contrast maximization technique using skimage.exposure 's equalize_adapthist works quite well on these forms of low contrast images. 

Here is an example of a traffic sign image after applying the exposure function on Y-channel of YCrCb converted image.

![equalized](./examples/download6.png)

As a last step, I normalized the image data because normalized values having a wider distribution in the data would make it more difficult to train using a singlar learning rate. Different features could encompass far different ranges and a single learning rate might make some weights diverge. Normalization technique used was (image pixel - min image pixel)/(max image pixel - min image pixel). 

---

I decided to generate additional training data because the training set is very skewed and has unequal number of samples of each class which would lead to better prediction on some and not so good ones on others. I also thought the training size of each class is less and maybe 3000 images per class will be a good number to work with.   

To add more data to the the data set, I used the following techniques:

1. Scaling : zoom in/ zoom out an image randomly
2. Translating : shifting the image a bit in any direction randomly
3. Rotating : Rotate the image in a range of 30 degrees right/left from original image randomly
4. Warping : Similar to what a perspective warp does to images, but randomly
5. Brightening : Increasing or decreasing the lumosity of an image randomly

Here is how eqach operation applied individually on a image looks like.

![Augmentation](./examples/download7.png)

I created specialized function for each of the above tasks and then randomly chose any number of operations from above to be applied on each image to increase the randomness in the new data generated. Like say, one image would be first rotated by 15 degrees to right then warped and then brightened to create a new image. Using this technique i created a new training dataset of 3000 images per class. Here's what the histogram of new training set looks like :

![new training set](./examples/download8.png)

And this is what first 15 images of each class looks like after adding more data:

![15 images](./examples/download9.png)

Finally the contrast maximization and normalization was done for all the datasets to generate a final training , validation and testing set.

Here is an example of how augmented images of each class look like:

![final set](./examples/download10.png)

The difference between the original data set and the augmented data set is the following ... 

1. They are normalized, single channel 32,32,1 images
2. There are 3000 images per class for training set


#### 2. Model Architecture

I have tested 2 type of CNN models for my project. First one is inspired by Alexnet structure with slight modifications, second one is a inception module. However, my inception module doesnot give better results than first structure so I used the first one for rest of the predictions. My first model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized image   		    		| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3       | 1x1 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Avg pooling(flatten)  | 8x8 stride, outputs 4x4x32                    |
| Max pooling	      	| 2x2 stride, outputs 16x16x32 		     		|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64    |
| RELU                  |                                               |
| Avg pooling(flatten)  | 4x4 stride, outputs 4x4x64                    |
| Max pooling           | 2x2 stride, output 8x8x64                     |
| Convolution 3x3       | 1x1 stride, same padding, outputs 8x8x128     |
| RELU                  |                                               |
| Max pooling(flatten)  | 2x2 stride, output 4x4x128                    |
| Flat layer            | Add all flattened layers 3584 inputs          |
| Hidden layer          | 3584x1024    									|
| RELU                  |                                               |
| Hidden layer          | 1024x128    									|
| RELU                  |                                               |
| Hidden layer          | 128xlogits    								|
| One hot encoder		|           									|
| Softmax cross entropy	| Reduce mean loss								|
| Optimizer      		| Adam Optimizer								|
 
My second network looks like the following:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized image   		    		| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max Pooling           | 2x2 stride, outputs 16x16x32                  |
| ParallelConv2 3x3     | 1x1 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Max Pooling           | 2x2 stride, outputs 16x16x32                  |
| ParallelConv3 5x5     | 1x1 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Max Pooling           | 2x2 stride, outputs 16x16x32                  |
| Stacking parallels    | outputs 16x16x96                              |
| Convolution 2x2       | 1x1 stride, same padding, outputs 16x16x96    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 8x8x96                    |
| Flatten 8x8x96        | outputs 6144                                  |
| Avg pooling(layer1)   | inputs 32x32x32, 4x4 stride, outputs 8x8x32   |
| Convolution 1x1       | 1x1 stride, same padding, outputs 8x8x32      |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride, outputs 4x4x32 		     		|
| Flatten 4x4x32        | outputs 512                                   |
| Add flat layers       | outputs 6656                                  |
| Hidden layer          | 6656x1024    									|
| RELU                  |                                               |
| Hidden layer          | 1024x128    									|
| RELU                  |                                               |
| Hidden layer          | 128xlogits    								|
| One hot encoder		|           									|
| Softmax cross entropy	| Reduce mean loss								|
| Optimizer      		| Adam Optimizer								|
 

#### 3. Model Training

To train the first model (named it SAMnet), I used 

1. Adam optimizer
2. Batch size = 64
3. Epochs = 40
4. Learning rate = 0.0005
5. Dropout probability 0.5 for convolution layers and 0.7 for flat hidden layers
6. Initialization mean = 0 and deviation = 0.1

To train the second model (named it SaMnet), I used 

1. Adam optimizer
2. Batch size = 64
3. Epochs = 30
4. Learning rate = 0.0005
5. Dropout probability 0.5 for convolution layers and 0.7 for flat hidden layers
6. Initialization mean = 0 and deviation = 0.1

#### 4. Final Results

My final model results on SAMnet were:
* training set accuracy of 1
* validation set accuracy of 0.995 
* test set accuracy of **0.971 = 97.1%** 

My final model results on SaMnet were:
* training set accuracy of 1
* validation set accuracy of 0.990 
* test set accuracy of 0.967

I chose SAMnet architecture as a small variation of famous Alexnet model and it seemed to perform fra better than the inception module structure which was completely designed by me. Alexnet has been known to show great results on traffic sign classification for long and the final model accuracies keep fluctuating around the same 0.99 for traning and validation to 0.97 on test set. Since repititive iterations give same results and performs really well on new images it had led me to believe that the model is working well.

Since SAMnet gave better accuracy on test set, hereby all the testing is done using that model and not the inception model.

### Testing Model on New Images

#### 1. Five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![alt text](./examples/Stop.jpg) ![alt text](./examples/Speed_70.jpg) ![alt text](./examples/Road_Work.jpg) 
![alt text](./examples/turn_right.jpg) ![alt text](./examples/Do_Not_Enter.jpg)

The first image is easy to classify because its same as other training set stop signs. 

The second image may get confused with other speed limit signs.

The third image is also easy to classify.

Fourth image might be a bit difficult because it looks different from training set images for turn right 

Fifth image can be confused with no entry and no passing.

#### 2. Model predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Turn Right     		| Turn Right									|
| Road Work				| Road Work										|
| 70 km/h	      		| 70 km/h					 		     		|
| Do not enter  		| Do not enter      							|

Here is the image of predictions made based on images sent through the model.

![predictions](./examples/download11.png)

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.1%

#### 3. Model Certainity

Here is the image which shows the top 5 softmax probabilities of new images. 

![softmaxes](./examples/download12.png)

