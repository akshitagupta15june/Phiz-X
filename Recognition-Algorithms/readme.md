#Face Recognition using LBPH: Local Binary Patterns Histograms

#Overview

Local Binary Pattern (LBP) is a simple yet very efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number. Using the LBP combined with histograms we can represent the face images with a simple data vector. 

It has 4 main steps:

1. Collecting the Dataset
2. Applying the LBP Operations
3. Extracting the Histograms
4. Performing the Face Recognition

#Dependencies

```
pip install numpy
pip install opencv-python
pip install opencv-contrib-python
```

#Collecting the Dataset 

In this project, I coded so that my Laptop's camera get 400 images of my faces in the dataCollection.py file.
We also need to set an ID for each image, so the algorithm will use this information to recognize an input image and give us an output. 

#Applying the LBP Operations

The first step is to create an intermediate image that describes the original image in a better way, by highlighting the facial characteristics.

To do so, the algorithm uses a concept of a sliding window, which works as follows: 

1. Suppose we have a facial image in grayscale. We can get a part of this image as a window of 3x3 pixels. Then, we take the central value of the matrix to be used as the threshold. 

2. This value will be used to define the new values from the 8 neighbors. For each neighbor of the threshold, we set a new binary value. We set 1 for values equal or higher than the threshold and 0 for values lower than the threshold.

3. Now, the matrix will contain only binary values. We need to concatenate each binary value from each position from the matrix line by line into a new binary value (e.g. 10001101). Then, we convert this binary value to a decimal value and set it to the central value of the matrix, which is actually a pixel from the original image.

At the end of this procedure, we have a new image which represents better the characteristics of the original image.

#Extracting the Histograms

We extract the histogram of each region as follows:

1. As we have an image in grayscale, each histogram will contain only 256 positions representing the occurrences of each pixel intensity.
2. Then, we concatenate each histogram to create a new and bigger histogram. The final histogram represents the characteristics of the image original image.

#Performing the Face Recognition

Each histogram created is used to represent each image from the training dataset. So, given an input image, we perform the steps again for this new image and creates a histogram which represents the image.

So to find the image that matches the input image we just need to compare two histograms and return the image with the closest histogram.

