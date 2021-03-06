# About

Local Binary Pattern (LBP) is a simple yet very efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number. Using the LBP combined with histograms we can represent the face images with a simple data vector. 

It has 4 main steps:

1. Collecting the Dataset
2. Applying the LBP Operations
3. Extracting the Histograms
4. Performing the Face Recognition

# Dependencies

```
pip install numpy
pip install opencv-python
pip install sklearn
```

## Collecting the Dataset 

Right now I have not used some other data. I used the last face detecting code for collecting different samples and extraction of face. So, that I can only train the model on the basis of this dataset.

## LBP Operation

1. Suppose we have an image having dimentions N x M. We divide it into regions of same height and width resulting in m x m dimension for every region.
2. Local binary operator is used for every region. The LBP operator is defined in window of 3x3.
3. Using median pixel value as threshold, it compares a pixel to its 8 closest pixels using this function.
4. If the value of neighbor is greater than or equal to the central value it is set as 1 otherwise it is set as 0.
5. Thus, we obtain a total of 8 binary values from the 8 neighbors.
6. After combining these values we get a 8 bit binary number which is translated to decimal number for our convenience.
7. This decimal number is called the pixel LBP value and its range is 0-255.

## Extracting the histograms

We extract the histogram of each region as follows:

1. As we have an image in grayscale, each histogram will contain only 256 positions representing the occurrences of each pixel intensity.
2. Then, we concatenate each histogram to create a new and bigger histogram. The final histogram represents the characteristics of the image original image.

## Performing the Face Recognition

Each histogram created is used to represent each image from the training dataset. So, given an input image, we perform the steps again for this new image and creates a histogram which represents the image.

So to find the image that matches the input image we just need to compare two histograms and return the image with the closest histogram.

