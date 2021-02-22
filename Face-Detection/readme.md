#Face Detection using Viola-Jones Object Detection Framework

#Overview

In 2001, Paul Viola and Michael Jones developed an object detection framework that provided object detection in real time.
It has 4 main steps:

1. Selecting a Haar-like feature
2. Creating an integral image
3. Running Adaptibe Boosting training
4. Creating classifier cascades

The algorithm divides the image into many smaller regions and tries to find a face by looking for specific features in each subregion.

#Dependencies

```
pip install numpy
pip install opencv-python
pip install scikit-image
pip install scikit-learn
```

#Haar-Like Features

It is based on the fact that all human faces share some similarities, for example, the eye region of a person is darker than the bridge of the nose and the cheeks are brighter than the eye region. To find which region is lighter or darker we sum up the values of pixel in both the regions and compare them.

A Haar-like feature is represented by taking a rectangular part of an image and dividing that rectangle into multiple parts. The value of the feature is calculated as a single number which is the sum of pixel values in the black area minus the sum of pixel values in the white area.

Since this algorithm calculates these features in many subregions of an image this process quickly becomes computationally expensive, hence we use integral images.

#Integral Images

An integral image also known as a summed-area table is the name of both a data structure and an algorithm used to obtain this data structure. It is used as a quick and efficient way to calculate the sum of pixel values in an image or rectangular part of an image.

In an integral image, the value of each point is the sum of all pixels above and to the left including the target pixel. The integral image can be calculated in a single pass over the original image. This reduces summing the pixel intensities within a rectangle into only three operations with four numbers regardless the size of the rectangle.

#Adaptive Boosting

In the Viola-Jones algorithm, each Haar-like feature represents a weak learner. To decide the type and size of a feature that goes into the final classifier, AdaBoost checks the performance of all classifiers that we supply to it.

The classifiers that performed well are given higher importance or weight. The final result is a strong classifier, also called a boosted classifier, that contains the best performing weak classifiers.

It would still be computationally expensive to run all these classifiers on every region in every image, so we use cascading classifiers.

#Cascading Classifiers

In this process the strong classifier which is made up of thousands of weak classifiers is turned into a cascade where each weak classifier represents one stage. The job of this cascade is to quickly discard non-faces and avoid wasting precious time and computations on them.

To accomplish efficiency, it is important to put the best performing classifier early in the cascade. In this algorithm the eyes and the nose bridge classifiers are best performing weak classifiers.