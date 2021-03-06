# About:
To perform traditional face detection with OpenCV using Viola-Jones Object detection framework. We will have a pre-trained face detector model with the library.

# Dependencies: 

```
pip install numpy
pip install opencv-python
pip install skimage
```

# Viola-Jones Object detection framework:
This algorithm consists of 4 main steps:

    • Selecting Haar-like features.
    
    • Creating an integral image.
    
    • Running AdaBoost training.
    
    • Creating classifier cascades.
    
## 1). Haar-like features:
Human faces are made up of some brighter and darker regions. These 4 images can give a complete human face.

Example- Eye region in human face, Somewhere it matches with 2nd image. As human face contains brighter and darker region in horizontal form.


## 2). Integral Images :
An integral image (also known as a summed-area table) is the name of both a data structure and an algorithm used to obtain this data structure. It is used as a quick and efficient way to calculate the sum of pixel values in an image or rectangular part of an image.
## 3). Adaptive Boosting(AdaBoost):
In the Viola-Jones algorithm, each Haar-like feature represents a weak learner. To decide the type and size of a feature that goes into the final classifier, AdaBoost checks the performance of all classifiers that you supply to it.
The classifiers that performed well are given higher importance or weight. The final result is a strong classifier, also called a boosted classifier, that contains the best performing weak classifiers.
## 4). Cascading Classifiers:
We set up a cascaded system in which we divide the process of identifying a face into multiple stages. In the first stage, we have a classifier which is made up of our best features, in other words, in the first stage, the subregion passes through the best features such as the feature which identifies the nose bridge or the one that identifies the eyes. In the next stages, we have all the remaining features.


