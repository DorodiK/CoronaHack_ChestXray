# Feature extraction techniques for localization of region of interest of abnormalities in biomedical images
## Final_Project - Pattern_Recognition (SYDE 675)

### Problem Statement 

In the domain of medical diagnostics, annotating medical image data with useful information is an arduous process that requires medical expertise and specific domain knowledge. As such, labelled training data of high quality is not always available and acquiring the labels can be costly. In some cases, such as classifying rare diseases, sufficient training data might just not exist. Finally, training data with labels for new and unknown classes is always unavailable. Faced with the challenge of an exhaustive archive of medical images without consistent annotations, unsupervised learning presents the means for feature extraction and localization of regions of interest that will assist in the retrieval of key information of biomedical concepts. We compare automated methods regarding reliability and efficiency in detecting abnormalities, and whether they can ultimately support decision-making in clinical treatment.


### Summary 

The chosen task is classifying signs of pneumonia in X-ray chest image data. X-rays are one of the most effective methods to diagnose pneumonia. Pneumonia causes a staggering number of mortalities every year across the world. Classifying pneumonia using chest X-rays requires professionals like radiologists and such expert knowledge has limited availability.

We address this challenge by implementing and evaluating three different unsupervised feature learning approaches. One of these approaches relies on a key-point extraction and description algorithm, such as ORB, to train a support vector machine (SVM). The second one is a weakly supervised feature localization with a convolutional neural network (CNN) and finally, a U-Net based autoencoder network. The learned features are validated through qualitative feature analysis and trained by classifiers for biomedical concept detection.  Overall, all three methods address the challenge with some level of certainty. For realising the performance in a practical realm, these models can be evaluated through information retrieval systems in clinical setups. 

![normal_image](https://user-images.githubusercontent.com/38030229/117044782-d322fc00-acdc-11eb-99e8-1b6e647e9b1c.png)
![abnormal_image](https://user-images.githubusercontent.com/38030229/117044790-d5855600-acdc-11eb-8acb-b92639f9a432.png)

### Data Availability 

The CoronaHack-Chest X-Ray-Dataset used in this paper is publicly available and downloadable from the following link: https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset 

