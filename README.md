# Feature extraction techniques for localization of region of interest of abnormalities in biomedical images
## Final_Project - Pattern_Recognition (SYDE 675)

### Problem Statement 

In the domain of medical diagnostics, annotating medical image data with useful information is an arduous process that requires medical expertise and specific domain knowledge. Faced with the challenge of medical images without consistent annotations, unsupervised learning presents the means for feature extraction and localization of regions of interest that will assist in the retrieval of key information of biomedical concepts. We compare automated methods regarding reliability and efficiency in detecting abnormalities, and whether they can ultimately support decision-making in clinical treatment.


### Summary 

The chosen task is classifying signs of pneumonia in X-ray chest image data. X-rays are one of the most effective methods to diagnose pneumonia. Pneumonia causes a staggering number of mortalities every year across the world.

We address this challenge by implementing and evaluating three different unsupervised feature learning approaches. One of these approaches relies on a key-point extraction and description algorithm, such as ORB, to train a support vector machine (SVM). ORB was accompanied by PCA for dmensionality reduction while retaining existing information. Since no ground truth labels of the data are known, we use k-means algorithm for the formation of precise clusters to increase accuracy. With the possible domain knowledge of normal, abnormal and mutual features present in both samples we chose a partition of k=3 clusters. Then we fit the SVM classifier for cluster prediction. Finally, in the testing stage we manually test normal and abnormal images where orb extractor obtains the descriptors which is then fed into the SVM for end-to-end feature learning.

### Results 

For visualization (figures listed below) in the final step we draw circles around regions of interest in the image where green indicate normal, red indicate abnormal and blue indicate the features belonging to both classes. 

![normal_image](https://user-images.githubusercontent.com/38030229/117044782-d322fc00-acdc-11eb-99e8-1b6e647e9b1c.png)
![abnormal_image](https://user-images.githubusercontent.com/38030229/117044790-d5855600-acdc-11eb-8acb-b92639f9a432.png)

### Data Availability 

The CoronaHack-Chest X-Ray-Dataset used in this paper is publicly available and downloadable from the following link: https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset 

