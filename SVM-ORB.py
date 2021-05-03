#%% import libraries 
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os 
from skimage.feature import ORB
from skimage import io,color
#%%load data 

train_data_path = "simages"
img_list = "Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/Small_Data/simages"
base_path = os.getcwd() #get working directory
img_dir_list  =os.listdir(os.path.join(base_path,img_list)) #image listss

# train_datagen = ImageDataGenerator(rescale=1/255.0)
# #test_datagen = ImageDataGenerator(rescale=1/255.0)

# train_generator = train_datagen.flow_from_directory(
#         train_data_path,
#         batch_size=10,
#         color_mode = 'grayscale'     
#         )
# print(train_generator)

#%% testing
# for i in train_generator:
#     x = i
#     break



# =============================================================================

def keypoint_extractor(img_dir_list):
  # orb = cv2.ORB_create(nfeatures=100)
  orb = ORB(n_keypoints=100)
  descriptors = []
  for index,i in enumerate(img_dir_list):
    img_name = os.path.join(base_path,img_list + "\\" + i)
    # image = cv2.imread(img_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = io.imread(img_name)
    image = color.rgb2gray(image)
    # train_keypoints, train_descriptor = orb.detectAndCompute(image , None)
    orb.detect_and_extract(image)
    descriptors.append(orb.descriptors)
    print(index)
  return descriptors


#%% 

descriptors = keypoint_extractor(img_dir_list)

    


print(len(descriptors))

# =============================================================================

feature_desc = np.concatenate((descriptors), axis=0)*1.0

# =============================================================================
# Dimensional Reduction - PCA 
# =============================================================================

from sklearn.decomposition import PCA
#from sklearn.decomposition import IncrementalPCA 

# feature_desc = (feature_desc - feature_desc.mean(axis=0))/feature_desc.std(axis=0)
pca = PCA(n_components=0.8)
pca.fit(feature_desc)
feature_desc_red = pca.transform(feature_desc)

# =============================================================================
# K-Means
# =============================================================================
from sklearn.cluster import KMeans

inertias = []
for i in range(2,20): 
  kn = KMeans(n_clusters=i)
  kn.fit(feature_desc_red)
  inertias.append(kn.inertia_)

#%%plot of KMeans 
import matplotlib.pyplot as plt
plt.figure(dpi=200)
plt.plot(inertias, marker='o', markersize=3)
plt.title('k-means inertia plot')
plt.xlabel('number of clusters')
plt.ylabel('inertia score')
plt.xticks(ticks=range(len(inertias)))
plt.grid(True)
plt.show()
#%% setting k-means to 3 clusters 

kn = KMeans(n_clusters=3)
kn.fit(feature_desc_red)
labels = kn.labels_


#%% plot for occurance of clusters 
plt.figure(dpi=200)
plt.hist(labels, color='red')
plt.xticks(ticks=[0,1,2])
plt.grid(True)
plt.title('histogram of cluster occurance')
plt.xlabel('clusters')
plt.ylabel('frequency')
plt.show()


# =============================================================================
# 
# =============================================================================


labels_count = np.asarray([(labels == i).sum() for i in np.unique(labels)])
prob_occurance = labels_count/labels_count.sum()



# =============================================================================
# SVM classifer 
# =============================================================================
from sklearn.svm import SVC

svc = SVC(C = 1)
svc.fit(feature_desc, labels)

#%% r2score

from sklearn.metrics import r2_score

#%%

predictions = svc.score(feature_desc, labels)
y_pred = svc.predict(feature_desc)
R2score = r2_score(labels, y_pred)



#%%5 Fold Cross Validation 

from sklearn.model_selection import cross_val_score

crossval = cross_val_score(svc, feature_desc, labels, cv = 5, scoring = 'r2')
crossval.mean()

# =============================================================================
# Testing Stage 
# =============================================================================


img_list = ['normal.jpeg']


# def keypoint_extractor(img_list, length_of_list):
#   orb = cv2.ORB_create(nfeatures=500)
#   descriptors = []
#   for i in range(length_of_list):
#     image = cv2.imread(img_list[i])
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     train_keypoints, train_descriptor = orb.detectAndCompute(image, None)
#     descriptors.append(train_descriptor)
#   return descriptors, train_keypoints 

def keypoint_extractor_test(img_dir_list):
  # orb = cv2.ORB_create(nfeatures=100)
  orb = ORB(n_keypoints=100)
  descriptors = []
  keypoints = []
  for i in img_dir_list:
    # image = cv2.imread(img_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = io.imread(i)
    image = color.rgb2gray(image)
    # train_keypoints, train_descriptor = orb.detectAndCompute(image , None)
    orb.detect_and_extract(image)
    descriptors.append(orb.descriptors)
    keypoints.append(orb.keypoints)
   
  return descriptors,keypoints


descriptors_test,keypoints_test = keypoint_extractor_test(img_list)

feature_desc_tst = np.concatenate((descriptors_test), axis=0) * 1
test_pred = svc.predict(feature_desc_tst)
test_condition = (test_pred == 1).any()
test_condition


abnormal_pts = keypoints_test[0][test_pred == 2]
normal_pts = keypoints_test[0][test_pred == 1]
both_pts = keypoints_test[0][test_pred == 0]

image_mat = image = io.imread('normal.jpeg')
image_mat = color.rgb2gray(image_mat)

from matplotlib.patches import Circle
fig = plt.figure(dpi=150)
ax = fig.subplots(1)
ax.set_aspect('equal')
ax.imshow(image_mat, cmap="gray")
ax.set_title("normal image")
ax.axis("off")
r = 12
for x,y in abnormal_pts:
    circ = Circle((y,x),r, color="red", fill=False, lw=0.5)
    ax.add_patch(circ)
    
for x,y in normal_pts:
    circ = Circle((y,x),r, color="green", fill=False, lw=0.5)
    ax.add_patch(circ)
    
for x,y in both_pts:
    circ = Circle((y,x),r, color="blue", fill=False, lw=0.5)
    ax.add_patch(circ)

plt.show()









