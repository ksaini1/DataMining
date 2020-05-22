# -*- coding: utf-8 -*-
"""Assignment2(final).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YaDRGk8IDqJLfMeTGe1WLd_zST5CFE4Y
"""

import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import pandas as pd
import math
import scipy.linalg as la
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt




np.random.seed(1)



data11= pd.read_csv("water-treatment.data",header=None)
#print data11
data11.drop(columns=0, axis=1, inplace=True)
data11=data11.T.reset_index(drop=True).T
print ("\n\n")
#print data11
def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

#NAN
data11=data11[data11.applymap(isnumber)]
data11=data11.fillna(data11.mean())
#data fill
data11 = data11.fillna(method='bfill')
data11 = data11.fillna(method='ffill')
#print data11

for i in range(38):
    data11.iloc[:,i-1]  = data11.iloc[:,i-1].astype(float).fillna(0.0)


#print data11

#normalization


#print type(data12.iloc[:,0])
#for i in range(38):
#    data11.iloc[:,i-1]  = data11.iloc[:,i-1].astype(float).fillna(0.0)



data11.iloc[:,0] = data11.iloc[:,0].apply(lambda x: (x - 37226.56 ) / (60081 - 10000 ))
data11.iloc[:,1] = data11.iloc[:,1].apply(lambda x: (x - 2.36 ) / (33.5 - 0.1 ))
data11.iloc[:,2] = data11.iloc[:,2].apply(lambda x: (x - 7.81 ) / (8.7 - 6.9 ))
data11.iloc[:,3] = data11.iloc[:,3].apply(lambda x: (x - 188.71 ) / (438 - 38 ))
data11.iloc[:,4] = data11.iloc[:,4].apply(lambda x: (x - 406.89 ) / (941 - 81 ))
data11.iloc[:,5] = data11.iloc[:,5].apply(lambda x: (x - 227.44 ) / (2008 - 98 ))
data11.iloc[:,6] = data11.iloc[:,6].apply(lambda x: (x - 61.39 ) / (85 - 13.2 ))
data11.iloc[:,7] = data11.iloc[:,7].apply(lambda x: (x - 4.59 ) / (36 - 0.4 ))
data11.iloc[:,8] = data11.iloc[:,8].apply(lambda x: (x - 1478.62 ) / (3230 - 651 ))
data11.iloc[:,9] = data11.iloc[:,9].apply(lambda x: (x - 7.83 ) / (8.5 - 7.3 ))
data11.iloc[:,10] = data11.iloc[:,10].apply(lambda x: (x - 206.20 ) / (517 - 32 ))
data11.iloc[:,11] = data11.iloc[:,11].apply(lambda x: (x - 253.95 ) / (1692 - 104 ))
data11.iloc[:,12] = data11.iloc[:,12].apply(lambda x: (x - 60.37 ) / (93.5 - 7.1 ))
data11.iloc[:,13] = data11.iloc[:,13].apply(lambda x: (x - 5.03 ) / (46 - 1 ))
data11.iloc[:,14] = data11.iloc[:,14].apply(lambda x: (x - 1496.03 ) / (3170 - 646 ))
data11.iloc[:,15] = data11.iloc[:,15].apply(lambda x: (x - 7.81 ) / (8.4 - 7.1 ))
data11.iloc[:,16] = data11.iloc[:,16].apply(lambda x: (x - 122.34 ) / (285 - 26 ))
data11.iloc[:,17] = data11.iloc[:,17].apply(lambda x: (x - 274.04 ) / (511 - 80 ))
data11.iloc[:,18] = data11.iloc[:,18].apply(lambda x: (x - 94.22 ) / (244 - 49 ))
data11.iloc[:,19] = data11.iloc[:,19].apply(lambda x: (x - 72.96 ) / (100 - 20.2 ))
data11.iloc[:,20] = data11.iloc[:,20].apply(lambda x: (x - 0.41 ) / (3.5 - 0.0 ))
data11.iloc[:,21] = data11.iloc[:,21].apply(lambda x: (x - 1490.56 ) / (3690 - 85 ))
data11.iloc[:,22] = data11.iloc[:,22].apply(lambda x: (x - 7.70 ) / (9.7 - 7.0 ))
data11.iloc[:,23] = data11.iloc[:,23].apply(lambda x: (x - 19.98 ) / (320 - 3 ))
data11.iloc[:,24] = data11.iloc[:,24].apply(lambda x: (x - 87.29 ) / (350 - 9 ))
data11.iloc[:,25] = data11.iloc[:,25].apply(lambda x: (x - 22.23 ) / (238 - 6 ))
data11.iloc[:,26] = data11.iloc[:,26].apply(lambda x: (x - 80.15 ) / (100 - 29.2 ))
data11.iloc[:,27] = data11.iloc[:,27].apply(lambda x: (x - 0.03 ) / (3.5 - 0.0 ))
data11.iloc[:,28] = data11.iloc[:,28].apply(lambda x: (x - 1494.81 ) / (3950 - 683 ))
data11.iloc[:,29] = data11.iloc[:,29].apply(lambda x: (x - 39.08 ) / (79.1 - 0.6 ))
data11.iloc[:,30] = data11.iloc[:,30].apply(lambda x: (x - 58.51 ) / (96.1 - 5.3 ))
data11.iloc[:,31] = data11.iloc[:,31].apply(lambda x: (x - 90.55 ) / (100 - 7.7 ))
data11.iloc[:,32] = data11.iloc[:,32].apply(lambda x: (x - 83.44 ) / (94.7 - 8.2 ))
data11.iloc[:,33] = data11.iloc[:,33].apply(lambda x: (x - 67.67 ) / (96.8 - 1.4 ))
data11.iloc[:,34] = data11.iloc[:,34].apply(lambda x: (x - 89.01 ) / (97 - 19.6 ))
data11.iloc[:,35] = data11.iloc[:,35].apply(lambda x: (x - 77.85 ) / (98.1 - 19.2 ))
data11.iloc[:,36] = data11.iloc[:,36].apply(lambda x: (x - 88.96 ) / (99.4 - 10.3 ))
data11.iloc[:,37] = data11.iloc[:,37].apply(lambda x: (x - 99.08 ) / (100 - 36.4 ))


for k in range (1, 11):

   # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
   kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data11.iloc[:, :])
   
   # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
   labels = kmeans_model.labels_

   # Sum of distances of samples to their closest cluster center
   interia = kmeans_model.inertia_
   print ("k:",k, " cost:", interia)

#print data11.iloc[:,0]

#distortions = []
#K = range(1,30)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k).fit(data11)
#    kmeanModel.fit(data11)
#    distortions.append(sum(np.min(cdist(data11, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data11.shape[0])
#
## Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()


Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(data11).score(data11) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


newk=KMeans(n_clusters=2)
newk=newk.fit(data11)
y_kmeans = newk.predict(data11)
plt.scatter(data11.iloc[:, 0], data11.iloc[:, 24], c=y_kmeans, s=50)

plt.show()


#PCA



pca=PCA(n_components=12)
red=pca.fit_transform(data11)
pca1 = pd.DataFrame(red)


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
#print pca.explained_variance_ratio_

#pca = PCA(n_components=5)
#data12 = pca.fit(data11)
#data13 = pd.DataFrame(data12)
print (pca1)
for i in range(12):
    pca1.iloc[:,i-1]  = pca1.iloc[:,i-1].astype(float).fillna(0.0)

print (pca1)
print (type(pca1.iloc[:,0]))


for k in range (1, 11):

   # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
   kmeans_l = KMeans(n_clusters=k, random_state=1).fit(pca1.iloc[:, :])
   
   # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
   labels = kmeans_l

   # Sum of distances of samples to their closest cluster center
   interia = kmeans_l.inertia_
   print ("k:",k, " cost:", interia)
   
   
#PCA kmeans
distortion = []
K = range(1,10)
for k in K:
    kmean = KMeans(n_clusters=k).fit(pca1)
    kmean.fit(pca1)
    distortion.append(sum(np.min(cdist(pca1, kmean.cluster_centers_, 'euclidean'), axis=1)) / pca1.shape[0])

# Plot the elbow
plt.plot(K, distortion, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

nk=KMeans(n_clusters=2)
nk.fit(pca1)
y_k = nk.predict(pca1)


plt.scatter(pca1.iloc[:,0], pca1.iloc[:, 4], c=y_k, s=50)

plt.show()


clus = pd.DataFrame(nk.labels_)
print (clus)
clus.to_csv('clust.txt', sep=' ' )


encoding_dim = 2

input_df = Input(shape=(38,))
encoded = Dense(encoding_dim, activation='relu')(input_df)
decoded = Dense(38, activation='sigmoid')(encoded)

# encoder
autoencoder = Model(input_df, decoded)

# intermediate result
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(data11, data11,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(data11, data11))

encoder_input=Input(shape=(encoding_dim, ))
encoder_output= encoder.predict(data11)

print (encoder_output)
autoencoder=pd.DataFrame(encoder_output)

plt.scatter(autoencoder.iloc[:, 0], autoencoder.iloc[:, 1], s=30)
plt.show()