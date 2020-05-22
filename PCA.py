import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import pandas as pd
import math
from sklearn.decomposition import PCA
import scipy.linalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.preprocessing import Imputer
from sklearn.datasets import  load_digits



data11= pd.read_csv("dist1_500_1.txt",sep=" ",header=None)
data12= pd.read_csv("dist1_500_1.txt",sep=" ", header =None)
data21=pd.read_csv("dist2_500_1.txt",sep=" ",header=None)
data22=pd.read_csv("dist2_500_2.txt",sep=" ",header=None)
data11=data11.dropna()
data12=data12.dropna()
data21=data21.dropna()
data22=data22.dropna()
data11=data11.append(data12)
data21=data21.append(data22)
f1 = []
f2 = []

df=pd.DataFrame(data11)
dataset1 = []
dataset1=df.values.tolist()

df=pd.DataFrame(data21)
dataset2 = []
dataset2=df.values.tolist()

datarandom1 = []
for i in range(10):
    j=random.randint(0,999)
    datarandom1.append(dataset1[j])

temp = []
print "\nThe 10 random data samples from data set 1 are: "
print datarandom1

plt.boxplot(datarandom1)
plt.title("DATASET 1 (10 SAMPLES)")
plt.show()

plt.hist(datarandom1)
plt.title("DATASET 1 HISTOGRAM")
plt.show()

for small_list in datarandom1:
    temp+=small_list

datarandom1=temp
datarandom2 = []

for i in range(10):
    j=random.randint(0,999)
    datarandom2.append(dataset2[j])

print  "\n The 10 random data samples from data set 2 are :  "
print datarandom2

plt.boxplot(datarandom2)
plt.title("DATASET 2 BOXPLOT")
plt.show()

plt.hist(datarandom2)
plt.title("DATASET 2 HISTOGRAM")
plt.show()

temp1 = []
for small_list in datarandom2:
    temp1+=small_list
datarandom2=temp1

keys = []
values = []
freq = {}

def countfreq(datarandom1):

    for item in datarandom1:
        if(item in freq):
            freq[item]+=1
        else:
            freq[item]=1
    for key,value in freq.items():
        #print (" % f : % d "%(key,value))
        f1.append((key,value))
 #   print "function"


countfreq(datarandom1)
#print "dict 1 "
#print freq
dictlistkey = []
dictlistvalue = []
for key,value in freq.iteritems():
    tempk = [key]
    tempv = [value]
    dictlistkey.append(tempk)
    dictlistvalue.append(tempv)

dictlist = []


for key, value in freq.iteritems():
    temp = [key,value]
    dictlist.append(temp)

x , y =zip(*dictlist)

freq2 = {}
keys2 = []
values2 = []

def countfreqs(datarandom2):
    for item in datarandom2:
        if(item in freq2):
            freq2[item]+=1
        else:
            freq2[item]=1
    for key2,value2 in freq2.items():
        f2.append((key2,value2))


dictlistkey2 = []
dictlistvalue2 = []
for key,value in freq2.iteritems():
    tempk2 = [key]
    tempv2 = [value]
    dictlistkey2.append(tempk2)
    dictlistvalue2.append(tempv2)
countfreqs(datarandom2)
dictlist2 = []
for key, value in freq2.iteritems():
    temp = [key,value]
    dictlist2.append(temp)


f1.sort(reverse=True)
f2.sort(reverse=True)
print "\nFrequency List for dataset 1 "
print dictlist

print "\nFrequency List for dataset 2 "
print dictlist2


x1 , y1 =zip(*f1)
x2 , y2=zip(*f2)

plt.plot(x1,y1)
plt.title("DATASET 1")
plt.show()

plt.plot(x2,y2)
plt.title('DATASET 2')
plt.show()

plt.hist(datarandom1)
plt.title("DATASET 1 HISTOGRAM COMPLETE 10 SAMPLES TOGETHER")
plt.show()

plt.hist(datarandom2)
plt.title("DATASET 2 HISTOGRAM COMPLETE 10 SAMPLES TOGETHER")
plt.show()

plt.boxplot(datarandom1)
plt.title("BOXPLOT FOR 10 SAMPLES DATA TOGETHER")
plt.show()

plt.boxplot(datarandom2)
plt.title("BOXPLOT FOR 10 SAMPLES DATA TOGETHER")
plt.show()

X = StandardScaler().fit_transform(data11)
X2 = StandardScaler().fit_transform(data21)

c1 = np.cov(X.T)
eigenvalues1, eigenvector1 = np.linalg.eig(c1)

c2 = np.cov(X2.T)
eigenvalues2, eigenvector2 = np.linalg.eig(c2)
print "\nEigenvalues for data set 1:"
print eigenvalues1

print "\nEigenvalues for data set 2:"
print eigenvalues2

pairs1= [(np.abs(eigenvalues1[i]), eigenvector1[:, i]) for i  in range(len(eigenvalues1))]
pairs1.sort()
pairs1.reverse()

pairs2 = [(np.abs(eigenvalues2[i]), eigenvector2[:, i]) for i in range(len(eigenvalues2))]
pairs2.sort()
pairs2.reverse()

total1 = sum(eigenvalues1)
total2 = sum(eigenvalues2)

variance1 = [(i/total1)*100 for i in sorted(eigenvalues1, reverse=True)]
variance2 = [(i/total2)*100 for i in sorted(eigenvalues2, reverse=True)]

cum1 = np.cumsum(variance1)
cum2 = np.cumsum(variance2)

plt.bar(range(100), variance1, alpha=0.5, align='center')
plt.title("DATASET  PCA")
plt.xlabel("Principal component")
plt.ylabel("Contribution")
plt.show()

plt.bar(range(100), variance2, alpha=0.5, align='center')
plt.title("DATASET 2 PCA")
plt.xlabel("Principal component")
plt.ylabel("Contribution")
plt.show()

pca=PCA(n_components=90)
red=pca.fit_transform(X)
j=1
for i in range(0,88):
    plt.scatter(red[:,i],red[:,j])
    j=j+1
plt.scatter(red[:,89],red[:,0])
plt.title("SCATTERED FOR DATASET 1 PCA")
plt.show()

pca=PCA(n_components=90)
red=pca.fit_transform(X2)
j=1
for i in range(0,88):
    plt.scatter(red[:,i],red[:,j])
    j=j+1
plt.scatter(red[:,89],red[:,0])
plt.title("SCATTERED FOR DATASET 2 PCA")
plt.show()

emc2_image = X
ica = FastICA(n_components= 90)
ica.fit(X)
emc2_image_ica = ica.fit_transform(emc2_image)
j=1
for i in range(0,88):
    plt.scatter(emc2_image_ica[:,i],emc2_image_ica[:,j])
    j=j+1
plt.scatter(emc2_image_ica[:,89],emc2_image_ica[:,0])
plt.title("ICA FOR DATASET 1")
plt.show()

emc2_image2 = X2
ica2 = FastICA(n_components= 90)
ica2.fit(X2)
emc2_image_ica2 = ica2.fit_transform(emc2_image2)
j=1
for i in range(0,88):
    plt.scatter(emc2_image_ica2[:,i],emc2_image_ica2[:,j])
    j=j+1
plt.scatter(emc2_image_ica2[:,89],emc2_image_ica2[:,0])
plt.title("ICA FOR DATASET 2")
plt.show()

data11 = data11.values
data11 = data11.T
dct1 = np.zeros([100,1000])
n = 100
counti=0
pie = math.pi
for i in range(0,99):
    for j in range(0,999):
        x=data11[i][j]
        if(counti == 0):
            root = math.sqrt(1/n)
            cost = ((2*x)+1)*counti*pie
            dct1[i][j] = (root * math.cos( cost / (2*n)) )

        else:

            root = math.sqrt(1 / 100)
            cost = ((2 * x) + 1) * counti * pie
            dct1[i][j] = (0.01 * math.cos(cost / (2 * n)))

        counti+= 1

data21 = data21.values
data21 = data21.T
dct2 = np.zeros([100,1000])
n = 100
counti=0
pie = math.pi
for i in range(0,99):
    for j in range(0,999):
        x=data21[i][j]
        if(counti == 0):
            root = math.sqrt(1/n)
            cost = ((2*x)+1)*counti*pie
            dct2[i][j] = (root * math.cos( cost / (2*n)) )

        else:

            root = math.sqrt(1 / 100)
            cost = ((2 * x) + 1) * counti * pie
            dct2[i][j] = (0.01 * math.cos(cost / (2 * n)))

        counti+= 1



dct1=dct1.tolist()
dct2=dct2.tolist()
plt.hist(dct1)
plt.title("DCT FOR DATASET 1")
plt.show()

plt.hist(dct2)
plt.title("DCT FOR DATASET 2")
plt.show()

print "\nDCT VALUES FOR DATASET 1"
print dct1

print "\nDCT VALUES FOR DATASET 2"
print dct2

