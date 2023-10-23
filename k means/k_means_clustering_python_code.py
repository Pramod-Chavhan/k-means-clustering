
"""  #QUE 1.	Perform K means clustering on the airlines dataset to obtain optimum number of clusters
 Draw the inferences from the clusters obtained. Refer to EastWestAirlines.xlsx dataset."""

import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Visualization
import numpy as np                #for Mathematical calculations
import seaborn as sns             #for Advanced visualizations

air = pd.read_excel(r"D:\assignments\All datasets\Kmeans\EastWestAirlines.xlsx")

air

# We see the columns in the dataset
air.columns

# As a part of the Data cleansing we check the data for any missing/ na values
air.isna().sum()

#  check the data for any duplicate values
air1 = air.duplicated()
sum(air1)



# We now plot the boxplot for the data using each feature independently and check for Outliers
plt.boxplot(air.Balance);plt.title('Boxplot');plt.show()

# We see that there are Outliers present for "Balance" Feature

plt.boxplot(air.Qual_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.cc1_miles);plt.title('Boxplot');plt.show()  # No outliers

plt.boxplot(air.cc2_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.cc3_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Bonus_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Bonus_trans);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Flight_miles_12mo);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Flight_trans_12);plt.title('Boxplot');plt.show()  # outliers present


from scipy.stats.mstats import winsorize

air['Balance']=winsorize(air.Balance,limits=[0.07, 0.093])   
plt.boxplot(air['Balance']);plt.title('Boxplot');plt.show()

air['Qual_miles']=winsorize(air.Qual_miles,limits=[0.06, 0.094])   
plt.boxplot(air['Qual_miles']);plt.title('Boxplot');plt.show()

air['cc2_miles']=winsorize(air.cc2_miles,limits=[0.02, 0.098])   
plt.boxplot(air['cc2_miles']);plt.title('Boxplot');plt.show()

air['cc3_miles']=winsorize(air.cc3_miles,limits=[0.01, 0.099])   
plt.boxplot(air['cc3_miles']);plt.title('Boxplot');plt.show()

air['Bonus_miles']=winsorize(air.Bonus_miles,limits=[0.08, 0.092])   
plt.boxplot(air['Bonus_miles']);plt.title('Boxplot');plt.show()

air['Bonus_trans']=winsorize(air.Bonus_trans,limits=[0.01, 0.099])   
plt.boxplot(air['Bonus_trans']);plt.title('Boxplot');plt.show()

air['Flight_miles_12mo']=winsorize(air.Flight_miles_12mo,limits=[0.15, 0.85])   
plt.boxplot(air['Flight_miles_12mo']);plt.title('Boxplot');plt.show()

air['Flight_trans_12']=winsorize(air.Flight_trans_12,limits=[0.15, 0.85])   
plt.boxplot(air['Flight_trans_12']);plt.title('Boxplot');plt.show()

# Now we check the data for zero variance values
(air == 0).all()

# We drop the features that have zero variance values
air1 = air
air1.drop(["Qual_miles","Flight_miles_12mo","Flight_trans_12"], axis=1,inplace = True)
air1.columns

# We see the data again now to check whether the data is in scale
air1.describe

# we notice that the data needs to be normalise, using normalization

from sklearn import preprocessing   #package for normalize
air_normalized = preprocessing.normalize(air1)
print(air_normalized)

##########################Univariate, Bivariate################
plt.hist(air1["Balance"])   #Univariate

plt.hist(air1["Days_since_enroll"])

plt.scatter(air1["Balance"], air["Days_since_enroll"]);plt.xlabel('Days_since_enroll');plt.ylabel('Balance')   #Bivariate

air1.skew(axis = 0, skipna = True) 

air1.kurtosis(axis = 0, skipna = True)


# calculating TWSS - Total within SS using different cluster range
from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(air_normalized)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Plotting the Scree plot using the TWSS from above defined function
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters, 
# as the curve is seemingly bent or showinf an elbow format at K = 4

model = KMeans(n_clusters = 4)
model.fit(air_normalized)

model.labels_ # getting the labels of clusters assigned to each row

mb = pd.Series(model.labels_)  # converting numpy array into pandas series object

air1 = pd.read_excel("EastWestAirlines.xlsx","data")
air1['clust'] = mb # creating a  new column and assigning it to new column

air1.head()

air = air1.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
air.head()

# We can clearly see that we have the labels in the dataset in the form of a column called "clust", symbolizing the clusters

# In order to see the clusters we aggregate the records within the clusters and group them by the clusters to visualize the 
# 4 nos of clear cluster formed
air.iloc[:, 1:12].groupby(air.clust).mean()



"""#  QUE 2. Perform clustering for the crime data and identify the number of clusterformed and draw inferences.
 Refer to crime_data.csv dataset."""
 
 
import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Visualization
import numpy as np                #for Mathematical calculations
import seaborn as sns             #for Advanced visualizations

crime = pd.read_csv(r"D:\assignments\All datasets\Kmeans\crime_data.csv")

crime.head()

# We see the columns in the dataset
crime['State'] = crime.iloc[:,0]
crime = crime.iloc[:, [5,1,2,3,4]]

crime.head()



# we check the data for any missing/ na values
crime.isna().sum()

#  we check the data for any duplicate values
crime1 = crime.duplicated()
sum(crime1)


# We now plot the boxplot for the data using each feature independently and check for Outliers
plt.boxplot(crime.Murder);plt.title('Boxplot');plt.show()

# We see that there are Outliers present for "Balance" Feature

plt.boxplot(crime.Assault);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(crime.UrbanPop);plt.title('Boxplot');plt.show()  # No outliers

plt.boxplot(crime.Rape);plt.title('Boxplot');plt.show()  # outliers present



from scipy.stats.mstats import winsorize

crime['Rape'] = winsorize(crime.Rape, limits=[0.07, 0.093])   
plt.boxplot(crime['Rape']);plt.title('Boxplot');plt.show()

# Now we check the data for zero variance values
(crime == 0).all()

# We see the data again now to check whether the data is in scale
crime.describe

# we notice that the data needs to be normalise, using normalization

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])

##########################Univariate, Bivariate################
plt.hist(crime["Murder"])   #Univariate

plt.hist(crime["Assault"])

plt.hist(crime["UrbanPop"])

plt.hist(crime["Rape"])

crime.skew(axis = 0, skipna = True) 

crime.kurtosis(axis = 0, skipna = True)

# calculating TWSS - Total within SS using different cluster range
from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Plotting the Scree plot using the TWSS from above defined function
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters, 
# as the curve is seemingly bent or showinf an elbow format at K = 4

model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row

mb = pd.Series(model.labels_)  # converting numpy array into pandas series object

crime['clust'] = mb # creating a  new column and assigning it to new column

crime.head()

crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

# We can clearly see that we have the labels in the dataset in the form of a column called "clust", symbolizing the clusters

# In order to see the clusters we aggregate the records within the clusters and group them by the clusters to visualize the 
# 4 nos of clear cluster formed
crime.iloc[:, 1:6].groupby(crime.clust).mean()


"""#QUE 3. Analyze the information given in the following ‘Insurance Policy dataset’ to  create clusters of persons falling in the same type. 
Refer to Insurance Dataset.csv"""

import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Visualization
import numpy as np                #for Mathematical calculations
import seaborn as sns             #for Advanced visualizations

ins = pd.read_csv(r"D:\assignments\All datasets\Kmeans\Insurance Dataset.csv")

ins.head()

# We see the columns in the dataset
ins.columns

ins.head()


# As a part of the Data cleansing we check the data for any missing/ na values
ins.isna().sum()

# Additionally we check the data for any duplicate values, now this can be an optional check depending on the data being used
ins1 = ins.duplicated()
sum(ins1)

# We see the data again now to check whether the data is in scale
ins.describe

# we notice that the data needs to be normalise, using normalization

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ins.iloc[:,:])

##########################Univariate, Bivariate################
plt.hist(ins)   #Univariate

ins.skew(axis = 0, skipna = True) 

ins.kurtosis(axis = 0, skipna = True)



# calculating TWSS - Total within SS using different cluster range
from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Plotting the Scree plot using the TWSS from above defined function
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters, 
# as the curve is seemingly bent or showinf an elbow format at K = 4

model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row

mb = pd.Series(model.labels_)  # converting numpy array into pandas series object

ins['clust'] = mb # creating a  new column and assigning it to new column

ins.head()

ins = ins.iloc[:,[5,0,1,2,3,4]]
ins.head()

# We can clearly see that we have the labels in the dataset in the form of a column called "clust", symbolizing the clusters

# In order to see the clusters we aggregate the records within the clusters and group them by the clusters to visualize the 
# 4 nos of clear cluster formed
ins.iloc[:, 1:6].groupby(ins.clust).mean()




"""#QUE 4.	Perform clustering analysis on the telecom dataset. The data is a mixture of both categorical and numerical data. 
It consists of the number of customers who churn. Derive insights and get possible information on factors that may affect the churn decision. 
Refer to Telco_customer_churn.xlsx dataset."""


# libraries required
import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Visualization
import numpy as np                #for Mathematical calculations
import seaborn as sns             #for Advanced visualizations

tele = pd.read_excel(r"D:\assignments\All datasets\Kmeans\Telco_customer_churn.xlsx")

tele.info()

# As a part of the Data cleansing we check the data for any missing/ na values
tele.isna().sum()

# Additionally we check the data for any duplicate values, now this can be an optional check depending on the data being used
tele1 = tele.duplicated()
sum(tele1)



# Now we import the label encoder function from scikit learn
from sklearn.preprocessing import LabelEncoder

#creating instance of labelencoder
labelencoder=LabelEncoder()

x = tele.iloc[:, [3,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23]]   # moving columns neede for encoding into x
x.isna().sum()       
y = tele.iloc[:, [0,1,2,4,5,8,12,24,25,26,27,28,29]]     # moving columns which are not needed for encoding into y

# We start creating labels for the categorical features for the ease of working on the data,
# in other words easier for the system or program to understand and interpret

x['Referred a Friend']=labelencoder.fit_transform(x['Referred a Friend'])
x['Offer']=labelencoder.fit_transform(x['Offer'])
x['Phone Service']=labelencoder.fit_transform(x['Phone Service'])
x['Multiple Lines']=labelencoder.fit_transform(x['Multiple Lines'])
x['Internet Service']=labelencoder.fit_transform(x['Internet Service'])
x['Internet Type']=labelencoder.fit_transform(x['Internet Type'])
x['Online Backup']=labelencoder.fit_transform(x['Online Backup'])
x['Online Security']=labelencoder.fit_transform(x['Online Security'])
x['Device Protection Plan']=labelencoder.fit_transform(x['Device Protection Plan'])
x['Premium Tech Support']=labelencoder.fit_transform(x['Premium Tech Support'])
x['Streaming TV']=labelencoder.fit_transform(x['Streaming TV'])
x['Streaming Movies']=labelencoder.fit_transform(x['Streaming Movies'])
x['Streaming Music']=labelencoder.fit_transform(x['Streaming Music'])
x['Unlimited Data']=labelencoder.fit_transform(x['Unlimited Data'])
x['Contract']=labelencoder.fit_transform(x['Contract'])
x['Paperless Billing']=labelencoder.fit_transform(x['Paperless Billing'])
x['Payment Method']=labelencoder.fit_transform(x['Payment Method'])

# label encode y ##
y = pd.DataFrame(y)

# concatenate x and y
tele_new=pd.concat([x,y],axis=1)
tele_new.columns
tele_new.isna().sum()
tele_new.describe()
tele_new.info()



# Univariate and Bivariate analysis on the dataset
plt.hist(tele["Referred a Friend"])   #Univariate

plt.hist(tele["Offer"])

plt.hist(tele["Phone Service"])

plt.hist(tele["Multiple Lines"])

plt.hist(tele["Internet Service"])

plt.hist(tele["Internet Type"])

plt.hist(tele["Online Security"])

plt.hist(tele["Online Backup"])

plt.hist(tele["Device Protection Plan"])

plt.hist(tele["Premium Tech Support"])

plt.hist(tele["Streaming TV"])

plt.hist(tele["Streaming Movies"])

plt.hist(tele["Streaming Music"])

plt.hist(tele["Unlimited Data"])

plt.hist(tele["Contract"])

plt.hist(tele["Paperless Billing"])

plt.hist(tele["Payment Method"])

plt.hist(tele["Number of Referrals"])

plt.hist(tele["Tenure in Months"])

plt.hist(tele["Avg Monthly Long Distance Charges"])

plt.hist(tele["Avg Monthly GB Download"])

plt.hist(tele["Monthly Charge"])

plt.hist(tele["Total Charges"])

plt.hist(tele["Total Refunds"])

plt.hist(tele["Total Extra Data Charges"])

plt.hist(tele["Total Long Distance Charges"])

plt.hist(tele["Total Revenue"])

plt.scatter(tele["Tenure in Months"], tele["Total Revenue"]);plt.xlabel('Tenure in Months');plt.ylabel('Total Revenue')   #Bivariate

tele_new.skew(axis = 0, skipna = True)   #skewness

tele_new.kurtosis(axis = 0, skipna = True)    #kurtosis

tele_new.describe()

tele_new.info()

tele_new = tele_new.drop(['Customer ID','Count','Quarter'], axis = 1)

tele_new.info()

# Since the data is with different scales we Normalize the same by defining and running the Norm function
def norm(i):
    x = (i - i.min())/(i.max() - i.min())
    return x

df_norm = norm(tele_new.iloc[:,:])
df_norm.head()



# calculating TWSS - Total within SS using different cluster range
from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Plotting the Scree plot using the TWSS from above defined function
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df_norm['clust'] = mb # creating a  new column and assigning it to new column

df_norm.head()

# We can clearly see that we have the labels in the dataset in the form of a column called "clust", symbolizing the clusters
# In order to see the clusters we aggregate the records within the clusters and group them by the clusters to visualize the 
# 3 nos of clear cluster formed
df_norm.iloc[:, 0:53].groupby(df_norm.clust).mean()


"""#QUE 5.	Perform clustering on mixed data. Convert the categorical variables to numeric by using dummies or label encoding and perform normalization techniques.
 The dataset has the details of customers related to their auto insurance. Refer to Autoinsurance.csv dataset"""

import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Visualization
import numpy as np                #for Mathematical calculations
import seaborn as sns             #for Advanced visualizations

auto = pd.read_csv(r"D:\assignments\All datasets\Kmeans\AutoInsurance.csv")

auto

# We see the columns in the dataset
auto.columns


# As a part of the Data cleansing we check the data for any missing/ na values
auto.isna().sum()

# check the data for any duplicate values
auto1 = auto.duplicated()
sum(auto1)


# Importing the Label Encoder 
from sklearn.preprocessing import LabelEncoder

#creating instance of labelencoder
labelencoder = LabelEncoder()

# separating the categorical and non in the dataset, which can be concatenated later
x = auto.iloc[:, [1,3,4,5,7,8,10,11,17,18,19,20,22,23]]   #moving columns neede for encoding into x
x.isna().sum()       
y = auto.iloc[:, [0,2,6,9,12,13,14,15,16,21]]     #moving columns which are not needed for encoding into y

# Createing labels for each column
x['State'] = labelencoder.fit_transform(x['State'])
x['Response'] = labelencoder.fit_transform(x['Response'])
x['Coverage'] = labelencoder.fit_transform(x['Coverage'])
x['Education'] = labelencoder.fit_transform(x['Education'])
x['EmploymentStatus'] = labelencoder.fit_transform(x['EmploymentStatus'])
x['Gender'] = labelencoder.fit_transform(x['Gender'])
x['Location Code'] = labelencoder.fit_transform(x['Location Code'])
x['Policy Type'] = labelencoder.fit_transform(x['Policy Type'])
x['Policy'] = labelencoder.fit_transform(x['Policy'])
x['Renew Offer Type'] = labelencoder.fit_transform(x['Renew Offer Type'])
x['Sales Channel'] = labelencoder.fit_transform(x['Sales Channel'])
x['Marital Status'] = labelencoder.fit_transform(x['Marital Status'])
x['Vehicle Class'] = labelencoder.fit_transform(x['Vehicle Class'])
x['Vehicle Size'] = labelencoder.fit_transform(x['Vehicle Size'])

# label encode y ##
y = pd.DataFrame(y)

# Concatenate x and y
auto_new = pd.concat([x, y], axis = 1)
auto_new.columns
auto_new.isna().sum()

auto_new.describe()
auto_new.info()


# We now plot the boxplot for the data using each feature independently and check for Outliers
plt.boxplot(auto_new['Customer Lifetime Value']);plt.title('Boxplot');plt.show()

# We see that there are Outliers present for "Balance" Feature

plt.boxplot(auto_new['Total Claim Amount']);plt.title('Boxplot');plt.show()

# Outliers are present

plt.boxplot(auto_new['Monthly Premium Auto']);plt.title('Boxplot');plt.show()

# Outliers are present


from scipy.stats.mstats import winsorize

auto_new['Customer Lifetime Value']=winsorize(auto_new['Customer Lifetime Value'],limits=[0.09, 0.091])   
plt.boxplot(auto_new['Customer Lifetime Value']);plt.title('Boxplot');plt.show()

auto_new['Total Claim Amount']=winsorize(auto_new['Total Claim Amount'],limits=[0.05, 0.095])   
plt.boxplot(auto_new['Total Claim Amount']);plt.title('Boxplot');plt.show()

auto_new['Monthly Premium Auto']=winsorize(auto_new['Monthly Premium Auto'],limits=[0.05, 0.095])   
plt.boxplot(auto_new['Monthly Premium Auto']);plt.title('Boxplot');plt.show()

# Now we check the data for zero variance values
(auto_new == 0).all()   #finding which values contains all zero values

auto_new.columns

# We see the data again now to check whether the data is in scale
auto_new.describe

# we notice that the data needs to be normalise, using normalization

auto_new = auto_new.drop(['Customer','State','Effective To Date'], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())/ (i.max() - i.min())
    return (x)

# Normalized data frame 
df_norm = norm_func(auto_new.iloc[:,:])
df_norm.head()


# calculating TWSS
from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Plotting the Scree plot using the TWSS from above defined function
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters, 
# as the curve is seemingly bent or showinf an elbow format at K = 4

model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_

mb = pd.Series(model.labels_)  # converting numpy array into pandas series object

df_norm['clust'] = mb # creating a  new column and assigning it to new column

df_norm.head()

#rearranging the features to get the clusters at the first 
auto1 = df_norm.iloc[:,[21,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
auto1.head()

# We can clearly see that we have the labels in the dataset in the form of a column called "clust", symbolizing the clusters

# In order to see the clusters we aggregate the records within the clusters and group them by the clusters to visualize the 
# 4 nos of clear cluster formed

df_norm.iloc[:, 0:21].groupby(auto1.clust).mean()
