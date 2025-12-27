# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

 import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("bmi.csv")

df.head()

output <img width="836" height="214" alt="image" src="https://github.com/user-attachments/assets/36eacb6b-f035-4cfe-a457-a25444a5df52" />

df.dropna()

output <img width="694" height="447" alt="image" src="https://github.com/user-attachments/assets/ce10b3d0-e7f3-417f-a6c9-db0691789583" />

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

output <img width="1360" height="184" alt="image" src="https://github.com/user-attachments/assets/ab2d0976-cfcf-4879-98fe-92694a742d7c" />

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head(10)

output <img width="613" height="377" alt="image" src="https://github.com/user-attachments/assets/a62a03a2-5599-48ca-b3ed-96b91c828e41" />

df3=pd.read_csv("bmi.csv")

from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3

output <img width="1065" height="450" alt="image" src="https://github.com/user-attachments/assets/54e03602-6fc8-4349-8acb-0de3aa8bc570" />

df4=pd.read_csv("bmi.csv")

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4

output <img width="865" height="451" alt="image" src="https://github.com/user-attachments/assets/838f948f-f9e6-4cfe-a34e-01ad571bb38a" />

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV,LassoCV,Ridge,Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("titanic_dataset.csv")

df.columns

output <img width="1066" height="88" alt="image" src="https://github.com/user-attachments/assets/ca52b969-e966-4c11-afbe-d94bb639ce81" />

df.shape

output <img width="496" height="37" alt="image" src="https://github.com/user-attachments/assets/719de42f-d011-4e6a-93da-3f0e72ca495d" />

x = df.drop("Survived",1)
y = df['Survived']

<img width="1322" height="90" alt="image" src="https://github.com/user-attachments/assets/dfd396a3-2a5c-40cf-896f-16ee47b3f2b0" />

df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df1.columns

output <img width="1028" height="32" alt="image" src="https://github.com/user-attachments/assets/f87fb862-e2b2-4c01-ae84-4939954c15f8" />

 df1['Age'].isnull().sum()

 output <img width="698" height="46" alt="image" src="https://github.com/user-attachments/assets/866f5de4-1f87-4843-8b30-0e79fa5a66a7" />

df1['Age'].fillna(method='ffill')

output <img width="739" height="270" alt="image" src="https://github.com/user-attachments/assets/8e288d77-b562-4121-8259-13449dc25118" />

df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()

output <img width="816" height="43" alt="image" src="https://github.com/user-attachments/assets/65698aa4-6b95-4265-8072-bf96e0f1287f" />

df1.columns

output <img width="1028" height="34" alt="image" src="https://github.com/user-attachments/assets/cb01fd4b-8ab8-480a-a435-a93b7213f1e8" />

cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]
df1=df1[cols]

df1.columns

output <img width="1039" height="36" alt="image" src="https://github.com/user-attachments/assets/b0274d8d-918e-47df-a713-8c8ebde456f6" />

x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns

output <img width="1026" height="45" alt="image" src="https://github.com/user-attachments/assets/e5e30654-c9cb-4fd3-9487-0e54407dfeec" />

y=y.to_frame()

y.columns

output <img width="763" height="49" alt="image" src="https://github.com/user-attachments/assets/91b1cf12-e270-4e41-b0b0-b5ba905a9b14" />

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data=pd.read_csv('titanic_dataset.csv')

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x

output <img width="1209" height="460" alt="image" src="https://github.com/user-attachments/assets/f693ad5c-8cd3-43ee-b666-72a1a5ae36dd" />

data

output <img width="1333" height="463" alt="image" src="https://github.com/user-attachments/assets/1f42c13a-d20b-4328-a3e8-8b1fc392a99d" />

x.info()

<img width="630" height="370" alt="image" src="https://github.com/user-attachments/assets/babade27-0d98-4232-a153-d70a2c955dc0" />

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("titanic_dataset.csv")

df.columns

output <img width="1037" height="85" alt="image" src="https://github.com/user-attachments/assets/c7ef4176-a613-4dea-9104-c0bc9f573db9" />

df

output <img width="1343" height="450" alt="image" src="https://github.com/user-attachments/assets/6b8baa8d-079a-42a6-83bd-5dc5e0e2b901" />

df=df.dropna()

df.isnull().sum()

output <img width="492" height="288" alt="image" src="https://github.com/user-attachments/assets/6897abf5-9617-4a26-b78a-98031afef276" />

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips =sns.load_dataset('tips')

tips.head()

output <img width="732" height="219" alt="image" src="https://github.com/user-attachments/assets/3b69881c-f570-440d-8f85-5ff2947ff716" />

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

output <img width="771" height="94" alt="image" src="https://github.com/user-attachments/assets/b6f503c4-a018-477a-bc12-69f4f8337cb2" />

# RESULT:

 Thus, the program to implement Feature Scaling and Feature Selection was completed successfully.
