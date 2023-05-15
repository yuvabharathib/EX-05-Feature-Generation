# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
# Data.csv:
```
Name: YUVABHARATHI.B
Reg No. : 212222230181
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
# Encoding.csv :
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
# Titanic.csv :
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT:
# Data.csv :
## Initial Dataset:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/ff977c81-26f2-4d8b-8be1-5e89fe36ab38)


## Binary Encoding:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/2fad02d6-4ce8-4604-b1ff-cea0449d64a9)
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/ff6915c2-58c0-4ad2-bc8c-8fb9238c3c2a)


## Encoded Dataset:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/b815f439-3072-4b1e-91a4-44b584c1119f)


## Data Scaling using MinMaxScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/5048b0c6-2628-4f9d-bbb1-ed80f2bc35ae)


## Data Scaling using StandardScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/6a389ecb-2474-416a-b4f4-391221576727)


## Data Scaling using MaxAbsScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/5773ed91-8285-4844-b207-df7144acc1d2)


## Data Scaling using RobustScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/5b0f5311-b56a-4c1c-b030-07bff0c740e1)

# Encoding.csv :

## Initial Dataset:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/67676f44-8636-420f-becb-5b908095b2b3)



## Binary Encoding:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/ece35eaa-3a4b-4c2c-8066-27e1f7092a11)
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/b9a1925a-1ad7-4a6d-813e-5b0316ad5cf4)


## Encoded Dataset:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/03976772-4a94-4e61-af02-00dc5c65da49)


## Data Scaling using MinMaxScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/a07f92f0-794f-4a0c-98b8-0eafdccb7271)


## Data Scaling using StandardScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/ab44364a-3a4a-42f4-9c67-1da83d1f2205)


## Data Scaling using MaxAbsScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/3eb206c9-1554-47cc-98a2-f7c5771b7c9d)


## Data Scaling using RobustScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/be9ee516-9e18-4e48-8015-1e1d1ee2feaf)


# Titanic.csv :
## Initial Dataset:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/1671d561-15fd-4cd5-a66b-7b40b6e69e43)


## Data cleaning before encoding:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/61a173cb-03a4-4d47-bcc2-a3f5a8bab0cc)


## Cleaned Dataset:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/c27c966d-909f-4ece-afbd-f79acffb46fa)


## Binary Encoding:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/f219460d-6ea1-4f2b-9551-a99c6630072e)


## Encoded Dataset:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/57996f42-ec9a-434e-8c9d-99e9d82cd8f6)


## Data Scaling using MinMaxScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/258b1eec-eb63-4def-83a9-cb8c9e1b1f81)


## Data Scaling using StandardScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/a5681ca2-13a0-4fe4-85e4-b5a72bbc60a9)


## Data Scaling using MaxAbsScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/b0026785-84dc-4e54-9466-6d443c717550)


## Data Scaling using RobustScaler:
![image](https://github.com/yuvabharathib/EX-05-Feature-Generation/assets/113497404/3027bd6c-6955-4e60-9f2e-ad0d869677d9)

# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.


