#!/usr/bin/env python
# coding: utf-8

# # Name: Kore Pratiksha Jayant

# # Task 2: Email Spam Filtering

# # 

# # Exploratory Data Analysis(EDA) and Visualization

# In[1]:


##Import Library

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import numpy as np
import pandas as pd


# In[2]:


##Import Dataset

data=pd.read_csv('C:/Users/stati/OneDrive/Desktop/emails dataset.csv',encoding='latin-1', usecols = ['v1', 'v2'])
data


# In[3]:


##Rename Columns

data.rename(columns={"v1":"Class","v2":"Message"},inplace=True)
data.head(2)


# In[4]:


pd.set_option('display.max_columns', 500)


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


##To check duplicate value

data.duplicated().sum()


# In[8]:


##Removing the duplicate values

data.drop_duplicates(inplace=True)
data


# Removed duplicated data

# In[9]:


##To check null value

data.isnull().sum()


# No null values are present

# In[10]:


##To check data is Blanced or not

classes = data.groupby('Class').count()
classes['Message']


# In[11]:


import matplotlib.pyplot as plt

colors=['gray','lightblue']
plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
plt.pie(data['Class'].value_counts(),labels=['Ham','Spam'],colors=colors,autopct="%0.2f",shadow=True)
plt.title("Total Spam & Ham",size=8,color='black')


# As seen Data is Imbalance here first we process with imbalace data.

# In[12]:


data['Class'] = data['Class'].replace({'spam': 1, 'ham': 0})


# In[13]:


##Apply undersampling

counts_class_0,counts_class_1=data['Class'].value_counts()
data_0=data[data['Class']==0]
data_1=data[data['Class']==1]
data_under_0=data_0.sample(counts_class_1,random_state=42)
data_under=pd.concat([data_under_0,data_1],axis=0)
data_under.shape


# In[14]:


view=data_under['Class'].value_counts()
slice=list(round(view, 1))
colors=['gray','lightblue']
plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
plt.pie(slice,labels=['Ham','Spam'],colors=colors,autopct="%0.2f",shadow=True)
plt.title("Total Spam & Ham",size=8,color='black')


#  Now it became Balanced Data

# # Feature Engineering

# In[15]:


# Total No. of Characters in Data
data["characters"] = data["Message"].apply(len)


# In[16]:


pip install nltk


# In[17]:


import nltk
nltk.download('punkt')  # Download the NLTK punkt tokenizer data if you haven't already

# Tokenize and count words in the "Message" column
data["word"] = data["Message"].apply(lambda x: len(nltk.word_tokenize(x)))

# Calculate the total number of words
total_words = data["word"].sum()

print("Total Number of Words in Data:", total_words)


# In[18]:


# Total No. of Sentence
data["sentence"] = data["Message"].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[19]:


New_data=data
New_data.head()


# In[20]:


# Statistical Analysis of new features
New_data[["characters","word", "sentence"]].describe()


# In[21]:


# Statistical Analysis for Ham Data

New_data[New_data["Class"]==0][["characters","word", "sentence"]].describe()


# In[22]:


# Statistical Analysis for spam Data

New_data[New_data["Class"] ==1][["characters","word", "sentence"]].describe()


# In[23]:


New_data.describe().apply(round)


# # Histogram Plot

# In[24]:


plt.figure(figsize=(20,10))
sns.histplot(New_data[New_data["Class"]==0]["characters"],label= "ham",color="lightblue")
sns.histplot(New_data[New_data["Class"]==1]["characters"],label= "spam",color = "red")
plt.title("Spam Vs Ham : Characters",size=15,color='black')
plt.legend()
plt.show()


# In[25]:


plt.figure(figsize=(20,10))
sns.histplot(New_data[New_data["Class"]==0]["word"],label= "ham",color="lightblue")
sns.histplot(New_data[New_data["Class"]==1]["word"],label= "spam",color = "red")
plt.title("Spam Vs Ham : Word",size=15,color='black')
plt.legend()
plt.show()


# Ham Characters and Words are more than Spam 

# In[26]:


##boxplot(Checking Outliers)

fig,axs=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
axes=axs.flatten()
num_columns=['characters', 'word', 'sentence']
for i,col in enumerate(num_columns):
    if(col!='SeniorCitizen'):
        sns.boxplot(x=col,data=New_data,showmeans=True,ax=axes[i],color='lightblue')
fig.tight_layout()
plt.show()


# Outliers are present

# In[27]:


#Handling Outliers
#1. Characters
New_data[New_data['characters']>500]
New_data.drop(New_data[New_data['characters']>500].index,inplace=True,axis=0)

#2. word
New_data[New_data['word']>130]
New_data.drop(New_data[New_data['word']>130].index,inplace=True,axis=0)

#3. Sentence
New_data[New_data['sentence']>15]
New_data.drop(New_data[New_data['sentence']>15].index,inplace=True,axis=0)


# In[28]:


from nltk.stem.porter import PorterStemmer


# In[29]:


# Intilizing Porter Stemmer Class
ps = PorterStemmer()


# In[30]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()                                   # Convert text to lowercase
    words = nltk.word_tokenize(text)                      # Tokenize the text (split into words)
    filtered_words = [word for word in words if word.isalnum()] 
                                                          # Remove special characters and keep only alphanumeric characters
    filtered_words = [word for word in filtered_words if word not in stopwords.words('english') and word not in string.punctuation]
                                                          # Remove stopwords and punctuation
    stemmer = PorterStemmer()                             # Apply Porter Stemming to reduce words to their root form
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    preprocessed_text = " ".join(stemmed_words)           # Join the preprocessed words back into a single string

    return preprocessed_text


# In[31]:


preprocessed_text = preprocess_text("shree ram 🚩 and shree mahakaleshawar 🔱 are everything for me")
print(preprocessed_text)


# In[32]:


New_data.sample(5)


# In[33]:


# Transforming dataset in new column "New_Message"

New_data["New_Message"] = New_data["Message"].apply(preprocess_text)
New_data.sample(5)


# # Barplot

# In[34]:


##Split SPAM Sentences into Words
spam_corpus = []
for msg in New_data[New_data["Class"] == 1]["New_Message"].tolist():
    for word in msg.split():
        spam_corpus.append(word)

##Get the top 50 SPAM Words
from collections import  Counter

word_counts = Counter(spam_corpus)
top_50_words = word_counts.most_common(50)
top_50_words_df = pd.DataFrame(top_50_words, columns=["Word", "Count"])

##Create a bar plot
plt.figure(figsize=(12, 5))
sns.barplot(data=top_50_words_df, x="Word", y="Count", palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Word")
plt.ylabel("Count")

plt.title("Top 50 SPAM Words")
plt.show()


# In[35]:


##Split HAM Sentences into Words
ham_corpus = []
for msg in New_data[New_data['Class'] == 0]['New_Message'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

##Get the top 50 HAM Words
word_counts = Counter(ham_corpus)
top_50_words = word_counts.most_common(50)
top_50_words_df = pd.DataFrame(top_50_words, columns=["Word", "Count"])

#Create a bar plot
plt.figure(figsize=(12, 5))
sns.barplot(data=top_50_words_df, x="Word", y="Count", palette="magma")
plt.xticks(rotation=90)
plt.xlabel("Word")
plt.ylabel("Count")

plt.title("Top 50 HAM Words")
plt.show()


# # Lable Encoding
# 

# In[36]:


##Import library for lable Encoding

from sklearn.preprocessing import LabelEncoder

New_data=New_data.apply(LabelEncoder().fit_transform)
New_data.head()


# # Data Preprocessing

# In[37]:


##Import Library

from sklearn.model_selection import train_test_split


# In[38]:


##Split inputs and output features

x=New_data.drop('Class',axis=1)
y=New_data['Class']


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(x, New_data['Class'], test_size=0.33, random_state=42)
print([np.shape(x_train), np.shape(x_test)])


# In[40]:


##Train and Test Dataset

from sklearn.preprocessing import StandardScaler


# In[41]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# # Model Training and Model Evaluation

# In[42]:


##Importing all library need for model training

##Logistic Regression
from sklearn.linear_model import LogisticRegression

##Random Forest
from sklearn.ensemble import RandomForestClassifier

##SVM
from sklearn import svm

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.metrics import classification_report


# Logistic Regression

# In[43]:


LR=LogisticRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)


# In[44]:


print("Model Accuracy :",accuracy_score(y_pred,y_test))
print("Model F1-Score :",f1_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[45]:


SVM = svm.SVC(kernel='linear')
SVM.fit(x_train,y_train)
y_pred=SVM.predict(x_test)


# In[46]:


print("Model Accuracy :",accuracy_score(y_pred,y_test))
print("Model F1-Score :",f1_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[47]:


RF=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=42)
RF.fit(x_train,y_train)
y_pred2=RF.predict(x_test)


# In[48]:


print("Model Accuracy :",accuracy_score(y_pred2,y_test))
print("Model F1-Score :",f1_score(y_pred2,y_test))
print(classification_report(y_pred2,y_test,zero_division=1))


# In[49]:


##Over Sampling

X=New_data.drop('Class',axis=1)
Y=New_data['Class']


# In[50]:


X.shape


# In[51]:


Y.shape


# In[52]:


get_ipython().system('pip install imblearn')


# In[53]:


from imblearn.over_sampling import SMOTE


# In[54]:


x_res,y_res=SMOTE().fit_resample(X,Y)


# In[55]:


y_res.value_counts()


# In[56]:


X_train,X_test,Y_train,Y_test=train_test_split(x_res,y_res,test_size=0.3,random_state=42)


# In[57]:


def mymodel(model):
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    
    print("Model Accuracy :",accuracy_score(Y_pred,Y_test))
    print("Model F1-Score :",f1_score(Y_pred,Y_test))
    print(classification_report(Y_pred,Y_test,zero_division=1))
    
    return model


# In[58]:


lr=mymodel(LogisticRegression())


# In[59]:


svm=mymodel(svm.SVC())


# In[60]:


rf=mymodel(RandomForestClassifier())


# In[61]:


RF1=RandomForestClassifier()
RF1.fit(x_res,y_res)


# In[62]:


get_ipython().system('pip install joblib')


# In[63]:


import joblib


# In[64]:


joblib.dump(RF1,"Spam")


# In[65]:


model=joblib.load("Spam")


# In[66]:


prediction=model.predict([[100,12,40,10,70]])


# In[67]:


if prediction==0:
    print("Email is Spam")
else:
    print("Email is Ham")


# # 

# # END
