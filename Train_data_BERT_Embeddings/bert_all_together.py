import numpy as np
import pandas as pd
import pickle

# from transformers import AutoModel
# model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

from sentence_transformers import SentenceTransformer
model_path = "E:/Ayan_Final_Year_Project/Restaurant-Reviews-Sentiment-Analysis/BERT_models/"
model = SentenceTransformer(model_path + 'bert-base-nli-mean-tokens')

# model = SentenceTransformer('bert-base-nli-mean-tokens')

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#dataset = pd.read_csv('train_E6oV3lV.csv')

#lines = ['Hello this is a tutorial on how to convert the word in an integer format',
#       'this is a beautiful day','Jack is going to office']


import re
#import nltk
#nltk.download('stopwords')
corpus = []
dataset["Clean_Review"] = ""
for i in range(len(dataset)):
    #review = re.sub('[^a-zA-Z0-9]', ' ', dataset['tweet'][i])
    review = re.sub('[^a-zA-Z0-9]', ' ', dataset['Review'][i])
    review = review.lower()
    review = re.sub(' +', ' ', review)
    review = re.sub('user', '', review)
    review = review.strip()
    #review = review.split()
    #review = [word for word in review if not word in set(stopwords.words('english'))]
    #review = ' '.join(review)
    dataset.at[i, "Clean_Review"] = review
    corpus.append(review)


# Corpus is a list type
corpus_df = pd.DataFrame(corpus)
corpus_df.to_csv("clean_review_corpus.csv")
dataset.to_csv("Restaurant_Reviews_clean.csv", index=False)

'''
corpus_token = []
for j in range(0, 31962):
    text_sample = corpus[j]
    text_sample = text_sample.split()
    corpus_token.append(text_sample)
    
corpus_token_original = corpus_token    
'''

# Getting list of Y
y = dataset.iloc[:, 1].values
    

corpus_list = corpus_df[0].values.tolist()


# Calculating BERT Sentence embeddings
sentence_embeddings = model.encode(corpus_list)
#sentence_embeddings, _ = model.encode(corpus_list, [])
df_embedding = sentence_embeddings.copy()
    

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_embedding , y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


'''
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
'''

'''
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
'''


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuray = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])

TP  = cm[0][0]
TN  = cm[1][1]
FP  = cm[1][0]
FN  = cm[0][1]

Precision = TP/(TP+FP)
Recall  = TP/(TP+FN)

F1_Score  = 2*(Recall*Precision) / (Recall + Precision)
# With the GloVe embeddings loaded in a dictionary, we can look up the embedding for each word in the corpus of the airline tweets

# save the model to disk
filename = 'finalized_TSA_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)



##################################################################################################

sample_data = '@user @user lumpy says i am a . prove it lumpy.'
sample_data = dataset.iloc[999,:]
sample_data_np  = np.array(sample_data)
sample_data_2d  = sample_data_np.reshape(-1,1)
sample_data_2d_T = np.transpose(sample_data_2d)
Prediction = classifier.predict(sample_data_2d_T)

y_pred_1 = np.array(y_pred)


term = 'uhbdbdkbchkc'
if term in emb_dict:
    print('Present')
else:
    print('notPresent')


result_dict = {}
result_dict['key'] = 0   
    
a = [1,2,3,4,5] 
arr1 = np.array(a)
b = [1,1,1,1,1]
arr2 = np.array(b)
c = [10,10,10,10,10]
arr3 = np.array(c)

list1 = [arr1, arr2, arr3]
list1 = np.array(list1)
sum_ist = list1.sum(axis=0)
