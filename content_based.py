
import numpy as np
import pandas as pd

import os
import math
import time


# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances


news_articles = pd.read_csv("news_data.csv")



news_articles = news_articles[news_articles['is_active'] == 'yes']


news_articles.rename(columns = {'main_title':'headline'}, inplace = True)


news_articles.sort_values('headline',inplace=True, ascending=False)
duplicated_articles_series = news_articles.duplicated('headline', keep = False)
news_articles = news_articles[~duplicated_articles_series]
print("Total number of articles after removing duplicates:", news_articles.shape[0])

news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split())>5)]
print("Total number of articles after removal of headlines with short title:", news_articles.shape[0])

news_articles.isna().sum()

news_articles.dropna(inplace = True)

di = {
     1:"Fashion",
     2:"Entertainment",
     3:"Buisness",
     4:"Sports",
     9:"Technology",
     12:"Test",
     13:"Elections",
     14:"Test",
     15:"World",
     19:"Security",
     20:"Big Data",
     21:"Cloud",
     22:"AI",
     23:"IOT",
     24:"Blockchain",
     25:"Automation",
     26:"Digital Transformation",
     27:"AR/VR",
     28:"Others",
     29:"Buisness",
     30:"Buisness",
     31:"People",
     32:"NASSCOM Research",
     33:"Startup",
     34:"Case Study"
     }

news_articles.replace({"category_id": di},inplace= True)
news_articles.rename(columns = {'category_id':'category'}, inplace = True)
news_articles['created_at'] = pd.to_datetime(news_articles['created_at'],format='%Y-%m-%d',errors='coerce')
news_articles['date'] = news_articles['created_at'].dt.date
news_articles.reset_index()

# Adding a new column containing both day of the week and month, it will be required later while recommending based on day of the week and month
#news_articles["day and month"] = news_articles["date"].dt.strftime("%a") + "_" + news_articles["date"].dt.strftime("%b")

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
stop= set(stopwords.words('english'))

news_articles_temp = news_articles.copy()
news_articles_temp.astype(str)
news_articles_temp.dtypes

news_articles_temp["headline"] = news_articles_temp["headline"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

news_articles_temp["headline"] = news_articles_temp["headline"].apply(lemmatize_text)
news_articles_temp["headline"] = news_articles_temp["headline"].apply(lambda x : " ".join(x))


from nltk.tokenize import RegexpTokenizer
import re
# Function for removing NonAscii characters
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)
# Function for converting into lower case
def make_lower_case(text):
    return text.lower()
# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text
# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text
#Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
#Function for removing rdquo, ldquo, quot
def remove_words(text):
  word_list = ["rdquo","ldquo","quot"]
  tokenizer = RegexpTokenizer(r'\w+')
  words = tokenizer.tokenize(text)
  text = ' '.join([word for word in words if word not in word_list])
  return text
  
# Applying all the functions in description and storing as a cleaned_desc
news_articles_temp['cleaned_desc'] = news_articles_temp['short_description'].apply(_removeNonAscii)
news_articles_temp['cleaned_desc'] = news_articles_temp.cleaned_desc.apply(func = make_lower_case)
news_articles_temp['cleaned_desc'] = news_articles_temp.cleaned_desc.apply(func = remove_stop_words)
news_articles_temp['cleaned_desc'] = news_articles_temp.cleaned_desc.apply(func=remove_punctuation)
news_articles_temp['cleaned_desc'] = news_articles_temp.cleaned_desc.apply(func=remove_words)
news_articles_temp['cleaned_desc'] = news_articles_temp.cleaned_desc.apply(func=remove_html)

news_articles_temp.reset_index()

pd.set_option('display.max_colwidth', -1)

#TF-IDF
tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp['headline'])
tfidf_desc_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp['cleaned_desc'])
#based on category and headline
from sklearn.preprocessing import OneHotEncoder 
category_onehot_encoded = OneHotEncoder().fit_transform(np.array(news_articles_temp["category"]).reshape(-1,1))

def tf_idf_based_model(row_index, num_similar_items,w1,w2,w3):
    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index].reshape(1,-1))
    couple_desc_dist = pairwise_distances(tfidf_desc_features,tfidf_desc_features[row_index].reshape(1,-1))
    category_dist = pairwise_distances(category_onehot_encoded, category_onehot_encoded[row_index]) + 1
    weighted_couple_dist   = (w1 * couple_dist +  w2 * couple_desc_dist +  w3 * category_dist)/float(w1 + w2 + w3)
    indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
    print(indices)
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
               'Weighted Euclidean similarity with the queried article': weighted_couple_dist[indices].ravel(),
               'Headline similarity': couple_dist[indices].ravel(),
               'Desc similarity': couple_desc_dist[indices].ravel(),
               'Category based Euclidean similarity': category_dist[indices].ravel(),
               'Category': news_articles['category'][indices].values,
               'Desc': news_articles['short_description'][indices].values
               })
    # print("="*30,"Queried article details","="*30)
    # print('headline : ',news_articles['headline'][indices[0]])
    # print('category : ',news_articles['category'][indices[0]])
    # print('desc : ',news_articles['short_description'][indices[0]])
    # print("\n","="*25,"Recommended articles : ","="*23)
    indices.pop(0)
    rec=[]
    for i in indices:
        rec.append('News article {iid} (predicted rating: {est})'.format(iid=news_articles['id'][i],est=weighted_couple_dist[i].ravel())) 
    
    #return df.iloc[1:,1]
    return rec




# from flask import Flask, render_template, request

# app = Flask(__name__)


# @app.route('/home', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         input_data = request.form['input']
       
#         list1 = [1, 2, 3]
#         list2 = ['a', 'b', 'c']
#         return render_template("home.html", list1=list1, list2=list2)

        
#     else:
#         return render_template('result.html')
    

# @app.route('/',methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         input_data = request.form['input']
       
#         list1 = ["Microsoft headquarters to open on Feb 28th for all employees",
# "Telecom deals will transform mobile payments in India"	,
# "Google to allow 32 people on Duo video call soon",
# "MG Motor India collaborates with Cognizant for India’s first ‘Connected Internet Car’",	
# "	New malware EventBot may attack Indian banking apps	",
# "First AI and Robotic Tech park launched in Karnataka",
# "This is how Raymond is fuelling digital transformation",
# "Google Employees who work from home could be facing a pay cut",
# "Facebook disabled 1.3 bn fake accounts between Oct-Dec 2020	",
# "Quick Heal makes a strategic investment in Ray"]
#         list2 = [	2.828427,2.828427,	2.828427,2.828427,3.000000,3.000000,3.000000,3.000000,3.000000,3.000000]
#         return render_template("home.html",show_lists=True, list1=list1, list2=list2)
#     else:
#         return render_template('home.html')
# if __name__ == "__main__":
#     app.run()

