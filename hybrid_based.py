# from google.colab import drive
# drive.mount('/content/drive/')

import numpy as np
import pandas as pd
import seaborn as sns


# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances

news_articles = pd.read_csv("C:/Users/Aishwarya K/Desktop/book-recommendation-system-webapp-master/data/news_data.csv")

news_articles = news_articles[news_articles['is_active'] == 'yes']
news_articles.rename(columns = {'main_title':'headline'}, inplace = True)
news_articles.reset_index(drop=True)



news_articles.sort_values('headline',inplace=True, ascending=False)
duplicated_articles_series = news_articles.duplicated('headline', keep = False)
news_articles = news_articles[~duplicated_articles_series]
print("Total number of articles after removing duplicates:", news_articles.shape[0])

news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split())>5)]
print("Total number of articles after removal of headlines with short title:", news_articles.shape[0])

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
news_articles

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop= set(stopwords.words('english'))

news_articles_temp = news_articles.copy()
news_articles_temp["headline"] = news_articles_temp["headline"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
news_articles_temp

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

news_articles_temp["headline"] = news_articles_temp["headline"].apply(lemmatize_text)
news_articles_temp["headline"] = news_articles_temp["headline"].apply(lambda x : " ".join(x))

import pickle
cv = pickle.load(open('C:/Users/Aishwarya K/Desktop/book-recommendation-system-webapp-master/data/cv.pkl','rb'))
tfidf_transformer = pickle.load(open('C:/Users/Aishwarya K/Desktop/book-recommendation-system-webapp-master/data/tfidf_transformer.pkl','rb'))

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

feature_names=cv.get_feature_names_out()


def extract_topn_keywords(text):
  # Create tf-idf vector for current row
  tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))
        
  # Sort the tf-idf vectors by descending order of scores
  sorted_items = sort_coo(tf_idf_vector.tocoo())
        
  # Extract only the top 10 keywords
  keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
        
  return keywords

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
news_articles_temp['keyword_extracted'] = news_articles_temp['cleaned_desc'].apply(extract_topn_keywords)

news_articles_temp

news_articles_temp['keys'] = news_articles_temp['keyword_extracted'].apply(lambda x: ' '.join(x.keys()))
news_articles_temp = news_articles_temp.reset_index(drop = True)
news_articles_temp

import pandas as pd
df=pd.read_csv('C:/Users/Aishwarya K/Desktop/book-recommendation-system-webapp-master/data/User_Database - Users.csv')
df

# df.groupby(['user_id' , 'news_id']).agg(['viewed','shared','voice_used','quick_viewed'])

data = df.groupby(['user_id' , 'news_id'], as_index = False)[['viewed' ,'shared', 'voice_used', 'quick_viewed', 'bookmark']].sum()
print()

## COLLABORATIVE FILTERING DATA

#REMOVING OUTLIERS
data = data[data['viewed'] < 50]
data = data[data['bookmark'] <=1]

#TIME SPENT COLUMN RANDOM DATA
import random
import statistics

# set the mean and standard deviation for the normal distribution
mean_time = 180  # in seconds
std_dev = 60  # in seconds

# generate a dataset of random values using a normal distribution
num_articles = len(data)  # set the number of articles to read
time_spent_data = [int(random.normalvariate(mean_time, std_dev)) for _ in range(num_articles)]


#USER ids FROM abc123 ----> User 1 , User 2

# create dictionary mapping string values to integers
mapping = {value: index + 1 for index, value in enumerate(df['user_id'].unique())}

# replace string values with integers
data['user_id'] = data['user_id'].replace(mapping)
data['time_spent_full'] = np.array(time_spent_data)

data

data['user_id'].value_counts() # CHECK SKEWEDNESS OF DATA

# create content-based recommendation
tfidf = TfidfVectorizer(stop_words='english')
news_articles_temp['content'] = news_articles_temp['headline'] + ' ' + news_articles_temp['cleaned_desc'] + ' ' + news_articles_temp['keys'].fillna('')
tfidf_matrix = tfidf.fit_transform(news_articles_temp['content'])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(news_articles_temp.index, index=news_articles_temp['id'])

# calculate total score for each article based on user interactions
data['score'] = data['quick_viewed']*2 + data['viewed']*3 + data['shared']*4 + data['bookmark']*5
total_score_df = data.groupby('news_id')['score'].sum().reset_index()
total_score_df.rename(columns={'score': 'total_score'}, inplace=True)

#merge article data with total score data
news_articles_temp = news_articles_temp.merge(total_score_df, left_on= 'id',right_on = 'news_id')
news_articles_temp['total_score'].fillna(0, inplace=True)
news_articles_temp

# create collaborative filtering recommendation
user_article_matrix = data.pivot_table(index='user_id', columns='news_id', values='score', fill_value=0)
user_similarity = cosine_similarity(user_article_matrix, user_article_matrix)

def hybrid_recommendation(uid:int):
    content_scores = []
    cf_scores = []
    interacted_articles = data[data['user_id'] == uid]['news_id'].tolist()
    for article_id in news_articles_temp['id']:
        content_score = 0
        cf_score = 0
        if article_id in interacted_articles:
            continue
        idx = indices[article_id]      
        content_scores.append((article_id, content_similarity[idx].sum()))
        if article_id in list(user_article_matrix.columns.values):
          for i, score in enumerate(user_similarity[uid-1]):
            cf_score += score * user_article_matrix.loc[i+1, article_id]
        cf_scores.append((article_id, cf_score))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
    cf_scores = sorted(cf_scores, key=lambda x: x[1], reverse=True)
    hybrid_scores = []
    for i in range(len(content_scores)):
        hybrid_score = content_scores[i][1] + cf_scores[i][1] * 0.5 + news_articles_temp.loc[news_articles_temp['id'] == content_scores[i][0], 'total_score'].values[0] * 0.2
        hybrid_scores.append((content_scores[i][0], hybrid_score))
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for i in range(min(10, len(hybrid_scores))):
        recommendations.append(hybrid_scores[i][0])
    return recommendations