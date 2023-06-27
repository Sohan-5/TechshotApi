import random
import statistics

# set the mean and standard deviation for the normal distribution
mean_time = 180  # in seconds
std_dev = 60  # in seconds

# generate a dataset of random values using a normal distribution
num_articles = 62171  # set the number of articles to read
time_spent_data = [int(random.normalvariate(mean_time, std_dev)) for _ in range(num_articles)]

import pandas as pd
df=pd.read_csv("C:/Users/Aishwarya K/Desktop/book-recommendation-system-webapp-master/data/User_Database - Users.csv")
df

import pandas as pd
import numpy as np
# create dictionary mapping string values to integers
mapping = {value: index + 1 for index, value in enumerate(df['user_id'].unique())}

# replace string values with integers
df['user_id'] = df['user_id'].replace(mapping)
df['time_spent_full'] = np.array(time_spent_data)
print(df)

print(len(df['news_id'].unique()))

# df.groupby(['user_id' , 'news_id']).agg(['viewed','shared','voice_used','quick_viewed'])

data = df.groupby(['user_id' , 'news_id'], as_index = False)[['viewed' ,'shared', 'voice_used', 'quick_viewed', 'bookmark','time_spent_full']].sum()
data



data = data[data['viewed'] < 50]
data = data[data['bookmark'] <=1]

data



pd.unique(data['bookmark'])



import pandas as pd
import numpy as np
def normalize(column_name):
  # Select the column you want to normalize
  column_to_normalize = data[column_name]

  # Calculate the mean and standard deviation of the column
  mean = np.mean(column_to_normalize)
  std = np.std(column_to_normalize)

  # Normalize the column using the z-score formula
  normalized_column = (column_to_normalize - mean) / std

  # Replace the original column with the normalized column in the dataframe
  data[column_name] = normalized_column

# normalize('bookmark')
# normalize('shared')
# normalize('viewed')
# normalize('quick_viewed')
# normalize('voice_used')
# normalize('time_spent_full')
data



data['bookmark'] = data['bookmark']*0.4
data['shared'] = data['shared']*0.4
data['viewed'] = data['viewed']*0.3
data['quick_viewed'] = data['quick_viewed']*0.2
data['voice_used'] = data['voice_used']*0.1
data['time_spent_full'] = data['time_spent_full']*0.4

# data['Rating'] = data.iloc[:, -5:-1].sum(axis=1)
data['Rating'] = data['bookmark']+data['viewed'] + data['shared'] + data['quick_viewed'] + data['voice_used']+data['time_spent_full']

data

#0-5 rating
max_rating = data['Rating'].max()

data['rating_scaled'] = (data['Rating'] / max_rating) * 5
data['rating_rounded'] = data['rating_scaled'].round(1)

data

# importing the required libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Creating an instance of the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()

# Scaling the Rating column of the created dataFrame and storing
data[["ScaledRating"]] = (scaler.fit_transform(data[["Rating"]])*9+1)
data['ScaledRating'] = data['ScaledRating'].round(1)

sorted(pd.unique(data['ScaledRating']))

data.drop( ['viewed','shared',	'voice_used','quick_viewed','bookmark','Rating','time_spent_full'],axis=1, inplace=True)
data

data.to_csv('data.csv', index=False)


from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
def predict_function(input_data):

# Define the rating scale (in this case, from 1 to 5)
  reader = Reader(rating_scale=(1, 100))

  # Load the data into a Surprise Dataset
  dataset = Dataset.load_from_df(data[['user_id', 'news_id', 'ScaledRating']], reader)

  # Split the data into training and testing sets
  trainset, testset = train_test_split(dataset, test_size=0.2)

  # Train an SVD model
  model = SVD(n_factors=50, lr_all=0.01, reg_all=0.1)
  model.fit(trainset)

  # Make predictions for the target user

          # Make predictions for the target user
  target_user = input_data

  target_news = data[data['user_id'] == target_user]['news_id'].unique()
  unseen_news = data[~data['news_id'].isin(target_news)]['news_id'].unique()
  predictions = [model.predict(target_user, news_id) for news_id in unseen_news]


  # Sort the list of predictions by their predicted ratings
  top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]
  # print(top_predictions)
  # Print the top 10 predicted news articles for the target users
  
    

  #     print(')
  # return render_template('svd.html', prediction_text='Recommended articles are  {}'.format(top_predictions ))
  # return render_template('svd.html', prediction=top_predictions)
  return top_predictions


# Print the top 10 predicted news articles for the target user


















