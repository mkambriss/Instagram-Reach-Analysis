import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


data= pd.read_csv("C:/Users/Ambris/Desktop/Data Science Projects/Instagram Reach Analysis/Dataset/Instagram_data_by_Bhanu.csv", encoding= 'latin1')
print(data.head())

#check if we have null values
data.isnull().sum()

#check data info
data.info()

#Analysis of impressions from people I appear to on the home Page
plt.figure(figsize=(10,8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions from Home")
sns.distplot(data['From Home'])
plt.show()

#impressions I am recieving from hashtags
plt.figure(figsize=(10,8))
plt.title("Distribution of Impressions from Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()

#Look at distribution of impressions gained from the explore page
plt.figure(figsize=(10,8))
plt.title("Distribution of Impressions from Explore")
sns.distplot(data['From Explore'])
plt.show()

#Distribution of impressions from instagram from various sources: 
home = data["From Home"].sum()
hashtags=data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels=['From Home','From Hashtags','From Explore','Other']
values=[home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels,
title='Impressions on Instagram Posts from Various Sources',hole=0.5)
fig.show()

#Creating word cloud for the caption column to check which words are mostly used
text=" ".join(i for i in data.Caption)
stopwords= set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Analyzing Relationships between the number of likes and the number of impressions
import statsmodels
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()

#Observing the relationship between the number of comments and the number of impressions:
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()

#Checking the relationship between the number of shares and impressions:
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()

#Checking the relationship between the number of posts saved and impressions: 
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()

#Checking the correlation between all of the columns with the impressions columns:
correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

#Analyzing the conversion rate ( Number of followers you get based on number of visits )
#((Number of follows/Number of Visits)*100)
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)

#Check the Relationship between Number of followers gained and Number of page visits
figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()

#Instagram Reach Prediction Model
#Split data into training and test sets
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#Prediction due to inputs

features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
print(model.predict(features))

