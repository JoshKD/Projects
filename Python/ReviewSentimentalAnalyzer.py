#https://www.datacamp.com/datalab/w/1e5741f3-4f78-4587-909a-72b71e6b61a9/edit

#Install plugins
!pip install nltk
!pip install strip-tags
!pip install vaderSentiment

#Import libraries
import re
import nltk
import requests
import numpy as np
import pandas as pd
import seaborn as sns
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from textblob import TextBlob
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from strip_tags import strip_tags
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# IMDb Movie review examples
# Deadpool Movie: https://www.imdb.com/title/tt1431045/reviews
# Borderlands Movie: https://www.imdb.com/title/tt4978420/reviews/?ref_=tt_ov_rt
# Saving Bikini Bottom Movie: https://www.imdb.com/title/tt23063732/reviews/?ref_=tt_ov_rt
# Shawshank Redemption: https://www.imdb.com/title/tt0111161/reviews/?ref_=tt_ov_rt (Best rated Movie)
# Disaster Movie: https://www.imdb.com/title/tt1213644/reviews/?ref_=tt_ov_rt (Worst rated Movie)

# Request URL and extract content from IMDb page
r = requests.get('https://www.imdb.com/title/tt2850386/reviews/?ref_=tt_ov_rt')  
c = r.content  
soup = BeautifulSoup(c)

#Find element of the webpage for reviews
main_content = soup.find_all('div', attrs = {'class': 'text show-more__control'})

main_content 

# Find the date element of the webpage
date = soup.find_all('span', attrs={'class': 'review-date'})
date

# Find rating element of the webpage
rating = soup.find_all('span', attrs={'class': 'rating-other-user-rating'})
rating


# Filter out stop words
STOPWORDS = " ".join([content.get_text() for content in main_content])
word_tokens = word_tokenize(STOPWORDS)
STOPWORDS = set(stopwords.words('english'))
filtered_words = [word for word in word_tokens if word.lower() not in STOPWORDS]

# Filter out other words that bog down sentiment representation
custom_stop_words = ['movie', 'film', 'even', 'people', 'well', 'saw', 'enough', 'never', 'make', 'different', 'character', 'films', 'movies', 'come', 'watched', 'time', 'want', 'big', 'sense', 'audience', 'role', 'writing', 'performance', 'really', "n't", 'game', 'Game','Black']

stop_words = STOPWORDS.union(custom_stop_words)

word_sentiments = {word: TextBlob(word).sentiment.polarity for word in filtered_words}

# Used Strip-tags to clean the unecessary html tags
stripped = [strip_tags(str(content), ["div"], minify=True) for content in main_content]

# Strip rating of unecessary html tags
clean_rating = [strip_tags(str(content), ["span"], minify=True) for content in rating]

# Strip date of unecessary html tags
clean_date = [strip_tags(str(content), ["span"], minify=True) for content in date]

# Join the list into a single string
text = ' '.join(stripped)  
words = text.split()

#Function to determine color of words
def sentiment_to_color(sentiment):
    if sentiment > 0:
        return 'green'  # Positive
    elif sentiment < 0:
        return 'red'    # Negative
    else:
        return 'black'   # Neutral

def color_func(word, **kwargs):
    sentiment = word_sentiments.get(word, 0)
    return sentiment_to_color(sentiment)

# Generate the word cloud with color function
filtered_text = ' '.join(filtered_words)
wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=color_func, stopwords=stop_words).generate(filtered_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Positive vs Negative Comparison

positive_words = [word for word in words if word_sentiments.get(word, 0) > 0]
negative_words = [word for word in words if word_sentiments.get(word, 0) < 0]

positive_wordcloud = WordCloud(width=400, height=400, background_color='white', color_func=lambda *args, **kwargs: 'green').generate(' '.join(positive_words))
negative_wordcloud = WordCloud(width=400, height=400, background_color='white', color_func=lambda *args, **kwargs: 'red').generate(' '.join(negative_words))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(positive_wordcloud, interpolation='bilinear')
axes[0].set_title('Positive Sentiment', fontsize=12)
axes[0].axis('off')

axes[1].imshow(negative_wordcloud, interpolation='bilinear')
axes[1].set_title('Negative Sentiment', fontsize=12)
axes[1].axis('off')

plt.show()

# Join the list into a single string
text = ' '.join(stripped)  
words = text.split()

# Function to determine color of words
def sentiment_to_color(sentiment):
    if sentiment > 0:
        return 'green'  # Positive
    elif sentiment < 0:
        return 'red'    # Negative
    else:
        return 'black'   # Neutral

def color_func(word, **kwargs):
    sentiment = word_sentiments.get(word, 0)
    return sentiment_to_color(sentiment)

# Filter words to include only positive and negative sentiments
positive_negative_words = [word for word in filtered_words if word_sentiments.get(word, 0) != 0]

# Generate the word cloud with color function
filtered_cloud = ' '.join(positive_negative_words)
wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=color_func, stopwords=stop_words).generate(filtered_cloud)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Extract the numerical rating values
numerical_ratings = []
for rating in clean_rating:
    try:
        numerical_ratings.append(int(rating.split('/')[0].strip()))
    except ValueError:
        continue

# Count the number of different ratings on the page
rating_counts = Counter(numerical_ratings)

# Plot the ratings on a bar graph
ratings = list(rating_counts.keys())
counts = list(rating_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(ratings, counts, color='gold')
plt.xlabel('Rating')
plt.ylabel('Viewers')
plt.title('Star Rating out of Ten')
plt.xticks(ratings)
plt.show()

# Calculate the average rating
if numerical_ratings:
    average_rating = (sum(numerical_ratings) / len(numerical_ratings))
    result_summary = f"The average rating of this movie is {average_rating:.2f}"

result_summary
