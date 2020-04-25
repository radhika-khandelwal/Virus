#!/usr/bin/env python
# coding: utf-8

# In[1]:


from twitterscraper import query_tweets
import datetime as dt
import pandas as pd


# In[2]:


get_ipython().system('pip install twitterscraper')


# In[3]:


begin_date = dt.date(2020,1,1)
end_date = dt.date(2020,4,23)

limit = 100
lang = "english"

tweets = query_tweets("virus", begindate=begin_date, enddate=end_date, limit=limit, lang=lang)
df = pd.DataFrame(t.__dict__ for t in tweets)


# In[4]:


new_df = df


# In[5]:


new_df = new_df[['username', 'tweet_id','text']]


# In[6]:


new_df.head()


# In[7]:


import nltk
words = set(nltk.corpus.words.words())
new_df['text'].map(lambda sent
                   :" ".join(w for w in nltk.wordpunct_tokenize(sent) if w.lower() in words or not w.isalpha()))


# In[8]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()


# In[9]:


all_words = new_df['text'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = all_words.index.values[2:50],
            y = all_words.values[2:50],
            marker= dict(colorscale='Jet',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Top 50 (Uncleaned) Word frequencies in the training dataset'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')


# In[10]:


from nltk.corpus import stopwords
import string
stop = stopwords.words('english') 


# In[11]:


def stuff(p):
    
    temp = p.split()
    for i in temp:
        if i not in stop and i.isalpha() and len(i)>3:
            return i


# In[12]:


# new_df['text'].apply(lambda p: i for i in p if i not in stop and i.isalpha() and len(i) > 2)
new_df['cleanwords'] = new_df['text'].apply(stuff)
new_df['cleanwords']


# In[13]:


# Storing the first text element as a string
first_text = new_df.cleanwords
print(first_text)
print("="*90)
#print(first_text.split(" "))


# In[14]:


nltk.download('stopwords')


# In[15]:


stopwords = nltk.corpus.stopwords.words('english')
len(stopwords)


# In[16]:


stemmer = nltk.stem.PorterStemmer()


# In[17]:


print("The stemmed form of running is: {}".format(stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer.stem("run")))


# In[18]:


print("The stemmed form of leaves is: {}".format(stemmer.stem("leaves")))


# In[19]:


from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
print("The lemmatized form of leaves is: {}".format(lemm.lemmatize("leaves")))


# In[20]:


nltk.download('wordnet')


# In[21]:


def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer


# In[23]:


lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# In[24]:


new_df['cleanwords'].loc[new_df['cleanwords'].map(lambda p: type(p) is float)]


# In[25]:


new_df.cleanwords = new_df.cleanwords.loc[new_df.cleanwords.map(lambda p:p is not None)]


# In[26]:


cleanwordss = new_df.cleanwords.loc[new_df.cleanwords.map(lambda p:type(p) != float)]


# In[27]:


cleanwordss.map(lambda p: type(p)).value_counts()


# In[28]:


new_df['cleanwords'] = new_df['cleanwords'].dropna()


# In[29]:


text = list(cleanwordss)
# Calling our overwritten Count vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)


# In[30]:


import numpy as np
feature_names = tf_vectorizer.get_feature_names()
count_vec = np.asarray(tf.sum(axis=0)).ravel()
zipped = list(zip(feature_names, count_vec))
x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
# Now I want to extract out on the top 15 and bottom 15 words
Y = np.concatenate([y[0:15], y[-16:-1]])
X = np.concatenate([x[0:15], x[-16:-1]])

# Plotting the Plot.ly plot for the Top 50 word frequencies
data = [go.Bar(
            x = x[0:50],
            y = y[0:50],
            marker= dict(colorscale='Jet',
                         color = y[0:50]
                        ),
            text='Word counts')]

layout = go.Layout(
    title='Top 50 Word frequencies after Preprocessing'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')


# In[31]:


from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=11, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)


# In[32]:


lda.fit(tf)


# In[33]:


n_top_words = 40
print("\nTopics in LDA model: ")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# In[34]:


first_topic = lda.components_[0]
second_topic = lda.components_[1]
third_topic = lda.components_[2]
fourth_topic = lda.components_[3]


# In[35]:


first_topic.shape


# In[36]:


first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]
second_topic_words = [tf_feature_names[i] for i in second_topic.argsort()[:-50 - 1 :-1]]
third_topic_words = [tf_feature_names[i] for i in third_topic.argsort()[:-50 - 1 :-1]]
fourth_topic_words = [tf_feature_names[i] for i in fourth_topic.argsort()[:-50 - 1 :-1]]


# In[37]:


get_ipython().system('pip install wordcloud')


# In[40]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

firstcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(first_topic_words))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()


# In[39]:


cloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(second_topic_words))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[41]:


cloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(third_topic_words))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[42]:


cloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(fourth_topic_words))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# #### Let's take a look at the bigram

# In[43]:


bigrams = nltk.bigrams(cleanwordss)


# In[44]:


from collections import Counter

counter = Counter(bigrams)
print(counter.most_common(10))


# In[ ]:




