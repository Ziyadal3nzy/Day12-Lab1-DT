#!/usr/bin/env python
# coding: utf-8

# |  Column name  |  Description  |
# | ----- | ------- |
# | Num_posts        | Number of total posts that the user has ever posted   |
# | Num_following    | Number of following                                   |
# | Num_followers    | Number of followers                                   |
# | Biography_length | Length (number of characters) of the user's biography |
# | Picture_availability | Value 0 if the user has no profile picture, or 1 if has |
# | Link_availability| Value 0 if the user has no external URL, or 1 if has |
# | Average_caption_length | The average number of character of captions in media |
# | Caption_zero     | Percentage (0.0 to 1.0) of captions that has almost zero (<=3) length |
# | Non_image_percentage | Percentage (0.0 to 1.0) of non-image media. There are three types of media on an Instagram post, i.e. image, video, carousel
# | Engagement_rate_like | Engagement rate (ER) is commonly defined as (num likes) divide by (num media) divide by (num followers)
# | Engagement_rate_comment | Similar to ER like, but it is for comments |
# | Location_tag_percentage | Percentage (0.0 to 1.0) of posts tagged with location |
# | Average_hashtag_count   | Average number of hashtags used in a post |
# | Promotional_keywords | Average use of promotional keywords in hashtag, i.e. regrann, contest, repost, giveaway, mention, share, give away, quiz |
# | Followers_keywords | Average use of followers hunter keywords in hashtag, i.e. follow, like, folback, follback, f4f|
# | Cosine_similarity  | Average cosine similarity of between all pair of two posts a user has |
# | Post_interval      | Average interval between posts (in hours) |
# | real_fake          | r (real/authentic user), f (fake user/bought followers) |

# # Q1: Import labraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Q2: Read instagram_users.csv file

# In[2]:


df = pd.read_csv('instagram_users.csv')


# # Q3: Split tha dataset into training and testing

# In[3]:


from sklearn.model_selection import train_test_split


# In[10]:


df.info()


# In[9]:


X = df.drop('real_fake',axis=1)
y = df['real_fake']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# # Q4: Build three machine models 

# ## Q4.1: The first machine model
# - Print the model's name.
# - Print the model's accuracy.
# - Print the model's confusion matrix.

# In[12]:


from sklearn.tree import DecisionTreeClassifier


# In[13]:


dtree = DecisionTreeClassifier()


# In[32]:


print(accuracy_score(y_test,predictions))


# In[33]:


print(confusion_matrix(y_test,predictions))


# In[14]:


dtree.fit(X_train,y_train)


# ## Q4.2: The second machine model
# - Print the model's name.
# - Print the model's accuracy.
# - Print the model's confusion matrix.

# In[15]:


predictions = dtree.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# In[17]:


print(classification_report(y_test,predictions))


# In[18]:


print(accuracy_score(y_test,predictions))


# In[19]:


print(confusion_matrix(y_test,predictions))


# In[20]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(dtree, X_test, y_test)  
plt.show()


# ## Q4.3: The third machine model
# - Print the model's name.
# - Print the model's accuracy.
# - Print the model's confusion matrix.

# In[22]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[23]:


rfc_pred = rfc.predict(X_test)


# In[24]:


print(confusion_matrix(y_test,rfc_pred))


# In[25]:


plot_confusion_matrix(rfc, X_test, y_test)  
plt.show()


# In[26]:


print(classification_report(y_test,rfc_pred))


# In[27]:


print(accuracy_score(y_test,predictions))

