#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option("display.max_columns" , None)
pd.set_option("display.float_format" , lambda x : "%.4f" %x)
pd.set_option("display.width", 200)


# In[3]:


################################################ #
# Rating Products
################################################ #

##########################################
# Application: User and Time Weighted Course Score Calculation
##########################################


# In[4]:


path = "/Users/gokhanersoz/Desktop/VBO_Dataset/course_reviews.csv"
course_reviews = pd.read_csv(path)


# In[5]:


df = course_reviews.copy()
df.head()


# In[8]:


# What stands out here is that timestamp is classified as object...

df.dtypes


# In[9]:


df.describe([.01, .99]).T


# In[157]:


def boxplot(dataframe,num_cols):
    
    i=1
    size = 15
    num = len(num_cols)
    plt.figure(figsize = (15,7))
    
    for col in num_cols:
        plt.subplot(1,num,i)
        
        sns.boxplot(dataframe[col])
        plt.title(f"{col.upper()} Outliers", fontsize = size)
        plt.xlabel(f"{col}",fontsize = size)
        plt.ylabel("Values" ,fontsize = size)
        plt.tight_layout(pad = 4)
        i+=1
        
    plt.show()


# In[24]:


num_cols = [col for col in df.columns if df[col].dtype != "object"]
boxplot(df,num_cols)


# In[28]:


# Rating Distribution
rating = pd.DataFrame(df["Rating"].value_counts()).reset_index()
rating.columns = ["Rating","Values"]
rating


# In[30]:


# Questions Asked Distribution

answered = pd.DataFrame(df["Questions Answered"].value_counts()).reset_index()
answered.columns = ["Questions Answered Values", "Values Counts"]
answered


# In[33]:


# The score given in the breakdown of the question asked

df.groupby("Questions Answered").agg({"Questions Asked" : "count" , "Rating" : "mean"})


# In[34]:


####################
# average
####################

# Average Score

# Is it the real average?

mean = df.Rating.mean()

# It may have lost the trend !!! Time can have an effect!!!!
# I want to catch the current trend ...

print("Rating Mean : {}".format(mean))


# In[35]:


####################
# Time-Based Weighted Average
####################

# Weighted Average by Point Times !!!!!
# Get weights based on time intervals


# In[36]:


# Now it was an object when we examined it above, now we will convert it to a time variable
# TimeStamp last logged in

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Timestamp"].dtype


# In[37]:


print("Max Time : {}".format(df["Timestamp"].max()))
print("Min Time : {}".format(df["Timestamp"].min()))


# In[38]:


current_date = pd.to_datetime("2021-02-10")
current_date


# In[39]:


df.head()


# In[40]:


import datetime as dt
df["days"] = (current_date - df["Timestamp"]).dt.days
df.sort_values(by = "days")


# In[82]:


def time_based_weighted_average(data , w1 = 28 , w2 = 26 , w3 = 24 , w4 = 22):
    
    results =     data.loc[ data["days"] <= 30 , "Rating"].mean() * w1 / 100 +     data.loc[ (data["days"] > 30) & (data["days"] <= 90) , "Rating"].mean() * w2/ 100 +     data.loc[ (data["days"] >90) & (data["days"] <=180) , "Rating"].mean() * w3 / 100 +     data.loc[ (data["days"] > 180) , "Rating"].mean() * w4 / 100
    
    return results


# In[75]:


print("By Time Average : {}".format(time_based_weigted_average(df)))


# In[51]:


print("Progress Max : {}".format(df.Progress.max()))
print("Progress Min : {}".format(df.Progress.min()))


# In[66]:


# Can the average of those who make 0 progress be equal to the average of those who make 100 progress?

df.groupby("Progress").agg({"Progress" : "count" , "Rating" : "mean"})


# In[67]:


####################
# User-Based Weighted Average
####################

# Weighted Average by User Quality


# In[70]:


def user_based_weighted_average(data , w1 = 22 , w2 = 24, w3 = 26, w4 = 28):
    
    results =         data.loc[ data["Progress"] <= 10 ,"Rating"].mean() * w1 / 100 +         data.loc[ (data["Progress"] > 10) & (data["Progress"] <= 45), "Rating"].mean() * w2/100 +         data.loc[ (data["Progress"] > 45) & (data["Progress"] <=75), "Rating"].mean() * w3/100 +         data.loc[ data["Progress"] > 75, "Rating"].mean() * w4 / 100
    
    return results


# In[71]:


# Do you think this is average?
df.loc[df["Progress"] <= 10, "Rating"].mean(),df.loc[df["Progress"] > 75 , "Rating"].mean()


# In[72]:


print("User Based Weighted Average : {}".format(user_based_weighted_average(df)))


# In[73]:


####################
# Weighted Rating
####################


# In[83]:


def course_weighted_rating(data, time_w = 50, user_w = 50):
    
    results = user_based_weighted_average(data) * user_w / 100 +               time_based_weighted_average(data) * time_w / 100
    
    return results


# In[85]:


print("Before Weighted Rating : {}".format(df.Rating.mean()))
print("After Weighted Rating : {}".format(course_weighted_rating(df)))


# In[ ]:





# In[86]:


################################################ #
# Sorting Products
################################################ #

################################################ #
# Application: Course Sorting
################################################ #


# In[87]:


path = "/Users/gokhanersoz/Desktop/VBO_Dataset/product_sorting.csv"
product_sorting = pd.read_csv(path)


# In[89]:


df = product_sorting.copy()
print("DataFrame Shape : {}".format(df.shape))


# In[90]:


df.head()


# In[91]:


df.describe([.01, .99]).T


# In[92]:


####################
# Sorting by Rating
####################


# In[93]:


# Did he buy the courses? Or did he give out coupons?
df.sort_values(by = "rating" , ascending = False).head(10)


# In[94]:


####################
# Sorting by Comment Count or Purchase Count
####################


# In[95]:


df.sort_values(by = "commment_count" , ascending = False).head(10)


# In[96]:


df.sort_values(by = "purchase_count" , ascending = False).head(10)


# In[97]:


####################
# Sorting by Rating, Comment and Purchase
####################

# We did these things because someone can dominate someone else, we have to prevent them ....


# In[98]:


from sklearn.preprocessing import MinMaxScaler

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit_transform(df[["purchase_count"]])
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit_transform(df[["commment_count"]])


# In[99]:


df.head()


# In[100]:


def weighted_sorting_score(data, w1 = 32 , w2 = 26 , w3 = 42):
    results = data["comment_count_scaled"]*w1 / 100 +               data["purchase_count_scaled"]*w2 / 100 +               data["rating"]*w3 / 100
    
    return results


# In[102]:


df["weighted_sorting_score"] = weighted_sorting_score(df)
df[["rating","weighted_sorting_score"]].head()


# In[104]:


df.sort_values(by = "weighted_sorting_score", ascending = False).head(10)


# In[123]:


####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

import scipy.stats as st
import math

def bayesian_average_rating(n, confidence=0.95):
    """
    
    Function used to calculate the wilson lower bound score in the N-star rating system.
     parameters
     ----------
     n: list or df
         keeps the frequencies of the scores.
         Example: [2, 40, 56, 12, 90] 2 points of 1, 40 points of 2, ... , 90 of 5 points.
     confidence: float
         confidence interval

    Returns
     -------
     BAR score: float
         BAR or WLB scores

     """

     # return zero if the sum of ratings is zero.
    
    if sum(n) == 0:
        return 0
    
     # unique number of stars. If there is a score from 5 stars, it will be 5.
    
    K = len(n)
    
     # Z-score relative to 0.95.
    
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    
     # total number of ratings.
    
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    
     # Browse star numbers with index information.
     # perform the calculations in the formulation.
    
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    
    return score


# In[124]:


new_df = df.loc[:, df.columns.str.contains("point")]
new_df.head()


# In[125]:


bayesian_average_rating([2,3,4,5,6])


# In[126]:


bayesian_average_rating([6,5,4,3,2])


# In[127]:


bayesian_average_rating([6,5,4,6,3])


# In[128]:


# Did you pay any attention here? The distribution goes from 1 to 5.
# DataFrame may sound different to you, but beware !!!!

df["bar_sorting_score"] = df.apply(lambda x : bayesian_average_rating(x[["1_point",
                                                                         "2_point",
                                                                         "3_point",
                                                                         "4_point",
                                                                         "5_point"]]), axis = 1 )

# Here, for example, we are looking at points 1 and 2 and 0 points, this may be an old one for us because it has no history, it is new
# He did not receive the resistance of his market did not receive every comment and was not evaluated !!!!
# Did he really catch this???


df.sort_values(by = "bar_sorting_score", ascending = False).head(10)


# In[129]:


df.sort_values("weighted_sorting_score", ascending=False).head(10)


# In[130]:


####################
# Hybrid Sorting: BAR Score + Other Factor
####################


# In[133]:


def weighted_sorting_score(data , w1 = 31 , w2 = 26 , w3=42):
    
    results = data["comment_count_scaled"] * w1 / 100 +               data["purchase_count_scaled"] * w2 / 100 +               data["rating"] * w3 / 100
    
    return results


def pointing_sorting_score(data):
    
    results = data.apply(lambda x : bayesian_average_rating(x[["1_point",
                                                               "2_point",
                                                               "3_point",
                                                               "4_point",
                                                               "5_point"]]), axis = 1)
    
    return results


# In[136]:


def hybrid_sorting_score(data , bar_w = 60 , wss_w = 40):
    
    results = pointing_sorting_score(data) * bar_w / 100 +               weighted_sorting_score(data) * wss_w / 100
    
    return results


# In[140]:


df["hybrid_sorting_score"] = hybrid_sorting_score(df)


# In[142]:


df.sort_values(by = "hybrid_sorting_score" , ascending= False) 


# In[143]:


df[df["course_name"].str.contains("Veri Bilimi")].sort_values(by = "hybrid_sorting_score" , ascending = False)


# In[ ]:





# In[144]:


############################################
# SORTING PRODUCTS - CONTINUE
############################################

############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################


# In[146]:


path = "/Users/gokhanersoz/Desktop/VBO_Dataset/movies_metadata.csv"
movies_metadata = pd.read_csv(path, low_memory=False)


# In[147]:


df = movies_metadata.copy()
print("DataFrame Shape : {}".format(df.shape))


# In[148]:


df.head()


# In[149]:


df = df[["title","vote_average","vote_count"]]
df.head()


# In[151]:


print("DataFrame Shape : {}".format(df.shape))
print("Unique Movie Values : {}".format(df.title.nunique()))


# In[152]:


########################
# Sort by Vote Average
########################


# In[153]:


df.describe([.01 , .99]).T


# In[159]:


from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)

num_cols = [col for col in df.columns if df[col].dtype != "object"]
boxplot(df, num_cols)


# In[160]:


df.sort_values(by = "vote_average", ascending = False).head()


# In[161]:


df[["vote_count"]].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T


# In[162]:


# The distribution of the votes is between 2183.82000 and 14075.00000 there are 35 people !!!! For instance ...
# How many movies are there in December we just looked at it !!!!

pd.DataFrame(pd.cut(df["vote_count"] , [10, 25, 50, 70, 80, 90, 95, 99, 100] ).value_counts()).T


# In[163]:


df[df["vote_count"] > 400].sort_values(by = "vote_average", ascending = False).head()


# In[164]:


########################
# vote_average * vote_count
########################

df["average_count_score"] = df["vote_average"] * df["vote_count"]
df.sort_values("average_count_score", ascending = False).head()


# In[165]:


df["vote_count_scaled"] = MinMaxScaler(feature_range=(1,10)).fit_transform(df[["vote_count"]])

df["average_count_score_2"] = df["vote_average"] * df["vote_count_scaled"]

df.sort_values("average_count_score_2", ascending = False).head()


# In[166]:


########################
# weighted_rating
########################

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)


# In[169]:


# Film 1:
r_1 = 8 # vote average
r_2 = 9.5 # vote average
M = 500 # Number of vote min
v = 1000 # Number of votes received
(v / (v + M))* r_1 , (v / (v + M))* r_2


# In[171]:


# Film 2:
# As the number of comments increases, the rating increases....

r = 8
M = 500
v = 3000
v_ = 4000
(v / (v + M))* r , (v_ / (v_ + M))* r


# In[172]:


# Film 1:
# 2. Section :
C = 7
M = 500
v = 1000
(M / (v + M))* C


# In[179]:


C = 9
M = 500
v = 1000
(M / (v + M))* C


# In[180]:


# Film1 :
5.333333333333333 + 2.333333333333333


# In[181]:


def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


# In[182]:


df.sort_values("average_count_score" , ascending = False).head(4)


# In[183]:


C = df["vote_average"].mean()
M = 2500

weighted_rating(8.10000,14075.00000 , M , C)


# In[184]:


weighted_rating(7.40000, 11444.00000, M, C)


# In[185]:


df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)

df.sort_values(by = "weighted_rating" , ascending = False).head()


# In[191]:


# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.



####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

def bayesian_average_rating(n, confidence=0.95):
    """
    The score system used to calculate the final lower bound score.
    parameters
    ----------
    n: list or df
        keeps the frequencies of the scores.
        Example: [2, 40, 56, 12, 90] 2 points of 1, 40 points of 2, ... , 90 of 5 points.
    trust: floating
        confidence interval

    Returns
    -------
    BAR score: float
        BAR or WLB scores

    """

    #If your ratings are dry, it is zero.
    if sum(n) == 0:
        return 0
    # number of defective stars. If there is a score from 5 stars, it will be 5.
    K = len(n)
    # Z-score relative to 0.95.
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    # total number of ratings.
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    # Browse star numbers with index information.
    # Performs the accounts in #.
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


# In[193]:


# The Shawshank Redemption (9.2)
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])


# In[194]:


# The Dark Night (9)
bayesian_average_rating([30345, 7172, 8083, 11429, 23236, 49482, 137745, 354608, 649114, 1034843])


# In[195]:


rating = pd.read_csv("/Users/gokhanersoz/Desktop/VBO_Dataset/imdb_ratings.csv")
rating = rating.iloc[:,1:] 
rating.head()


# In[197]:


# We've seen how close this process comes out...

rating["bar_score"] = rating.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
rating.head(5)


# In[ ]:




