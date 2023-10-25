#!/usr/bin/env python
# coding: utf-8

# # Q7.Calculate Mean, Median, Mode, Variance, Standard Deviation, Range &     comment about the values / draw inferences, for the given dataset. For Points, Score, Weigh. Find Mean, Median, Mode, Variance, Standard Deviation, and Range and also Comment about the values/ Draw some inferences.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


Q7=pd.read_csv(r"C:\Users\birad\Downloads\Q7 (1).csv")
Q7.head()


# # Median

# In[4]:


data=Q7[['Points','Score','Weigh']]
print(data.median())


# # Mean

# In[26]:


data=Q7[['Points','Score','Weigh']]
print(data.mean())


# # Mode

# In[5]:


data=Q7[['Points','Score','Weigh']]
print(data.mode())


# # Standard Deviation

# In[6]:


data=Q7[['Points','Score','Weigh']]
print(data.std())


# # variance

# In[29]:


data=Q7[['Points','Score','Weigh']]
print(data.var())


# # Range

# In[30]:


data=Q7[['Points','Score','Weigh']]
print(data.max()-data.min()


# In[7]:


f,ax=plt.subplots(figsize=(15,5))
plt.subplot(1,3,1)
plt.boxplot(Q7.Points)
plt.title('Points')
plt.subplot(1,3,2)
plt.boxplot(Q7.Score)
plt.title('Score')
plt.subplot(1,3,3)
plt.boxplot(Q7.Weigh)
plt.title('Weigh')
plt.show()


# # Q9)_(a) Calculate Skewness, Kurtosis & draw inferences on the following data(Cars speed and distance )
# 

# In[8]:


Q9=pd.read_csv(r"C:\Users\birad\Downloads\Q9_a (1).csv")
Q9.head()


# In[9]:


Q9.skew()


# In[10]:


Q9.kurt()


# In[ ]:


##Inferences
1. The skewness is **negative** then the distribution of the **speed** data distribution is slightly **left skewed** so the tail is extended to the left.
2. The skewness is **positive** for **dist** variable so the it has outliers at the right and the tail is extended to the right..
3. The kurtosis of **speed** is Negative,It means that the data has lighter tails and is less peaked than a normal distribution (kurtosis of 3).
4. The kurtosis of **dist** is Positive,It means that the data has lighter tails and is less peaked than a normal distribution (kurtosis of 3).


# # Q9_(b) Calculate Skewness, Kurtosis & draw inferences on the following data(SP and Weight(WT) )

# In[11]:


Q9b=pd.read_csv(r"C:\Users\birad\Downloads\Q9_b (1).csv")
Q9b.head()


# In[45]:


Skewness=Q9b[['SP','WT']].skew()
Skewness


# In[46]:


Kurtosis=Q9b[['SP','WT']].kurtosis()
Kurtosis


# In[ ]:


##Inferences
1. The column SP has positive skewness so the tail would be at right and have the oultiers in the right
2. The column SP has positive kurtosis, It means that the data has lighter tails and is less peaked than a normal distribution (kurtosis of 3)
3. The column WT has negative Skewness so the tail would be at left and has the outlier in the left
4. The column WT has positive kurtosis, It means that the data has lighter tails and is less peaked than a normal distribution (kurtosis of 3)


# # Q 20) Calculate probability from the given dataset for the below cases
# 
# Data _set: Cars.csv
# 
# Calculate the probability of MPG  of Cars for the below cases.
# MPG <- Cars$MPG
# 
# a.P(MPG>38)
# 
# b.P(MPG<40)
# 
# c.P (20<MPG<50)
# 
# 

# In[14]:


Q20=pd.read_csv(r"C:\Users\birad\Downloads\Cars.csv")
Q20.head()


# # a.P(MPG>38)

# In[15]:


data1=Q20['MPG']
data1.head()


# In[16]:


mean=data1.mean()
stdv=data1.std()
print(mean)
print(stdv)


# In[17]:


from scipy import stats
prob1=1-stats.norm.cdf(38,mean,stdv)
print(f'probability of MPG>38 is = {prob1}')


# # b.P(MPG<40)

# In[18]:


prob2 = stats.norm.cdf(40,mean,stdv)
print(f"Probability of MPG<40 is {prob2}")


# # c.P (20<MPG<50)

# In[19]:


d = stats.norm.cdf(50, mean, stdv) - stats.norm.cdf(20, mean, stdv)
print(f"The probability of lying between 20 and 50 is {d}")


# In[20]:


import seaborn as sns
sns.boxplot(Q20.MPG)


# # Q 21) Check whether the data follows normal distribution
# a)Check whether the MPG of Cars follows Normal Distribution 
# 

# In[70]:


data1.head()


# In[71]:


plt.plot(data1)
plt.title('Line chart for single column')
plt.xlabel('Index or Time')
plt.ylabel('Column values')
plt.show()


# In[73]:


plt.hist(data1,bins=20,edgecolor='black')


# In[74]:


stats.probplot(data1, dist="norm", plot=plt)


# # b)Check Whether the Adipose Tissue (AT) and Waist Circumference(Waist)  from wc-at data set  follows Normal Distribution 
#      
# 
# 

# In[21]:


Q21b=pd.read_csv(r"C:\Users\birad\Downloads\wc-at.csv")
Q21b.head()


# In[22]:


sns.distplot(Q21b.Waist)
plt.ylabel('density')


# In[23]:


sns.distplot(Q21b.AT)
plt.ylabel('density')


# In[24]:


Q21b.Waist.mean(),Q21b.Waist.median()


# In[25]:


Q21b.AT.mean(),Q21b.AT.median()


# In[ ]:




