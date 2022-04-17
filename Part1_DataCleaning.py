#!/usr/bin/env python
# coding: utf-8

# # Assessment 1

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot


# In[2]:


gb_cleaning = pd.read_csv('GBmatrixc.csv')
sb_cleaning = pd.read_csv('SBmatrixc.csv')


# ## Question 1 - Data Cleaning

# ## 1. For GBMatrixc dataset

# ### 1.1 Checking unnamed column

# #### We want to know th type of bed, so I add a new column called type. Besides, we also want to know the total quantity of each part.

# In[3]:


# Create new column 'type' for GB and SB.
gb_cleaning['type'] = 'Guest_beds'
sb_cleaning['type'] = 'Single_beds'


# In[4]:


gb_cleaning.head()


# In[5]:


gb_cleaning.tail()


# -  Rename column 'Unnamed: 0' as 'Part_No' because it shows the part codes for each component.
# -  For column 'Unamed:9', I assume to rename this column as'GB_unamed_part' because all its value is 1, meaning this part might be important that will be used in all part number.
# 

# In[6]:


# Rename columns
gb_cleaning.rename(columns = {'Unnamed: 0':'Part_No', 'Unnamed: 9':'GB_unnamed_part'},inplace=True)


# In[7]:


gb_cleaning.columns


# ### 1.2 Change data type

# In[8]:


gb_cleaning.info() # Find that 'UT_ER_299_ ' is object data type.


# In[9]:


# Transform dtype to float64, for 'BRIMNES_329_22','UT_ER_299_ ','Unnamed: 9'columns.
gb_cleaning['BRIMNES_329_22'] = gb_cleaning['BRIMNES_329_22'].astype('float')
gb_cleaning['UT_ER_299_'] = gb_cleaning['UT_ER_299_'].astype('float')
gb_cleaning['GB_unnamed_part'] = gb_cleaning['GB_unnamed_part'].astype('float')


# In[10]:


gb_cleaning.dtypes


# In[11]:


gb_cleaning.describe()


# In[12]:


# There are 78 rows and 11 columns in GB dataset.
gb_cleaning.shape


# ### 1.3 Check duplicates

# In[13]:


# For GBmatrixc: Check duplicates from columns 
gb_cleaning.columns


# In[14]:


# Checking for duplicates in columns using the 'duplicated' method
gb_cleaning.index.duplicated(keep=False)   

# check row duplicated. As all values are False, meaning no duplicated rows.


# In[15]:


gb_cleaning['Part_No'].duplicated().sum()  # there is no duplicated parts.


# In[16]:


gb_cleaning.index.duplicated().sum()  # there is no duplicated index for columns


# #### There are no duplicated rows and index in GBMatrixc dataset, so don't need to drop duplicates.

# ### 1.4 Check missing data and fill NA

# In[17]:


# GBmatrixc: #Fill 0 into NA
gb_cleaning = gb_cleaning.fillna(0)


# In[18]:


gb_cleaning.info()


# In[19]:


gb_cleaning


# ### 1.5 Split multiple part codes
# 
# In the multiple part codes, I assume that each unique 'Part No' need the same amount of quantities. This assumption is to avoid the lack of inventory for the multiple part.

# In[20]:


# Check unique values of 'Part No' columns
gb_cleaning['Part_No'].unique()


# In[21]:


# create a new column 'gb_part' that split '/'
gb_part = gb_cleaning['Part_No'].str.split('/', expand = True)
gb_part


# In[22]:


# Stack splited parts into one new column
gb_part = gb_part.stack()
gb_part


# In[23]:


# Reset index
gb_part = gb_part.reset_index(level = 1, drop=True)
gb_part


# In[24]:


# Modify the dataframe and rename the new column as a new 'Part No'
gb_part = gb_part.to_frame().rename({0:'Part_No'},axis =1)
gb_part


# In[25]:


# Join a new gb_cleaning table. 
# Remove old 'Part No' column in gb_cleaning and join new gb_part which has new 'Part No' column.
gb_cleaning = gb_cleaning.drop(['Part_No'],axis = 1).join(gb_part)


# In[26]:


gb_cleaning


# ### 1.6 Replace punctuation

# In[27]:


# Check the values of 'Part No', I find that there are punctuation.
gb_cleaning['Part_No'].unique()


# In[28]:


# Replace punctuation to '', and delete all space of Part_No.
gb_cleaning['Part_No']= gb_cleaning['Part_No'].str.replace(r'_', '').str.strip()


# In[29]:


gb_cleaning['Part_No'].unique()


# ### 1.7 Change columns

# In[30]:


gb_cleaning.columns


# In[31]:


# Create new column 'quantity' for GB and SB, calculating the total quantity of each Part No.
gb_cleaning['gbquantity'] = np.sum(gb_cleaning, axis = 1)
sb_cleaning['sbquantity'] = np.sum(sb_cleaning, axis = 1)


# In[32]:


# Combine columns which start as the same product names
gb_cleaning['BRIMNES']  = gb_cleaning[list(gb_cleaning.filter(regex='BRIMNES'))].sum(axis=1)
gb_cleaning['FLEKKE'] = gb_cleaning[list(gb_cleaning.filter(regex='FLEKKE'))].sum(axis=1)
gb_cleaning['FYRESDAL'] = gb_cleaning[list(gb_cleaning.filter(regex='FYRESDAL'))].sum(axis=1)
gb_cleaning['HEMNES'] = gb_cleaning[list(gb_cleaning.filter(regex='HEMNES'))].sum(axis=1)
gb_cleaning['TARVA'] = gb_cleaning[list(gb_cleaning.filter(regex='TARVA'))].sum(axis=1)
gb_cleaning['UT'] = gb_cleaning[list(gb_cleaning.filter(regex='UT'))].sum(axis=1)


# In[33]:


gb_cleaning.head()


# In[34]:


# Set Part number as index.

gb_cleaning.set_index('Part_No', inplace = False)

# In the last column 'Unnamed:9', I would keep this column because it has values 1.


# In[35]:


gb_cleaning.info()


# ### 1.8 Outliers

# In[36]:


# Detect outliers
for column in gb_cleaning.columns:
    print(column)
    print(gb_cleaning[column].value_counts())
    print('')


# #### Find outliers using z-scores
# Calculating mean and standard deviation for each column.

# In[37]:


# Calculate mean & std for quantity

mean = np.mean(gb_cleaning.gbquantity)
std = np.std(gb_cleaning.gbquantity)
print('mean of guest beds\' parts quantity is ', mean)
print('std. deviation of guest beds\' is ', std)



# Calculate Z score. If Z score > 3, print it as an outlier

threshold = 3
outlier =[]
for value in gb_cleaning.gbquantity:
    z = (value - mean)/std
    if z > threshold:
        outlier.append(value)
print('outlier in guest beds quantity is ', outlier)


# In[38]:


# Apply function in each row.

def is_outlier(value, mean=1, std=1, threshold=1):
    threshold=3
    z = (value - mean)/std
    return z > threshold

gb_cleaning[gb_cleaning['gbquantity'].apply(is_outlier, mean=mean, std=std, threshold=threshold)]


# ####  From the table above, it shows there are four outliers in 'quantity'. Quantity refers to the total quantity that is used in each part of guest beds. I assume that these four components are highly frequently used parts and thus key parts. So I will keep these outliers and consider them as key parts of guest beds.

# In[39]:


gb_cleaning.to_csv('GB_cleaning.csv')


# In[ ]:





# # - Data Cleaning
# ## 2. For SBMatrixc dataset

# ### 2.1 Checking unnamed column

# In[40]:


# For column 'Unnamed: 0', I want to change its name as 'Part_No' because it shows part codes for each single bed's component.
sb_cleaning.rename(columns = {'Unnamed: 0':'Part_No'}, inplace=True)


# In[41]:


sb_cleaning.columns


# In[42]:


sb_cleaning.head()


# In[43]:


# Combine columns which start as the same product names
sb_cleaning['FJELLSE']  = sb_cleaning[list(sb_cleaning.filter(regex='FJELLSE'))].sum(axis=1)
sb_cleaning['HEMNES']  = sb_cleaning[list(sb_cleaning.filter(regex='HEMNES'))].sum(axis=1)
sb_cleaning['MALM']  = sb_cleaning[list(sb_cleaning.filter(regex='MALM'))].sum(axis=1)
sb_cleaning['NORDLI']  = sb_cleaning[list(sb_cleaning.filter(regex='NORDLI'))].sum(axis=1)
sb_cleaning['TARVA']  = sb_cleaning[list(sb_cleaning.filter(regex='TARVA'))].sum(axis=1)


# ### 2.2 Change data type

# In[44]:


sb_cleaning.dtypes


# ## Investigating data

# In[45]:


sb_cleaning.describe()


# In[46]:


sb_cleaning.info()  #find there is a "Nan" column key with all 'Nan' values, which will be dealt with in later sector.


# In[47]:


# There are 50 rows and 10 columns in SB dataset.
sb_cleaning.shape


# ## 2.3 Check duplicates

# In[48]:


# For SBmatrixc: check duplicates from columns
sb_cleaning.columns


# In[49]:


# Check row duplicated. As all values are False, meaning no duplicated rows.
sb_cleaning.index.duplicated(keep=False)


# In[50]:


# Calculate the toal quantity of dulicates for rows and columns
sb_cleaning.duplicated().sum()  # No duplicates in rows


# In[51]:


sb_cleaning.index.duplicated().sum()  # No duplicates in columns


# #### As there are no duplicates in SBmatrixc dataset, we don't need to drop duplicates.

# ## 2.4 Check missing data and fill NA

# In[52]:


# Fill 0 into NA
sb_cleaning = sb_cleaning.fillna(0)


# In[53]:


sb_cleaning.info()


# In[54]:


# Because there is a "Nan" column, check unique values of column "Nan"
Nan = sb_cleaning.drop_duplicates(['Nan']) # In column "Nan", all values are "Nan"
Nan


# In[55]:


# Since there are no values in "Nan" column, delete column "Nan"
sb_cleaning = sb_cleaning.drop(columns = ['Nan'])
sb_cleaning


# In[56]:


sb_cleaning.shape 


# In[57]:


# From the table, find there are "Nan", replace "Nan" to 0.
sb_cleaning = sb_cleaning.replace('Nan', 0)
sb_cleaning


# ## 2.5 Split multiple part codes

# In the multiple part codes, I assume that each unique Part_No need the same amount of quantities. This assumption is to avoid the lack of inventory for those overlaopped components. So sufficient inventory level for these parts should be considered as priority.

# In[58]:


# Check unique values of 'Part_No' columns
sb_cleaning['Part_No'].unique()


# From the columns name above, we can see that there are '/' that need to be splitted.
# 

# In[59]:


sb_part = sb_cleaning['Part_No'].str.split('/', expand=True)
sb_part


# In[60]:


# Stack splited parts into one new column
sb_part = sb_part.stack()
sb_part


# In[61]:


# Reset index
sb_part = sb_part.reset_index(level=1, drop=True)
sb_part


# In[62]:


# Modify the dataframe and rename the new column as a new 'Part_No'
sb_part = sb_part.to_frame().rename({0:'Part_No'}, axis=1)
sb_part


# In[63]:


# Join a new sb_cleaning table by removing old 'Part_No' and add new 'sb_part'.
sb_cleaning = sb_cleaning.drop(['Part_No'], axis=1).join(sb_part)
sb_cleaning


# In[64]:


# In the new sb_cleaning dataset, there are 56 rows in new dataset, instead of 50 rows in raw dataset.
sb_cleaning.shape


# ## 2.6 Replace punctuation

# In[65]:


# Check the values of 'Part_No', I find that there are punctuation '_' and '*'.
sb_cleaning['Part_No'].unique()


# In[66]:


# Replace punctuation to '', and delete all space of Part_No.
sb_cleaning['Part_No'] = sb_cleaning['Part_No'].replace(r'[_*]','',regex=True).str.strip()


# In[67]:


sb_cleaning['Part_No'].unique()


# In[68]:


sb_cleaning.dtypes


# ## 2.7 Check outliers

# #### Find outliers using z-scores
# Calculating mean and standard deviation for each column.

# In[69]:


# Calculate mean and std for quantity
sbmean = np.mean(sb_cleaning.sbquantity)
sbstd = np.std(sb_cleaning.sbquantity)
print('mean of single beds\' parts quantity is ', sbmean)
print('std. deviation of single beds\' is ', sbstd)



# Calculate Z score. If Z score > 3, print it as an outlier

threshold = 3
outlier =[]
for value in sb_cleaning.sbquantity:
    z = (value - sbmean)/sbstd
    if z > threshold:
        outlier.append(value)
print('outlier in single beds quantity is ', outlier)


# #### From the result above, we can see that there are no outliers in single bed's component quantity.

# In[70]:


# Here is the final cleaning SB dataset.
sb_cleaning


# In[71]:


# Save cleaned SB dataset
sb_cleaning.to_csv('SB_cleaning.csv')


# In[ ]:





# In[ ]:




