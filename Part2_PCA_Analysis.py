#!/usr/bin/env python
# coding: utf-8

# # Question 2 - Data Merge

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


sb_cleaning = pd.read_csv('SB_cleaning.csv')  # Read cleaned SB dataset
gb_cleaning = pd.read_csv('GB_cleaning.csv')  # Read cleaned GB dataset


# In[3]:


sb_cleaning.info()


# - As I will do the analysis of each type's part number in their quantity usage, I will select Part_No, type, and quantity, and combined product names from two dataset. My data merge will based on these variables.

# In[4]:


# Select combined products from SB dataset.
sb = sb_cleaning[['Part_No','type','sbquantity', 'FJELLSE', 'HEMNES', 'MALM', 'NORDLI', 'TARVA']]
sb


# In[5]:


gb_cleaning.info()


# In[6]:


# Select combined products from GB dataset.
gb = gb_cleaning[['Part_No', 'type', 'gbquantity','BRIMNES', 'FLEKKE', 'FYRESDAL', 'HEMNES', 'TARVA', 'UT']]
gb


# ## 2.1 Combine SB and GB datasets

# #### As I consider the identical parts for two bed types, I will not use join function because it will replace the quantity from one type. Instead, I will use concat to combine two datasets. 
# 
# #### In this part, the analysis will based on:
# - Identical part No. and their ranking based on total quantity usage.
# - All parts and their ranking based on total quantity usage.

# In[7]:


# Combine gb and sb dataset, and reset index values.
sgb = pd.concat([sb,gb],ignore_index=True).fillna(0)
sgb


# In[8]:


sgb.columns


# In[9]:


# Based on part number, check duplicates of components.
sgb['Part_No'].duplicated().sum()  


# There are 24 duplicates components, meaning these part No. components are both used in two bed types. Thus, I will analyse these identical part No. in the next sector.

# ## 2.2 Merge Identical part No. are handled by two types of beds

# In[10]:


# Use inner join to check the identical parts are handled by both two types.
identical = pd.merge(gb,sb, how='inner', on = 'Part_No')
identical


# Check datatype os identical dataset, and change dtype

# In[11]:


# Check and change datatype
identical.dtypes


# In[12]:


# Change Part_No's data type
identical['Part_No'] = identical['Part_No'].astype('object')
identical.info()


# ##### From table above
#  - We can see there are 24 identical parts, and their usage quantity in each bed type. In order to analyse which identical parts are most popular, I will calculate the sum quantity of each identical parts.
# 
#  - As these identical parts appear in both single and guest beds, I will change their type name as 'Guest_Single_beds'.
#  
#  - There are two products appear in both bed types, HEMNES and TARVA. So we also need to calculate the sum quantity of these two products.

# In[13]:


# Calculate the sum quantity uage of identical parts
identical['quantity'] = identical['sbquantity'] + identical['gbquantity']

# Create new 'type' value for identical parts, named as 'Guests_Single_beds'
identical['type'] = 'Guest_Single_beds'

# Calculate identical quantity of HEMNES and TARVA.
identical['TARVA'] = identical[list(identical.filter(regex='TARVA'))].sum(axis=1)
identical['HEMNES'] = identical[list(identical.filter(regex='HEMNES'))].sum(axis=1)


# In[14]:


# Select combined parts from old 'identical' dataset
identical = identical[['Part_No', 'type', 'quantity','FJELLSE', 'HEMNES', 'MALM', 'NORDLI',
       'TARVA', 'BRIMNES', 'FLEKKE', 'FYRESDAL', 'UT']]


# In[15]:


identical


# ## 2.3 Rank the identical components (based on Part No.) by the quantity of their use.

# In[16]:


# Rank quantity of identical parts
identical.sort_values(by=['quantity'], ascending = False).set_index('quantity')


# From the table above, we can see that there components number No.112996 and No.118331 are highest usage in both two bed types, with 129 quantity in total. In addition, No.101350 is also highly frequency usage with 109 total quantity. 
# 
# On the contrary, there are six parts (No.104875, 111451, 100001, 100049, 113453,100089) are least used in both two types, below 10 quantity usage in total.

# ## 2.4 Combine all components by the total quantity of their use.

# #### Merge new sgb_cleaning dataset that only shows all non-duplicated values
# 
# - Drop duplicate rows from old 'sgb' dataset, and combine the 'identical' dataset into a new sgb_cleaning dataset. In new sgb_cleaning dataset, previous duplicate/identical parts' type is 'Guest_Single_beds', and their quantity is the total usage quantity in each product series.
# 

# In[17]:


# Drop all duplicates part numbers, and create dataset for unique components
sgb_uni = sgb.drop_duplicates('Part_No', keep=False)

# Calculate the total quantity 
sgb_uni['quantity'] = sgb_uni['sbquantity'] + sgb_uni['gbquantity']

sgb_uni = sgb_uni[['Part_No', 'type', 'quantity','FJELLSE', 'HEMNES', 'MALM', 'NORDLI',
       'TARVA', 'BRIMNES', 'FLEKKE', 'FYRESDAL', 'UT']] # this dataframe is all 89 unique parts.


# In[18]:


sgb_uni # this dataframe is all 89 unique parts.


# In[19]:


sgb_uni.info()


# In[20]:


# Combine identical dataset(sub) with dropped datasets(sgb_uni)
sgb_cleaning = pd.concat([sgb_uni,identical],ignore_index=True) 

sgb_cleaning


# In[21]:


sgb_cleaning.info()


# In[22]:


# From the table, we can see that No.112996 and No.119030 all value are correct.
sgb_cleaning.tail()


# ## 2.5 Rank total quantity of parts.

# In[23]:


# Sort values of total quantity
sgb_sorted = sgb_cleaning.sort_values(by=['quantity'], ascending = False)
sgb_sorted


# #### From the table above, we can see that the highest frequency use is Part No.112996 and No.118331, both 129 total usage quantity. And top 3 usage quantity are all 'guest_single_beds', meaning top 3 components are used in both bed types. Following top 3, rank 4 and 5 are used over 100 quantities, both are unique guest beds. 
# 
# #### On the contrary, the lowest usage quantity appears in single bed type, with only 1 quantity in No.100092 and No.100092. And No.111631, No.151641, and 100027 are belong to unique guest beds, with very low usage as 2 quantity. 

# # Question 3 Data Analysis

# ## Calculate the percentage of unique components in each bed type
# As each component has different quantity, I assume higher quantity means higher weight. Thus, I will calculate the percentage based on the total quantity of unique components in each bed type. 

# In[24]:


# Split unique single bed
unisingle = sgb_cleaning[sgb_cleaning['type'] == 'Single_beds']
unisingle


# In[25]:


# Calculate the total quantity of unique components of single beds.
unisingle = unisingle['quantity'].sum()


# In[26]:


unisingle # There are 292 unique components in total usage quantity of single beds.


# From the previous calculation, we have got identical dataset. So we only need to calculate the total quantity of 
# identical components.

# In[27]:


iden = identical['quantity'].sum()


# In[28]:


iden # There are 1002 components in total quantity that are used in both single and guest beds.


# In[29]:


# Calculate the percentage of unique components in single bed, and keep two decimal places using round()
single_per = round(unisingle / (unisingle + iden) * 100, 2) 
print('The percentage of unique components quantity in single bed is:', single_per, '%')


# ## For single beds, the percentage of unique components (in terms of their quantity usage) is: 
# ##       22.57 %

# In[30]:


# Split unique guest bed
uniguest = sgb_cleaning[sgb_cleaning['type'] == 'Guest_beds']

# Calculate the total quantity of unique components of guest beds.
uniguest = uniguest['quantity'].sum()


# In[31]:


uniguest # There are 848 unique components in total usage quantity of guest beds.


# In[32]:


# Calculate the percentage of unique components in single bed, and keep two decimal places using round()
guest_per = round(uniguest / (uniguest + iden) * 100, 2) 
print('The percentage of unique components quantity in single bed is:', guest_per, '%')


# ## For guest beds, the percentage of unique components (in terms of their quantity usage) is:
# ## 45.84 %

# # Question 4 Data Discovery

# ## - PCA Project to 2D

# In[33]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[34]:


sgb_cleaning.head()


# #### Name columns and normalise values in [0,1]

# In[35]:


sgb_cleaning.columns


# In[36]:


# Define quantity as parameters
features = ['FJELLSE', 'HEMNES', 'MALM', 'NORDLI',
       'TARVA', 'BRIMNES', 'FLEKKE', 'FYRESDAL', 'UT']

x = sgb_cleaning.loc[:,features].values 


# In[37]:


# Define target parameter, type as independent variable
y = sgb_cleaning.loc[:,['type']].values


# In[38]:


# Scale the parameter values
x = StandardScaler().fit_transform(x)


# In[39]:


pd.DataFrame(data = x, columns = features).head()


# ## - PCA Project to 2D

# In[40]:


# Create PCA by projecting the 4D parameters onto a 2D circular
pca = PCA(n_components =2)


# In[41]:


# Scaling the data onte 2D.
principalComponents = pca.fit_transform(x)
principalComponents


# In[42]:


# Extract principle components
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


# In[43]:


principalDf.head()


# In[44]:


sgb_cleaning[['type']]


# In[45]:


finalDf = pd.concat([principalDf, sgb_cleaning[['type']]], axis = 1)
finalDf.head()


# In[46]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title ( '2D component PCA', fontsize = 20)

targets = ['Single_beds', 'Guest_beds','Guest_Single_beds']
colors = ['r','g','b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
              , finalDf.loc[indicesToKeep, 'principal component 2']
              , c = color
              , s = 50)

ax.legend(targets)
ax.grid()


# # Conclusion

# ### Plot the distribution of diameter

# In[47]:


import seaborn as sns


# In[48]:


sgb_cleaning.head()


# #### - Histogram distribution of total quantity in each Part No.

# In[49]:



# Histogram distribution of total quantity in each part No.
chart = sns.displot(sgb_cleaning['quantity'], kde=False)
chart


# ##### From the histogram chart above, most parts are used below 20 quantity. But there are some parts are frequent used with over 80 quantity usage.

# #### Bart plots: component quantity of types of IKEA beds

# In[50]:


# Show the type's distribute of bar plots
plt.figure(figsize=(8,4))

# Add title and axis
plt.title('Quantity of Types of IKEA Beds')
plt.ylabel('Quantity')

# Bar chart showing diameter for each screw type
sns.barplot(x=sgb_cleaning['type'], y=sgb_cleaning['quantity'])


# ##### From the bar chart, we can see that the quantity of guest_single dual components is over 40 in total, meaning that many parts are used in two bed types. in contrary, single beds' part quantity is the lowest, with 10 in total usage.

# ### - Lineplot for all product series of IKEA bed

# In[51]:


sgb_cleaning.head()


# In[52]:


# Slice the columns of all product series, which start from 'FJELLSE' till 'UT'
products = sgb_cleaning.iloc[:,3:]
products


# In[53]:


# Set the width and height of the figure
plt.figure(figsize=(12,3.8))

# Line char showing how single and bed products differ
sns.lineplot(data=products)


# ##### From the lineplot, we can see that HEMNES has high quantity of components, with some parts may over 50 quantities. In some parts, HEMNES even amost 70 quantity. 

# In sum, most components are used under 10 quantity in total, but there are still some parts have very high usage with over 80 quantity. In addition, there are 24 identical parts are used in guest and single beds, and those identical parts accounts the most percentage of total parts quantity usage. For single beds, the percentage of unique components (in terms of their quantity usage) is 22.57%. For guest beds, the percentage of unique components (in terms of their quantity usage) is 45.84%. 

# In[ ]:




