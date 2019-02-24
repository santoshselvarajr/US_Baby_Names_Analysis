#Project: Baby Names
#Created by: Santosh Selvaraj
#Date: 13 Feb, 2019

#importing necessary libraries
import os
#Assigning working directory
wdDir = "C:\\Users\\Santosh Selvaraj\\Documents\\Working Directory\\Data Science Projects\\BabyNames"
#Setting path to working directory
os.chdir(wdDir)
#Importing other required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.stats import chi2_contingency

#Creating a list of text files
txtlist = []
for file in os.listdir(wdDir):
    if file.endswith(".TXT"):
        txtlist.append(file)

#Creating a dataframe with all text files appended
data = pd.DataFrame()
for file in txtlist:
    data = data.append(pd.read_csv(file,header=None))
#Adding column names to the dataset
data.columns = ["state","gender","birth_year","name","count"]

#Ensuring data import was done correctly
data.tail() #Expecting low count values for state WY
data.head() #Expecting high count values for state AK

#Checking for Nan values in the dataset and understanding datatypes
data.info()

#Understanding and summarizing the dataset
data.describe().astype(np.int64) #Data is from 1910 to 2017, with min count of 5 and max count ~ 10K

#Excercise 1: Jessie and Riley
data1 = data[(data["name"] == "Jessie") | (data["name"] == "Riley")]
data1["count"].sum() #Before group by: 417025
data1 = data1.groupby(["gender","name"], as_index = False).agg({"count":"sum"})
data1["count"].sum() #After group by: 417025

# Visualizing the distribution of counts across Jessie and Riley
n_groups = 2
female_count = data1[data1["gender"] == "F"]["count"].values
male_count = data1[data1["gender"] == "M"]["count"].values
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
bar1 = plt.bar(index, female_count, bar_width, alpha=opacity, color='b', label='Female')
bar2 = plt.bar(index + bar_width, male_count, bar_width, alpha=opacity, color='g', label='Male')
 
plt.xlabel('Baby Name')
plt.ylabel('Count')
plt.title('Baby Name Counts (1910-2017)')
plt.xticks(index + 0.5*bar_width, ('Jessie', 'Riley'))
plt.legend()
plt.tight_layout()
plt.show()

#Chisquare test
#Female percentage is 48% and Male percentage is 52%
#Hence the names are expected to be distributed in the same ratio across the gender
#Calculating the expected freqencies and using chisquare test
#Null Hypothesis: Baby names Jessie and Riley do not show association with any particular gender
#A chi-square goodness of fit test allows us to test whether the observed proportions for a categorical variable differ from hypothesized proportions.
crosstab = pd.crosstab(data1["name"],data1["gender"],values=data1["count"],aggfunc=sum,margins=None)
stat, p, dof, expected = chi2_contingency(crosstab)
print("ChiSquare Test Results:\nChiSquare Statistic: %d\nP-Value: %f" % (stat,p))
#Low p-value suggests that the  observed frequencies are significantly different from expected marginal frequencies
crosstab = crosstab.values
jessie_f = (crosstab[0,0]-expected[0,0])**2/expected[0,0]
jessie_m = (crosstab[0,1]-expected[0,1])**2/expected[0,1]
riley_f = (crosstab[1,0]-expected[1,0])**2/expected[1,0]
riley_m = (crosstab[1,1]-expected[1,1])**2/expected[1,1]
print("Individual ChiSquare Statistic:\nJessie Female: %d\nJessie Male: %d\nRiley Female: %d\nRiley Male: %d" %(jessie_f,jessie_m,riley_f,riley_m))
#Based on the individual statistic values, we can conclude that Riley associates with both male and female more as compared to Jessie

#Excercise 2
#Filter for year between 1900 and 2000 (Data starts at 1910)
data2 = data[(data["birth_year"]>=1900) & (data["birth_year"]<=2000)]
#Aggregate the counts across filtered years
common_names = data2.groupby(["gender","name"], as_index = False).agg({"count":"sum"})
#Sort counts in descending order
common_names = common_names.sort_values(by=["gender","count"], ascending = False)
#Get the top 5 names for each gender
common_names = common_names.groupby("gender").head(5).reset_index(drop=True)
#Get the 10 common names
common_names = common_names[["name","gender"]]
#Filter data to get only for the common names
data2 = pd.merge(data2,common_names,on=["name","gender"],how="inner")
#Grouping data to remove state column
data2 = data2.groupby(["birth_year","name"],as_index=False).agg({"count":"sum"})
#data2.groupby(["name","gender"]).nunique()

#Female Common Names
fig, ax = plt.subplots()
fig.set_size_inches(12,7)
for name in list(common_names[common_names["gender"]=="F"]["name"].values):
    plt.plot(data2[data2["name"]==name]["birth_year"],data2[data2["name"]==name]["count"]/1000, label = name)
plt.legend(loc = 1)
plt.xlabel("Time")
plt.ylabel("Baby Name Counts (in Thousands)")
plt.title("Baby Name Trends: 5 Most Common Female Names")
plt.show()

#Male Common Names
fig, ax = plt.subplots()
fig.set_size_inches(12,7)
for name in list(common_names[common_names["gender"]=="M"]["name"].values):
    plt.plot(data2[data2["name"]==name]["birth_year"],data2[data2["name"]==name]["count"]/1000, label = name)
plt.legend(loc = 1)
plt.xlabel("Time")
plt.ylabel("Baby Name Counts (in Thousands)")
plt.title("Baby Name Trends: 5 Most Common Male Names")
plt.show()

#Excercise 3:
#Finding any trends in the length of the name
data3 = data.groupby(["birth_year","name"], as_index = False).agg({"count":"sum"})
data3["name_length"] = [len(x) for x in data3["name"].values]
data3 = data3.groupby(["birth_year","name_length"], as_index = False).agg({"count":"sum"})
#Create custom groups
data3.loc[data3['name_length']<5, 'length_groups'] = "<5"
data3.loc[(data3['name_length']>=5) & (data3['name_length']<10), 'length_groups'] = "5-10"
data3.loc[data3['name_length']>=10, 'length_groups'] = "10+"
data3 = data3.groupby(["birth_year","length_groups"], as_index = False).agg({"count":"sum"})
#Plot name lengths across year
fig, ax = plt.subplots()
fig.set_size_inches(12,7)
for i in list(data3["length_groups"].unique()):
    plt.plot(data3[data3["length_groups"]==i]["birth_year"],data3[data3["length_groups"]==i]["count"]/1000, label = i)
plt.legend(loc = 1)
plt.xlabel("Time")
plt.ylabel("Baby Name Counts (in Thousands)")
plt.title("Baby Name Trends: Trends of Name Lengths")
plt.show()

#Unique baby names
data4 = data.groupby(["birth_year","gender"], as_index = False).agg({"name":"nunique"})

fig, ax = plt.subplots()
fig.set_size_inches(12,7)
plt.plot(data4[data4["gender"]=="F"]["birth_year"],data4[data4["gender"]=="F"]["name"], label = "Female")
plt.plot(data4[data4["gender"]=="M"]["birth_year"],data4[data4["gender"]=="M"]["name"], label = "Male")
plt.legend(loc = 2)
plt.xlabel("Time")
plt.ylabel("Unique Baby Name Counts")
plt.title("Baby Name Trends: Unique Baby Names")
plt.show()

#Baby Names based on alphabets
data5 = data.groupby(["birth_year","name"], as_index = False).agg({"count":"sum"})
data5["start_letter"] = data5["name"].str.slice(stop=1)
data5 = data5.groupby(["birth_year","start_letter"], as_index = False).agg({"count":"sum"})

#Plot
sorted_start_letters = sorted(data5["start_letter"].unique())
fig, axes = plt.subplots(nrows=6, ncols=5, sharex = True)
fig.set_size_inches(12,7)
fig.set_
x = 0
for letter in sorted_start_letters:
    m = x//5
    n = x%5
    data5[data5['start_letter']==letter].plot(x="birth_year",y="count",sharey = True,ax=axes[m, n],legend=True,label=letter)
    x+=1
plt.xlabel("Year")
plt.show()
