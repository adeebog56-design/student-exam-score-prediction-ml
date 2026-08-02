#Importing the module

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading csv file

df = pd.read_csv("Dataset/Students.csv")

print("============OBSERVATION============")
print()

print('Seeing the starting 10 lines of data :\n\n',df.head(10))
print()

print('Basic Info of Data : \n')
df.info()
print()

print('General Statistics of data : \n\n',df.describe())
print()

print('Duplicates Count : ',df.duplicated().sum())
print()

print("Missing values : \n\n",df.isna().sum())
print()

df = df.drop(columns=["parental_education_level"])

# Counts

print('============COUNTS============')
print()

print('Gender Count :\n\n',df['gender'].value_counts())
print()

print('Diet_Quality_Count :\n\n',df['diet_quality'].value_counts())
print()

print('Internet_Quality_Count : \n\n',df['internet_quality'].value_counts())
print()

# Graphs

print("============GRAPHS============")
print()

plt.figure(figsize=(8,5))

sns.histplot(df['netflix_hours'], bins =6)

plt.savefig("Outputs/Graphs/Hours.png")
plt.close()

print("Netflix_hours Histogram...")
print()

plt.figure(figsize=(8,5))

sns.histplot(df['mental_health_rating'], bins =6)

plt.savefig("Outputs/Graphs/mental_health_rating.png")
plt.close()

print("Mental_health_rating Hstogram...")
print()

plt.figure(figsize=(18,15))

correlation = df.select_dtypes(include='number').corr()

sns.heatmap(correlation,annot=True , cmap = 'Blues')

plt.title("Correlation Heatmap")

plt.savefig("Outputs/Graphs/Heatmap.png")
plt.close()

print("Heatmap...")
print()

print("============END============")