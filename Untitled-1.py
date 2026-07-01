# ==========================
# Student Performance Analysis
# ==========================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load Dataset
df = pd.read_csv("student-mat.csv", sep=";")

print("First 5 Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Create New Columns
df["Average"] = (df["G1"] + df["G2"] + df["G3"]) / 3
df["Result"] = np.where(df["G3"] >= 10, "Pass", "Fail")

# ==========================
# Data Analysis
# ==========================

print("\nAverage Grades")
print(df[["G1", "G2", "G3"]].mean())

print("\nGender-wise Average Grades")
print(df.groupby("sex")[["G1", "G2", "G3"]].mean())

print("\nStudy Time vs Average Grade")
print(df.groupby("studytime")["Average"].mean())

print("\nFailures vs Average Grade")
print(df.groupby("failures")["Average"].mean())

print("\nPass/Fail Count")
print(df["Result"].value_counts())

print("\nAverage Final Grade:", round(df["G3"].mean(), 2))
print("Highest Final Grade:", df["G3"].max())
print("Lowest Final Grade:", df["G3"].min())

# ==========================
# Visualizations
# ==========================

# 1 Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="sex", data=df)
plt.title("Gender Distribution")
plt.show()

# 2 Final Grade Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["G3"], bins=20, kde=True)
plt.title("Distribution of Final Grades")
plt.show()

# 3 Study Time vs Final Grade
plt.figure(figsize=(7,5))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grade")
plt.show()

# 4 Average Grade by Gender
plt.figure(figsize=(6,4))
sns.barplot(x="sex", y="Average", data=df)
plt.title("Average Grade by Gender")
plt.show()

# 5 School-wise Performance
plt.figure(figsize=(6,4))
sns.barplot(x="school", y="Average", data=df)
plt.title("Average Grade by School")
plt.show()

# 6 Absences vs Final Grade
plt.figure(figsize=(7,5))
sns.scatterplot(x="absences", y="G3", data=df)
plt.title("Absences vs Final Grade")
plt.show()

# 7 Weekend Alcohol Consumption
plt.figure(figsize=(7,5))
sns.boxplot(x="Walc", y="G3", data=df)
plt.title("Weekend Alcohol Consumption vs Final Grade")
plt.show()

# 8 Correlation Heatmap
plt.figure(figsize=(12,8))
corr = df.select_dtypes(include=["int64", "float64"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 9 Pass vs Fail
plt.figure(figsize=(5,4))
sns.countplot(x="Result", data=df)
plt.title("Pass vs Fail")
plt.show()

# 10 Pair Plot
sns.pairplot(df[["G1", "G2", "G3", "studytime", "absences"]])
plt.show()

# ==========================
# Final Insights
# ==========================

print("\n========== SUMMARY ==========")
print("Total Students:", len(df))
print("Average Final Grade:", round(df["G3"].mean(), 2))
print("Pass Percentage:", round((df["Result"] == "Pass").mean() * 100, 2), "%")
print("Average Absences:", round(df["absences"].mean(), 2))
print("=============================")