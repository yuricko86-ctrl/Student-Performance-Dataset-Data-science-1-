# ============================================
# Student Performance Analysis using Python
# ============================================

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load Dataset
# The CSV file is inside the folder named "student-mat.csv"
df = pd.read_csv("student-mat.csv/student-mat.csv", sep=";")

print("========== FIRST 5 RECORDS ==========")
print(df.head())

# 3. Explore & Clean Data
print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# 4. Data Analysis

# Average Final Grade
average_grade = df["G3"].mean()
print("\nAverage Final Grade (G3):", round(average_grade, 2))

# Students scoring above 15
students_above_15 = df[df["G3"] > 15].shape[0]
print("Students Scoring Above 15:", students_above_15)

# Correlation between study time and performance
correlation = df["studytime"].corr(df["G3"])
print("Correlation between Study Time and G3:", round(correlation, 2))

# Average performance by gender
gender_average = df.groupby("sex")["G3"].mean()

print("\nAverage Final Grade by Gender:")
print(gender_average)

# ============================================
# Visualizations
# ============================================

# Histogram of Grades
plt.figure(figsize=(8,5))
plt.hist(df["G3"], bins=10, edgecolor="black")
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.show()

# Scatter Plot: Study Time vs Grades
plt.figure(figsize=(8,5))
plt.scatter(df["studytime"], df["G3"])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.grid(True)
plt.show()

# Bar Chart: Male vs Female Average Score
plt.figure(figsize=(6,5))
gender_average.plot(kind="bar", color=["skyblue", "orange"])
plt.title("Average Final Grade by Gender")
plt.xlabel("Gender")
plt.ylabel("Average G3 Score")
plt.xticks(rotation=0)
plt.show()

# ============================================
# Summary
# ============================================

print("\n========== SUMMARY ==========")
print("Total Students:", len(df))
print("Average Final Grade:", round(average_grade, 2))
print("Students Scoring Above 15:", students_above_15)
print("Study Time Correlation:", round(correlation, 2))

if gender_average["F"] > gender_average["M"]:
    print("Female students perform better on average.")
elif gender_average["M"] > gender_average["F"]:
    print("Male students perform better on average.")
else:
    print("Male and Female students perform equally on average.")