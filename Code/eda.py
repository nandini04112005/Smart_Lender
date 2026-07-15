import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv("dataset/train.csv")

# Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# ===================================
# UNIVARIATE ANALYSIS
# ===================================

print("\nDisplaying: Loan Status")
df['Loan_Status'].value_counts().plot(kind='bar')
plt.title('Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

print("\nDisplaying: Gender")
df['Gender'].value_counts().plot(kind='bar')
plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

print("\nDisplaying: Married")
df['Married'].value_counts().plot(kind='bar')
plt.title('Married')
plt.xlabel('Married')
plt.ylabel('Count')
plt.show()

print("\nDisplaying: Education")
df['Education'].value_counts().plot(kind='bar')
plt.title('Education')
plt.xlabel('Education')
plt.ylabel('Count')
plt.show()

print("\nDisplaying: Self Employed")
df['Self_Employed'].value_counts().plot(kind='bar')
plt.title('Self Employed')
plt.xlabel('Self Employed')
plt.ylabel('Count')
plt.show()

print("\nDisplaying: Property Area")
df['Property_Area'].value_counts().plot(kind='bar')
plt.title('Property Area')
plt.xlabel('Property Area')
plt.ylabel('Count')
plt.show()

print("\nDisplaying: Applicant Income")
plt.hist(df['ApplicantIncome'], bins=20)
plt.title('Applicant Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

print("\nDisplaying: Loan Amount")
plt.hist(df['LoanAmount'].dropna(), bins=20)
plt.title('Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()

# ===================================
# BIVARIATE ANALYSIS
# ===================================

print("\nDisplaying: Education vs Loan Status")
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title('Education vs Loan Status')
plt.show()

print("\nDisplaying: Gender vs Loan Status")
sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.title('Gender vs Loan Status')
plt.show()

print("\nDisplaying: Married vs Loan Status")
sns.countplot(x='Married', hue='Loan_Status', data=df)
plt.title('Married vs Loan Status')
plt.show()

print("\nDisplaying: Property Area vs Loan Status")
sns.countplot(x='Property_Area', hue='Loan_Status', data=df)
plt.title('Property Area vs Loan Status')
plt.show()

# ===================================
# MULTIVARIATE ANALYSIS
# ===================================

print("\nDisplaying: Correlation Heatmap")
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("\nEDA Completed Successfully!")