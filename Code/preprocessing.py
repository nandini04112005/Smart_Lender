import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# ===================================
# Read Dataset
# ===================================

df = pd.read_csv("dataset/train.csv")

print("Dataset Shape:")
print(df.shape)

print("\nMissing Values Before:")
print(df.isnull().sum())

# ===================================
# Handle Missing Values
# ===================================

# Categorical columns
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(
    df['Self_Employed'].mode()[0]
)

# Numerical columns
df['LoanAmount'] = df['LoanAmount'].fillna(
    df['LoanAmount'].median()
)

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(
    df['Loan_Amount_Term'].median()
)

df['Credit_History'] = df['Credit_History'].fillna(
    df['Credit_History'].mode()[0]
)

print("\nMissing Values After:")
print(df.isnull().sum())

# ===================================
# Encode Categorical Variables
# ===================================

le = LabelEncoder()

categorical_columns = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Remove Loan_ID
df = df.drop('Loan_ID', axis=1)

print("\nEncoded Dataset:")
print(df.head())

# ===================================
# Balance the Dataset
# ===================================

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

print("\nLoan Status Distribution Before Balancing:")
print(y.value_counts())

ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(X, y)

print("\nLoan Status Distribution After Balancing:")
print(y_resampled.value_counts())

# ===================================
# Scale the Data
# ===================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_resampled)

print("\nScaling Completed")

# ===================================
# Split the Dataset
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nPreprocessing Completed Successfully!")