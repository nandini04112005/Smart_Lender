import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# Read dataset
df = pd.read_csv("dataset/train.csv")

# Missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Encoding
le = LabelEncoder()

cols = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

for col in cols:
    df[col] = le.fit_transform(df[col])

df = df.drop('Loan_ID', axis=1)

# X and y
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Balancing
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
rf = RandomForestClassifier()
rf.fit(X, y)

# Save model
joblib.dump(rf, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model Saved Successfully!")