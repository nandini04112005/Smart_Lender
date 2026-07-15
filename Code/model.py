import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from imblearn.over_sampling import RandomOverSampler

# ==================================
# Read Dataset
# ==================================

df = pd.read_csv("dataset/train.csv")

# ==================================
# Missing Values
# ==================================

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(
    df['Self_Employed'].mode()[0]
)

df['LoanAmount'] = df['LoanAmount'].fillna(
    df['LoanAmount'].median()
)

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(
    df['Loan_Amount_Term'].median()
)

df['Credit_History'] = df['Credit_History'].fillna(
    df['Credit_History'].mode()[0]
)

# ==================================
# Encoding
# ==================================

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

# ==================================
# Split X and y
# ==================================

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# ==================================
# Balance
# ==================================

ros = RandomOverSampler(random_state=42)

X, y = ros.fit_resample(X, y)

# ==================================
# Scaling
# ==================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==================================
# Train Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================
# Decision Tree
# ==================================

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

pred = dt.predict(X_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, pred))

# ==================================
# Random Forest
# ==================================

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, pred))

# ==================================
# KNN
# ==================================

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print("\nKNN Accuracy:")
print(accuracy_score(y_test, pred))

# ==================================
# XGBoost
# ==================================

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)

print("\nXGBoost Accuracy:")
print(accuracy_score(y_test, pred))