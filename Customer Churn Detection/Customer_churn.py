# Customer Churn Prediction - CodSoft Task 3

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import os
print("Current working directory:", os.getcwd())

# Step 1: Load the dataset
#df = pd.read_csv("Churn_Modelling.csv")
# ...existing code...
# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Documents\GitHub\Codsoft\Customer Churn Detection\Churn_Modelling.csv")
# ...existing code...

# Step 2: Drop irrelevant columns
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# Step 3: Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# Step 4: Feature-target split
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Step 5: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train models and evaluate
print("--- Logistic Regression ---")
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))

print("--- Random Forest ---")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

print("--- Gradient Boosting ---")
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print(classification_report(y_test, y_pred_gb))

# Step 8: Save the best model (Random Forest assumed best here)
joblib.dump(rf, "model.pkl")
print("Random Forest model saved as model.pkl")
