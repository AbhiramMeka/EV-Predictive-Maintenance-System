import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load and explore data
df = pd.read_csv('EV_Predictive_Maintenance_Dataset_15min.csv')
print(df.head())
print("\n--- Info ---")
print(df.info())
print("\n--- Description ---")
print(df.describe())

# Timestamp handling
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
print("\n--- After setting Timestamp index ---")
print(df.head())

# Downsampling
df_downsampled = df.resample('2h').mean()

# Plot 1: RUL vs TTF
plt.figure(figsize=(14, 5))
plt.plot(df_downsampled.index, df_downsampled['RUL'], label='RUL', color='orange')
plt.plot(df_downsampled.index, df_downsampled['TTF'], label='TTF', color='purple')
plt.title('RUL vs TTF Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Failure Probability
plt.figure(figsize=(14, 5))
plt.plot(df_downsampled.index, df_downsampled['Failure_Probability'], label='Failure Probability', color='blue')
plt.scatter(df.index[df['Failure_Probability'] == 1], [1]*sum(df['Failure_Probability'] == 1),
            color='red', label='Failure Events', s=5)
plt.title('Failure Probability Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Failure Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Component Health Score
plt.figure(figsize=(14, 5))
sns.lineplot(data=df_downsampled, x=df_downsampled.index, y='Component_Health_Score', color='green')
plt.title('Component Health Score Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Health Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap of EV Sensor Features')
plt.tight_layout()
plt.show()

# Extract time features
df['Hour'] = df.index.hour
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year

# Optional drop
df.drop(['Distance_Traveled'], axis=1, inplace=True)

# Fill missing
df.fillna(df.mean(numeric_only=True), inplace=True)

# Modeling
X = df.drop('Maintenance_Type', axis=1)
y = df['Maintenance_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Maintenance_Type Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

importances = clf.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette="viridis")
plt.title('Top 10 Important Features for Predicting Maintenance Type')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

import joblib
joblib.dump(clf, 'rf_maintenance_model.pkl')

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_bal, y_train_bal)

