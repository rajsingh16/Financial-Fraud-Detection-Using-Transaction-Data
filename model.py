
import pandas as pd

# Load the dataset
file_path = 'D:\\Btech sem\\project\\IBM intership\\User0_credit_card_transactions.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()

"""Cleaning data"""

# Clean the dataset
data['Amount'] = data['Amount'].replace('[\$,]', '', regex=True).astype(float)
data['Is Fraud?'] = data['Is Fraud?'].map({'No': 0, 'Yes': 1})
data = data.drop(columns=['Errors?'])  # Dropping column with many missing values
data = data.dropna()  # Dropping rows with any missing values

# Display the cleaned data
data.info(), data.head()

"""UNDERSAMPLING / DOWNSAMPLING"""

from sklearn.utils import resample

# Separate the majority and minority classes
majority_class = data[data['Is Fraud?'] == 0]
minority_class = data[data['Is Fraud?'] == 1]

# Downsample the majority class
majority_class_downsampled = resample(majority_class,
                                      replace=False,    # sample without replacement
                                      n_samples=10*len(minority_class), # match minority class
                                      random_state=42) # reproducible results

# Combine minority class with downsampled majority class
downsampled_data = pd.concat([majority_class_downsampled, minority_class])

# Display the class distribution
downsampled_data['Is Fraud?'].value_counts()

"""UPSAMPLING/ OVERSAMPLING"""

from sklearn.utils import resample

# Upsample the minority class
minority_class_upsampled = resample(minority_class,
                                    replace=True,     # sample with replacement
                                    n_samples=len(majority_class), # match majority class
                                    random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
upsampled_data = pd.concat([majority_class, minority_class_upsampled])

# Display the class distribution
upsampled_data['Is Fraud?'].value_counts()

"""RANDOM FOREST"""

from sklearn.model_selection import train_test_split

# Features and target variable
X = data.drop(columns=['Is Fraud?', 'Time', 'Use Chip','Merchant City','Merchant State'])
y = data['Is Fraud?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train the Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""RANDOM FOREST ON UNDERSAMPLE & OVERSAMPLE"""

# For undersampled data
X_downsampled = downsampled_data.drop(columns=['Is Fraud?', 'Time', 'Use Chip','Merchant City','Merchant State'])
y_downsampled = downsampled_data['Is Fraud?']
X_train_down, X_test_down, y_train_down, y_test_down = train_test_split(X_downsampled, y_downsampled, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_down = RandomForestClassifier(random_state=42)
rf_down.fit(X_train_down, y_train_down)

# Predictions
y_pred_down = rf_down.predict(X_test_down)

# Evaluation
print("Undersampled Data - Classification Report:\n", classification_report(y_test_down, y_pred_down))
print("Undersampled Data - Confusion Matrix:\n", confusion_matrix(y_test_down, y_pred_down))

# For oversampled data
X_upsampled = upsampled_data.drop(columns=['Is Fraud?', 'Time', 'Use Chip','Merchant City','Merchant State'])
y_upsampled = upsampled_data['Is Fraud?']
X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_up = RandomForestClassifier(random_state=42)
rf_up.fit(X_train_up, y_train_up)

# Predictions
y_pred_up = rf_up.predict(X_test_up)

# Evaluation
print("Oversampled Data - Classification Report:\n", classification_report(y_test_up, y_pred_up))
print("Oversampled Data - Confusion Matrix:\n", confusion_matrix(y_test_up, y_pred_up))

"""ORIGINAL DATA EVALUATION"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train the Random Forest classifier on the original data
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation metrics
accuracy_original = accuracy_score(y_test, y_pred)
precision_original = precision_score(y_test, y_pred)
recall_original = recall_score(y_test, y_pred)
f1_original = f1_score(y_test, y_pred)

print("Original Data - Accuracy:", accuracy_original)
print("Original Data - Precision:", precision_original)
print("Original Data - Recall:", recall_original)
print("Original Data - F1 Score:", f1_original)

"""UNDERSAMPLING EVALUATION"""

# For undersampled data
X_downsampled = downsampled_data.drop(columns=['Is Fraud?', 'Time', 'Use Chip','Merchant City','Merchant State'])
y_downsampled = downsampled_data['Is Fraud?']
X_train_down, X_test_down, y_train_down, y_test_down = train_test_split(X_downsampled, y_downsampled, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_down = RandomForestClassifier(random_state=42)
rf_down.fit(X_train_down, y_train_down)

# Predictions
y_pred_down = rf_down.predict(X_test_down)

# Evaluation metrics
accuracy_down = accuracy_score(y_test_down, y_pred_down)
precision_down = precision_score(y_test_down, y_pred_down)
recall_down = recall_score(y_test_down, y_pred_down)
f1_down = f1_score(y_test_down, y_pred_down)

print("Undersampled Data - Accuracy:", accuracy_down)
print("Undersampled Data - Precision:", precision_down)
print("Undersampled Data - Recall:", recall_down)
print("Undersampled Data - F1 Score:", f1_down)

"""OVERSAMPLING EVALUATAION"""

# For oversampled data
X_upsampled = upsampled_data.drop(columns=['Is Fraud?', 'Time', 'Use Chip', 'Merchant City', 'Merchant State'])
y_upsampled = upsampled_data['Is Fraud?']
X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_up = RandomForestClassifier(random_state=42)
rf_up.fit(X_train_up, y_train_up)

# Predictions
y_pred_up = rf_up.predict(X_test_up)

# Evaluation metrics
accuracy_up = accuracy_score(y_test_up, y_pred_up)
precision_up = precision_score(y_test_up, y_pred_up)
recall_up = recall_score(y_test_up, y_pred_up)
f1_up = f1_score(y_test_up, y_pred_up)

print("Oversampled Data - Accuracy:", accuracy_up)
print("Oversampled Data - Precision:", precision_up)
print("Oversampled Data - Recall:", recall_up)
print("Oversampled Data - F1 Score:", f1_up)

# Save the Random Forest models
import pickle

# Save the Random Forest model trained on original data
with open('rf_original.pkl', 'wb') as file:
    pickle.dump(rf, file)

with open('rf_downsampled.pkl', 'wb') as file:
    pickle.dump(rf_down, file)

with open('rf_upsampled.pkl', 'wb') as file:
    pickle.dump(rf_up, file)

print("Models saved successfully!")
