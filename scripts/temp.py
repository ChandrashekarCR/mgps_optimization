import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import GradientBoostingClassifier



# Read the data 
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
df = pd.concat([df.iloc[:,:-4],df['continent']],axis=1)
x_data = df[df.columns[:-1]][:2000]
print(x_data.shape)
y_data = df[df.columns[-1]][:2000]
le = LabelEncoder()
y_data = le.fit_transform(y_data)
print(le.classes_)


# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,random_state=123,test_size=0.2, stratify=y_data)
# Split train into train and validation as well
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2,stratify=y_train)

print('Training, Validation and Testing matrices shapes')
print("\nTraining\n")
print(X_train.shape, y_train.shape)
print("\nValidation\n")
print(X_val.shape, y_val.shape)
print("\nTesting\n")
print(X_test.shape, y_test.shape)


# Set the model XGB Classifier

xgb_classifier = XGBClassifier(objective="multi:softmax",
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42, n_estimators = 100, max_depth = 3,
)
xgb_classifier.fit(X_train, y_train)

# Train on the training dataset
xgb_classifier.fit(X_train,y_train)

# Validate on the validation dataset
y_pred = xgb_classifier.predict(X_test)

test_accuracy = accuracy_score(y_test,y_pred)
print(f"The test accuracy on the validation dataset is {test_accuracy:.4f}")

# Print classification report
print("\nClassfication Report:\n",classification_report(y_test,y_pred))

# Print Confusion Matrix
print("\nConfusion Matrix\n", confusion_matrix(y_test,y_pred))

