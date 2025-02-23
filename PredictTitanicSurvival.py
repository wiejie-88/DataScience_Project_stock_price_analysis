## Step 1 ##
## Load and Explore the Data ##
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


print("Script started...")  # Debugging Start

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display the first few rows
print(df.head())

# Check for missing values
print("Missing values before cleaning:\n", df.isnull().sum())

# Basic statistics
print(df.describe())



## Step 2 ##
## Data Cleaning ##
# Fill missing age values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with the most common value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop irrelevant columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Convert categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Verify no missing values remain
print("Missing values after cleaning:\n", df.isnull().sum())



## Step 3 ##
## Data Visualization ##
# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()



## Step 4 ##
## Feature Engineering ##
# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



## Step 5 ##
## Build a Machine Learning Model ##
# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debugging: Ensure data split is correct
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Check for missing values in training and test sets
print("Missing values in training set:\n", X_train.isnull().sum())
print("Missing values in test set:\n", X_test.isnull().sum())

# Verify categorical encodings
print("Unique values in Sex:", df['Sex'].unique())
print("Unique values in Embarked:", df['Embarked'].unique())

# Train a Random Forest model
print("Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Debugging: Ensure predictions are being made
print("First 10 Predictions:", y_pred[:10])

# Ensure y_pred and y_test have matching shapes
print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("Script completed successfully!")  # Debugging End
