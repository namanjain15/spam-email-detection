# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
df = pd.read_csv("mail.csv")

# Fill missing values with an empty string
df.fillna("")

# Create a copy of the original dataframe
le = LabelEncoder()
label_df = df

# Encode the categorical labels using LabelEncoder
label_df.Category = le.fit_transform(df.Category)

# Split the data into features (Message) and target (Category)
x = label_df["Message"]
y = label_df[["Category"]]

# Split the data into training and testing sets (70% for training and 30% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create a TF-IDF vectorizer to convert text data into numerical features
fe = TfidfVectorizer(min_df=1, stop_words='english', lowercase=1)

# Fit the vectorizer to the training data and transform both training and testing data
x_train_new = fe.fit_transform(x_train)
x_test_new = fe.transform(x_test)

# Convert the target variables to integers
y_train = y_train.astype("int")
y_test = y_test.astype("int")

# Create a Logistic Regression model
lr = LogisticRegression()

# Train the model on the training data
lr.fit(x_train_new, y_train)

# Evaluate the model on the training data
print("Training accuracy:", lr.score(x_train_new, y_train))

# Make predictions on the testing data
prediction = lr.predict(x_test_new)
print("Predicted labels:", prediction)

# Evaluate the model on the testing data
accuracy = accuracy_score(y_test, prediction)
print("Testing accuracy:", accuracy)

import pickle 
with open("spam", "wb") as file:
    pickle.dump(lr,file)

with open("words","wb") as file1:
    pickle.dump(fe,file1)
    
mails = input("Enter mail")
x_test = [mails]
x_test_new = fe.transform(x_test)
prediction = lr.predict(x_test_new)
# print(prediction)
if (prediction==0):
    print("The mail is SPAM")
else:
    print("it is REAL")