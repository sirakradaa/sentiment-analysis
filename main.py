import nltk
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

# Download the movie_reviews corpus
nltk.download("movie_reviews")

# Create a list of documents with their corresponding categories
# Each document is a string of words from the movie_reviews corpus(corpus is the collection of text data)
# Each category is the sentiment of the review(pos or neg)
documents = [(" ".join(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

df = pd.DataFrame(documents, columns=["review", "sentiment"])

#Vectorize the text data
#This converts the text data into a matrix of token counts
#This is done to convert the text data into a format that can be used by machine learning algorithms
#max_features is the maximum number of features (words) to consider
#x is the row of words for each review, capped at 2000, words such as "fantastic", "boring" are pulled from the reviews
#y is the sentiment of the review using binary classification(pos = 1, neg = 0)
#The matrix looks like this:
#loved	movie	fantastic	awful	boring
#1	    0	    1	        0	    0
#0	    1	    0	        1	    0
#1	    0	    0	        0	    1
vectorizer = CountVectorizer(max_features=2000)
x = vectorizer.fit_transform(df["review"]).toarray()
y = df["sentiment"]

#Split the data into training and testing sets
#This is done to train the model on a portion of the data and test it on the remaining data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#Train a Naive Bayes classifier on the training data
#This is a simple and effective algorithm for text classification
#It calculates the probability of each word appearing in a document and uses this to classify the document
#It is a probabilistic model, meaning it makes predictions based on the probability of the data
#For example, in emails the words "click here", "click this link" are often found in spam emails
#The model will classify these emails as spam based on the probability of the words "click here", "click this link" appearing in the email
model = MultinomialNB()
model.fit(x_train, y_train)

#Test the model on the testing data using predictions
y_pred = model.predict(x_test)

#Evaluate the model's performance
#Accuracy is the percentage of correctly classified reviews
#Classification report provides more detailed metrics such as precision, recall, and F1-score for each class
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

#Function to predict the sentiment of a given review
#It transforms the input text into a vector using the same vectorizer used for training
#It then uses the model to predict the sentiment of the review
#It returns the predicted sentiment such as pos or neg
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

#Example usage
example_review = "I love this movie! It was fantastic and awesome."
sentiment = predict_sentiment(example_review)
print(f"The sentiment of the review is: {sentiment}")
example_review2 = "I hate this movie! It was awful and boring."
sentiment2 = predict_sentiment(example_review2)
print(f"The sentiment of the review is: {sentiment2}")
example_review3 = "This movie was okay, it was neither good nor bad."
sentiment3 = predict_sentiment(example_review3)
print(f"The sentiment of the review is: {sentiment3}")

#This code provides a simple sentiment analysis model for movie reviews
#It can be used to classify the sentiment of new reviews as positive or negative
#It can be improved by using more sophisticated models, larger datasets, and more advanced techniques
