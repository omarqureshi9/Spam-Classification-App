# app.py

# Importing the required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import gradio as gr

# Loading the dataset
data = pd.read_csv('data/spam.csv')

# Converting labels to binary
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation (optional)
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Gradio interface
def predict_spam(message):
    msg_vector = vectorizer.transform([message])
    prediction = model.predict(msg_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

interface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=3, placeholder="Enter your message..."),
    outputs="text",
    title="Spam Classifier",
    description="Enter a message and the model will classify it as Spam or Not Spam."
)

interface.launch(share=True)
