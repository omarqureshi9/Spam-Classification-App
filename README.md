# 📧 Spam Message & Email Classifier

This project implements a machine learning based web app that classifies messages as **Spam** or **Not Spam** using a **Naive Bayes classifier** and a public dataset. The app is built with **scikit-learn** and deployed via **Gradio** for an interactive interface.

## 🔍 Overview

The app performs the following:

- Loads and preprocesses a labeled spam dataset.
- Converts text message/ email body into numerical features using **CountVectorizer**.
- Trains a **Multinomial Naive Bayes** model to detect spam.
- Evaluates performance with accuracy and a classification report.
- Provides a web interface to test messages in real-time.

## 🚀 Demo

Launch the app in your browser and test it live:

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **scikit-learn**
- **Gradio**

## 📂 Dataset

The dataset used is publicly available and contains labeled messages categorized as:
- `ham` (not spam)
- `spam`

Ensure the CSV file (`spam.csv`) has the following columns:
- `Category` (labels)
- `Message` (text)

You can find similar datasets on [Kaggle](https://www.kaggle.com/datasets) by searching for "spam emails" or "SMS spam".

## 📈 Model Performance

After training on 80% of the data:
- **Accuracy**: ~99% (Naive Bayes classifier)
- Precision, recall, and F1 scores available in the classification report.

## 💡 How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/omarqureshi9/spam-classification-app.git
   cd spam-classification-app

2. Install Dependencies
   ```bash
   pip install -r requirements.txt


3. Run the app
   ```bash
   python app.py



4. Enter any message to classify it as spam or not.

## 📸  Screenshots

![image](https://github.com/user-attachments/assets/d70ae492-81ee-40a6-a21f-07f3e97fb3e6)

## 📜 License
This project is licensed under the MIT License.

   
