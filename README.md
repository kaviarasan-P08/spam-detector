Spam Detection Project

A Python-based Spam Detection system using Naive Bayes and text vectorization to classify SMS messages as Spam or Not Spam. Includes a command-line predictor and an optional Streamlit web interface for easy interaction.

🛠 Features

Train a spam detection model using the SMS Spam Collection dataset

High accuracy (~98%) with Multinomial Naive Bayes

Save and load model for fast predictions

Command-line interface for single message prediction

Optional Streamlit UI for interactive web-based predictions

💻 Technologies Used

Python 3

Pandas

Scikit-learn (Naive Bayes, CountVectorizer)

Streamlit (optional for UI)

Joblib (for model serialization)

📁 Project Structure
spam_detector_project/
├── spam_data.csv         # Dataset of SMS messages
├── spam_detector.py      # Model training and saving
├── spam_model.pkl        # Saved trained model
├── vectorizer.pkl        # Saved CountVectorizer
├── predict.py            # Predict messages without retraining
└── spam_ui.py            # Optional Streamlit web UI

⚡ Installation

Clone the repo:

git clone https://github.com/yourusername/spam_detector_project.git
cd spam_detector_project


Install dependencies:

pip3 install pandas scikit-learn streamlit joblib

🚀 Usage
1. Train the Model
python3 spam_detector.py


Trains the model on the dataset

Saves spam_model.pkl and vectorizer.pkl

2. Predict a Message
python3 predict.py


Enter any message to check if it is Spam or Not Spam

3. Optional Streamlit UI
streamlit run spam_ui.py


Open the browser interface

Type a message to see instant predictions

📈 Accuracy

Typically achieves ~98% accuracy on the SMS Spam Collection dataset.

📂 Dataset

SMS Spam Collection Dataset from Kaggle

⚡ Contributing

Fork the repo

Create a new branch for feature enhancements

Submit pull requests with clear descriptions
