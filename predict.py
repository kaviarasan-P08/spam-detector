import joblib

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define message prediction function
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Run from terminal
if __name__ == "__main__":
    print("📩 Spam Predictor Ready!")
    user_input = input("✉️ Enter your message: ")
    result = predict_message(user_input)
    print("🔍 Result:", result)
