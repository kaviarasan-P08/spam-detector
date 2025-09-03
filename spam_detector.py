import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


print("🚀 Spam Detector Script Started")

try:
    # Step 1: Load and clean data
    data = pd.read_csv("spam_data.csv", encoding='latin-1')[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )

    # Step 3: Vectorization
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Step 4: Model training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Step 5: Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.2f}")

    # Step 6: Message prediction function
    def predict_message(msg):
        msg_vec = vectorizer.transform([msg])
        prediction = model.predict(msg_vec)[0]
        return "Spam" if prediction == 1 else "Not Spam"

    # Step 7: Test sample message
    if __name__ == "__main__":
        sample_msg = "You have won a free trip to Bahamas! Click here to claim your reward."
        print("\n🔍 Testing with sample message:")
        print("Message:", sample_msg)
        print("Prediction:", predict_message(sample_msg))

except FileNotFoundError:
    print("❌ Error: 'spam_data.csv' not found in the current folder.")
except Exception as e:
    print("❌ An unexpected error occurred:", e)
   

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("📦 Model and vectorizer saved!")

