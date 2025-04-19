import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import time

# Set NLTK data path if needed
nltk.data.path.append(r'C:\Users\ASUS\AppData\Roaming\nltk_data')

# Do NOT download if already present
# nltk.download('stopwords')
from nltk.corpus import stopwords

# Pre-load stopwords once (for speed)
stop_words = set(stopwords.words('english'))

# Load dataset
def load_data(true_file, fake_file):
    print("📄 Loading datasets...")
    true_df = pd.read_csv(true_file)
    fake_df = pd.read_csv(fake_file)
    
    true_df['label'] = 1
    fake_df['label'] = 0
    
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    print(f"✅ Loaded {len(combined_df)} total articles.")
    return combined_df

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Main function
if __name__ == "__main__":
    start_time = time.time()

    df = load_data('data/True.csv', 'data/Fake.csv')

    print("🛠️ Preprocessing text...")
    total = len(df)
    for i in range(total):
        df.at[i, 'text'] = preprocess_text(df.at[i, 'text'])
        if i % 1000 == 0:
            print(f"   ✅ Processed {i}/{total} articles...")

    print("📊 Splitting data and vectorizing...")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("💾 Saving processed data...")
    pd.to_pickle((X_train_vectorized, X_test_vectorized, y_train, y_test), 'processed_data.pkl')
    pd.to_pickle(vectorizer, 'vectorizer.pkl')

    end_time = time.time()
    print(f"\n✅ Done! Total time: {end_time - start_time:.2f} seconds")