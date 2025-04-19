# 📰 Fake News Classifier

## 📌 Overview
The **Fake News Classifier** is a complete end-to-end tool designed to classify news articles as *real* or *fake* using advanced Machine Learning and Natural Language Processing (NLP) techniques. The classifier achieves an accuracy of **93.09%** using NLTK-based preprocessing.

---

## 🚀 Features
- ✅ Reliable fake news detection with 93.09% accuracy
- 💻 User-friendly web interface
- ⚙️ Supports multiple ML models
- 🧹 Utilizes **NLTK** for text preprocessing
- 📊 Displays detailed classification results

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/devyansh-mishra-8/fake-news-classifier.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd fake-news-classifier
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   Open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Classify news articles:**
   Input the **title** and **body** of the news article to get real/fake predictions.

---

## 📂 Data & Preprocessing

- Data includes news articles from **reliable sources** and **flagged fake news** sites.
- Preprocessing with **NLTK** includes:
  - Stopword removal
  - Tokenization
  - Lemmatization
  - Vectorization using TF-IDF

---

## 🤖 Models Used

- 🧮 Naive Bayes
- 📈 Logistic Regression

> The final model achieves an accuracy of **93.09%** using NLTK and TF-IDF features.

---

## 📊 Results

| Model               | Accuracy |
|--------------------|----------|
| Naive Bayes        | ~85%     |
| Logistic Regression| **93.09%** |

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- Thanks to the open-source community for tools like **NLTK**, **scikit-learn**, and others.
- Inspired by research in misinformation detection and NLP.
