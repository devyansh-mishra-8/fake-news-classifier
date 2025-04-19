import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load processed data
X_train_vectorized, X_test_vectorized, y_train, y_test = pd.read_pickle('processed_data.pkl')

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)