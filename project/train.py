import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 1. Загружаем данные
data = pd.read_csv('training_data.csv')
X = data['message']
y = data['label']

# 2. Преобразуем текст в векторы
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 3. Обучаем модель
model = SVC(kernel='linear')
model.fit(X_vec, y)

# 4. Сохраняем модель и векторизатор
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Модель обучена и сохранена!")