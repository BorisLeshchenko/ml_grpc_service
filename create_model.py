import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# загружаем данные iris
iris = load_iris()
X, y = iris.data, iris.target

# обучаем модель
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# сохраняем модель
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("модель успешно сохранена в models/model.pkl")
