import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from database.load_data import load_data

print("Loading data from SQL...")
df = load_data()

X = df.drop("academic_risk", axis=1)
y = df["academic_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier()
}

best_acc = 0

print("Training models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(name, "Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
path = f"models/saved_models/model_{timestamp}.pkl"
joblib.dump(best_model, path)

pd.DataFrame([[timestamp, best_name, best_acc]]).to_csv(
    "models/model_log.csv", mode="a", header=False, index=False
)

print("Best model saved:", path)

import os
os.makedirs("models/saved_models", exist_ok=True)