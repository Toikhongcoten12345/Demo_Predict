import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Đọc dữ liệu từ file CSV
df = pd.read_csv("predict_Covid_19.csv")

# Tách dữ liệu thành features (x) và target (y)
x = df.drop('Danger Level', axis=1)
y = df['Danger Level']

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(x_train, y_train)

# Dự đoán trên tập kiểm tra
rf_pred = random_forest_model.predict(x_test)
print("Dự đoán từ RandomForestClassifier: ", rf_pred)

# Lưu mô hình RandomForestClassifier
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest_model, file)
