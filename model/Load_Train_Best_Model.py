import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

# โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('/content/data.csv')

# แปลงข้อมูลเป็นตัวเลข
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Marital_Status'] = data['Marital_Status'].map({'Single': 0, 'Married': 1})

if data['Status'].dtype == 'object':
    data['Status'] = data['Status'].astype('category').cat.codes  # แปลงเป็นตัวเลขอัตโนมัติ

# กำหนดฟีเจอร์ที่ใช้ทำนาย
X = data[['Age', 'Length_of_Service', 'Salary', 'Gender', 'Marital_Status']]
y = data['Status']

# แบ่งข้อมูลเป็นชุด train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# รายชื่อโมเดลที่ใช้
models = {
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=2),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naïve Bayes": GaussianNB(var_smoothing=2e-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
}

# สร้างโฟลเดอร์เก็บโมเดล (ถ้ายังไม่มี)
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# เทรนและบันทึกโมเดลที่ดีที่สุด
accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)  # ฝึกโมเดล
    y_pred = model.predict(X_test)  # ทำนายผล
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.2f}%")

    # บันทึกโมเดลลงไฟล์ .pkl
    model_path = os.path.join(model_dir, f"{name}.pkl")
    joblib.dump(model, model_path)

# ค้นหาโมเดลที่ดีที่สุด
best_accuracy = max(accuracy_results.values())
best_models = {name: models[name] for name, acc in accuracy_results.items() if acc == best_accuracy}

print("\n🏆 Best Models:")
for name, acc in accuracy_results.items():
    if acc == best_accuracy:
        print(f"- {name} with Accuracy: {acc:.2f}%")

# ✅ โหลดเฉพาะโมเดลที่ดีที่สุดจากไฟล์ .pkl
loaded_best_models = {}
for name in best_models.keys():
    model_path = os.path.join(model_dir, f"{name}.pkl")
    loaded_best_models[name] = joblib.load(model_path)

# ✅ Voting Classifier (รวมโมเดลที่ดีที่สุด)
VOTE_model = VotingClassifier(estimators=list(loaded_best_models.items()), voting='hard')
VOTE_model.fit(X_train, y_train)
y_pred_vote = VOTE_model.predict(X_test)
accuracy_vote = accuracy_score(y_test, y_pred_vote) * 100
print(f"\n✅ Voting Classifier (Best Models Only) Accuracy: {accuracy_vote:.2f}%")

# ✅ Stacking Classifier (ใช้โมเดลที่ดีที่สุดร่วมกัน)
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Meta-Model
STACK_model = StackingClassifier(estimators=list(loaded_best_models.items()), final_estimator=meta_model, passthrough=True)
STACK_model.fit(X_train, y_train)
y_pred_stack = STACK_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack) * 100
print(f"\n🚀 Stacking Classifier (Ensemble of Best Models) Accuracy: {accuracy_stack:.2f}%")

# บันทึกโมเดล Ensemble ใหม่ลงในไฟล์ .pkl
joblib.dump(STACK_model, 'Ensemble_Model.pkl')
print(f"\n✅ โมเดล Ensemble (Stacking) ถูกบันทึกลงไฟล์ 'Ensemble_Model.pkl'")

# บันทึกโมเดล Ensemble ใหม่ลงในไฟล์ .pkl และลงในโฟลเดอร์ saved_models
# ensemble_model_path = os.path.join(model_dir, 'Ensemble_Model.pkl')
# joblib.dump(STACK_model, ensemble_model_path)
# print(f"\n✅ โมเดล Ensemble (Stacking) ถูกบันทึกลงไฟล์ 'Ensemble_Model.pkl'")

# ถ้าคุณต้องการบันทึก Voting Classifier แทน Stacking ให้บันทึกไฟล์นี้:
# joblib.dump(VOTE_model, ensemble_model_path)
# print(f"\n✅ โมเดล Ensemble (Voting) ถูกบันทึกลงไฟล์ 'Ensemble_Model.pkl'")
