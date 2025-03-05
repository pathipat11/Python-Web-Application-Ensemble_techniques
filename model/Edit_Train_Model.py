import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('/content/data1_updated.csv')

# แปลงข้อมูลเพศ (Gender) และสถานะการสมรส (Marital_Status) เป็นตัวเลข
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Marital_Status'] = data['Marital_Status'].map({'Single': 0, 'Married': 1})

# ตรวจสอบว่าคอลัมน์ 'Status' ต้องการแปลงเป็นตัวเลขหรือไม่
if data['Status'].dtype == 'object':
    data['Status'] = data['Status'].astype('category').cat.codes  # แปลงเป็นตัวเลขอัตโนมัติ

# กำหนดฟีเจอร์ที่ใช้ทำนาย
X = data[['Age', 'Length_of_Service', 'Salary', 'Gender', 'Marital_Status']]

# เป้าหมาย (Target)
y = data['Status']

# แบ่งข้อมูลเป็นชุด train และ test (90-10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ฟังก์ชันสำหรับทำ Grid Search
def tune_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# กำหนดค่าพารามิเตอร์สำหรับการปรับจูน
param_grids = {
    "kNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9, 11]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {"criterion":['gini', 'entropy'], "max_depth": [5, 10, 20, 30, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 3, 5], "max_features": [None, "sqrt", "log2"], "ccp_alpha": [0.0, 0.0001, 0.001, 0.01], "min_impurity_decrease": [0.0, 0.0001, 0.001]}),
    "Logistic Regression": (LogisticRegression(max_iter=5000), {"C": [0.1, 0.2, 0.3, 1, 5, 10], "solver": ['lbfgs']}),
    "Naïve Bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}),
    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}),
    "ANN": (MLPClassifier(max_iter=1000), {"hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "activation": ['relu', 'tanh']}),
    "AdaBoost": (AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), random_state=42), {"n_estimators": [50,60,70, 100, 200], "learning_rate": [0.1, 0.5, 1.0]})
}

# เก็บค่า accuracy และพารามิเตอร์ที่ดีที่สุด
best_models = {}
accuracy_results = {}

# ทำการ Grid Search และ Train Model
for name, (model, param_grid) in param_grids.items():
    print(f"\n🔍 Tuning {name}...")
    best_model, best_params = tune_model(model, param_grid)
    best_models[name] = best_model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_results[name] = accuracy
    print(f"✅ {name} Best Params: {best_params}")
    print(f"🎯 {name} Accuracy: {accuracy:.2f}%")

# เลือกโมเดลที่ดีที่สุด
best_accuracy = max(accuracy_results.values())
best_selected_models = {name: model for name, model in best_models.items() if accuracy_results[name] == best_accuracy}

print("\n🏆 Best Models:")
for name, acc in accuracy_results.items():
    if acc == best_accuracy:
        print(f"- {name} with Accuracy: {acc:.2f}%")

# **Voting Classifier** รวมเฉพาะโมเดลที่ดีที่สุด
VOTE_model = VotingClassifier(estimators=list(best_selected_models.items()), voting='hard')
VOTE_model.fit(X_train, y_train)
y_pred_vote = VOTE_model.predict(X_test)
accuracy_vote = accuracy_score(y_test, y_pred_vote) * 100
print(f"\n✅ Voting Classifier Accuracy: {accuracy_vote:.2f}%")

# **Stacking Classifier** รวมโมเดลที่ดีที่สุด + Meta-Model
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)  # ใช้ Random Forest เป็น Meta Model
STACK_model = StackingClassifier(estimators=list(best_selected_models.items()), final_estimator=meta_model, passthrough=True)
STACK_model.fit(X_train, y_train)
y_pred_stack = STACK_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack) * 100
print(f"\n🚀 Stacking Classifier Accuracy: {accuracy_stack:.2f}%")