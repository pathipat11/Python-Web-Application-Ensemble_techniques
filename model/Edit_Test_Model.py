import pandas as pd
import joblib

# โหลดโมเดลที่บันทึกไว้
model_path = '/content/Edit_Ensemble_Model.pkl'
dt_model = joblib.load(model_path)

# ข้อมูลใหม่ที่ต้องการทดสอบ
input_data = {
    'Age': 39,
    'Length_of_Service': 10,
    'Salary': 47500,
    'Gender': 0,  # 0 = Male, 1 = Female
    'Marital_Status': 0  # 0 = Single, 1 = Married
}

# แปลงข้อมูลใหม่เป็น DataFrame
input_df = pd.DataFrame([input_data])

# ทำนายผล
prediction = dt_model.predict(input_df)

# แสดงผลการทำนาย
print(f"Prediction: {prediction[0]}")
if prediction[0] == 0:
    print("สถานะ: ไม่ได้ทำงาน")
else:
    print("สถานะ: ทำงาน")
