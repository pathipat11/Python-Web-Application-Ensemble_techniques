import pandas as pd
import joblib

# โหลดโมเดล Decision Tree จากไฟล์ pkl
dt_model = joblib.load('/content/Ensemble_Model.pkl')

# ข้อมูลใหม่ที่ต้องการทดสอบ
input_data = {
    'Age': 39,  # อายุ
    'Length_of_Service': 10,  # ระยะเวลาทำงาน (ปี)
    'Salary': 47500,  # เงินเดือน
    'Gender': 0,  # 0 = Male, 1 = Female
    'Marital_Status': 0  # 0 = Single, 1 = Married
}

# แปลงข้อมูลใหม่เป็น DataFrame
input_df = pd.DataFrame([input_data])

# ทำนายผล
prediction = dt_model.predict(input_df)

# แสดงผลการทำนาย
print(f"Prediction: {prediction[0]}")

# ถ้าผลลัพธ์ 0 = ไม่ได้ทำงาน, 1 = ทำงาน
if prediction[0] == 0:
    print("สถานะ: ไม่ได้ทำงาน")
else:
    print("สถานะ: ทำงาน")