# Python Web Application with Ensemble Techniques

This project is a machine learning web application that uses **Ensemble Techniques** to train multiple models and select the best ones for prediction. The application is developed using **Flask**, and the trained model is deployed on **Render**.

## 🚀 Live Demo
[Click here to access the deployed application](https://python-web-application-ensemble.onrender.com/)

---

## 📌 Project Overview
The goal of this project is to predict employee status (Still Employed or Resigned) based on various factors, such as:
- Age
- Length of Service
- Salary
- Gender
- Marital Status

To achieve high prediction accuracy, the project employs **Ensemble Learning** techniques, specifically:
- **Voting Classifier** (Hard Voting)
- **Stacking Classifier** with a meta-model (Random Forest)

The best-performing models are selected based on **accuracy** and then used in the ensemble model.

---

## ⚙️ Technologies Used
- **Python** (Machine Learning Model Training)
- **Flask** (Backend Web Framework)
- **Scikit-learn** (ML Models & Preprocessing)
- **Pandas** (Data Handling)
- **Joblib** (Model Persistence)
- **HTML & Bootstrap** (Frontend UI)
- **Render** (Cloud Deployment)

---

## 📂 Project Structure
```
📁 project-root/
│── 📁 model/                # Contains trained models
│   │── Train_model.py       # Train multiple models and save the best one
│   │── Load_Train_Best_Model.py  # Load and use the best model
│   │── Test_Model.py        # Test the trained model
│── 📁 templates/            # HTML templates for Flask
│── 📁 static/               # CSS & assets
│── app.py                   # Main Flask application
│── data.csv                 # Dataset used for training
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```

---

## 🛠️ How to Run the Project Locally
### 1️⃣ Clone the repository
```sh
git clone https://github.com/your-repo-url.git
cd project-root
```

### 2️⃣ Install dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Train the model
Run the script to train models and save the best one:
```sh
python model/Train_model.py
```

### 4️⃣ Run the Flask app
```sh
python app.py
```

The application will be available at `http://127.0.0.1:5000/`

---

## 🔥 Future Improvements
- Improve model performance using feature engineering
- Implement more advanced ensemble techniques
- Enhance UI design for better user experience
- Deploy on additional cloud platforms

---

## 📞 Contact
**Author:** pathipat.mattra@gmail.com & pathipat.m@kkumail.com