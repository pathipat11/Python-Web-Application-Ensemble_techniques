# Python Web Application with Ensemble Learning

## Overview
This project is a web application that uses **Ensemble Learning** techniques to train and evaluate multiple machine learning models for predicting employee status (Still Employed or Resigned). The best-performing models are selected and combined using **Voting Classifier** and **Stacking Classifier** to improve prediction accuracy. The trained model is then deployed using Flask for real-time predictions.

## Technologies Used
- **Python** (Machine Learning and Web Development)
- **Flask** (Web Framework)
- **scikit-learn** (Machine Learning Library)
- **pandas** (Data Processing)
- **joblib** (Model Saving & Loading)
- **HTML & CSS** (Frontend for the web app)
- **Render** (Hosting Platform)

## Project Structure
```
├── model
│   ├── Train_model.py       # Train multiple models and select the best ones
│   ├── Load_Train_Best_Model.py  # Load best models and create ensemble models
│   ├── Test_Model.py        # Test the trained model with sample input
│   ├── saved_models         # Directory for storing trained models
│
├── data
│   ├── data.csv             # Employee data used for training/testing
│
├── templates
│   ├── index.html           # Frontend for user input and displaying results
│
├── app.py                   # Flask web application for predictions
├── README.md                # Project documentation
├── requirements.txt         # Dependencies required for the project
```

## Model Training (Train_model.py)
1. Load employee data from `data.csv`.
2. Convert categorical data (`Gender`, `Marital_Status`, `Status`) into numerical values.
3. Split data into **90% training** and **10% testing**.
4. Train multiple models including:
   - k-Nearest Neighbors (kNN)
   - Decision Tree
   - Logistic Regression
   - Naïve Bayes
   - Random Forest
   - Artificial Neural Network (ANN)
   - AdaBoost
5. Evaluate accuracy and select the best-performing models.
6. Combine the best models using **Voting Classifier** and **Stacking Classifier**.

## Model Loading & Saving (Load_Train_Best_Model.py)
- The best models are saved in `saved_models/` as `.pkl` files using `joblib`.
- The ensemble models (Voting & Stacking) are retrained and saved for deployment.

## Testing the Model (Test_Model.py)
- Loads the trained `Ensemble_Model.pkl`.
- Accepts new employee data as input.
- Predicts the employee’s status (`Still Employed` or `Resigned`).

## Web Application (app.py)
- Uses **Flask** to create a web-based prediction tool.
- Accepts user input through an HTML form.
- Predicts and displays results in a user-friendly interface.
- Hosted on **Render** for online access.

## How to Run the Project
### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Train the Model
```
python model/Train_model.py
```

### 3. Load and Save Best Models
```
python model/Load_Train_Best_Model.py
```

### 4. Run the Flask Web App
```
python app.py
```

- Open `http://127.0.0.1:5000/` in a browser.
- Enter employee details and get predictions.

## Future Improvements
- Improve model accuracy by feature engineering.
- Use deep learning models for better predictions.
- Deploy as a cloud-based API for wider accessibility.

---
**Author:** Pathipat.Mattra@gmail.com