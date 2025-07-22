# Customer Churn Prediction - CodSoft Task 3

This project predicts customer churn for a subscription-based service using historical customer data.

## 📌 Objective

To develop a machine learning model that can predict whether a customer will leave the service based on features like credit score, geography, age, tenure, balance, activity, etc.

---

## 📂 Dataset

* **Name:** Churn\_Modelling.csv
* **Source:** Provided by CodSoft
* **Rows:** 10,000
* **Target Variable:** `Exited` (1 = Churned, 0 = Retained)

---

## 🧪 Features Used

* Credit Score
* Geography (France, Germany, Spain)
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

---

## 🛠️ Technologies & Libraries

* Python
* Pandas
* Scikit-learn
* Joblib

---

## 🧠 ML Models Applied

* Logistic Regression
* Random Forest ✅ Best Performer
* Gradient Boosting

---

## 🧾 Results (on test set)

* Accuracy \~85% with Random Forest
* Classification report included in console output

---

## 💾 Output

* Trained model saved as: `model.pkl`

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
├── churn_prediction.py
├── Churn_Modelling.csv
├── model.pkl
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository
2. Place `Churn_Modelling.csv` in the project directory
3. Run `churn_prediction.py`
4. View classification reports in the console
5. Use `model.pkl` for deployment if needed

---

## 📚 Developed For

> **CodSoft Internship - Task 3**

Feel free to use this as a reference for churn prediction projects!
