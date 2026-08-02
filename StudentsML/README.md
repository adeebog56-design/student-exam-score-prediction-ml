# Student Exam Score Prediction

A Machine Learning project that predicts student exam scores using academic, lifestyle, and personal factors.

## Models Used

* Linear Regression
* Random Forest Regressor

## Features

* Age
* Gender
* Study Hours
* Social Media Hours
* Netflix Hours
* Part-Time Job
* Attendance Percentage
* Sleep Hours
* Diet Quality
* Exercise Frequency
* Internet Quality
* Mental Health Rating
* Extracurricular Participation

## Performance

| Model             | R² Score |    MSE    | Cross Validation |
| ----------------- | :------: | :-------: | :--------------: |
| Linear Regression | **0.90** | **26.30** |    **0.8961**    |
| Random Forest     |   0.85   |   38.22   |      0.8549      |

**Best Model:** Linear Regression

## Project Structure

* `EDA.py`
* `Linear_Regression.py`
* `Random_Forest.py`
* `predict.py`
* `Models/`
* `Outputs/Graphs/`

## Libraries

* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

## 🚀 Run

```bash
python predict.py
```

---

Made by Aiden 🚀

