# Importing required modules

import pandas as pd
import joblib

# Loading Models

Linear_Regression = joblib.load("Models/Linear_Regression_Students.pkl")
Random_Forest     = joblib.load("Models/Random_Forest_Students.pkl")

print("========== Student Exam Score Prediction ==========\n")

# Sample Student

sample = pd.DataFrame({

    "age":[20],
    "gender":["Female"],
    "study_hours_per_day":[6.5],
    "social_media_hours":[2.0],
    "netflix_hours":[1.5],
    "part_time_job":["No"],
    "attendance_percentage":[94.0],
    "sleep_hours":[7.5],
    "diet_quality":["Good"],
    "exercise_frequency":[4],
    "internet_quality":["Good"],
    "mental_health_rating":[8],
    "extracurricular_participation":["Yes"]

})

# Prediction

lr_prediction = Linear_Regression.predict(sample)[0]
rf_prediction = Random_Forest.predict(sample)[0]

# Limiting Linear Regression output between 0 and 100

lr_prediction = max(0, min(100, lr_prediction))

print("============= RESULT =============")
print()

print(f"Linear Regression : {lr_prediction:.2f}")
print(f"Random Forest     : {rf_prediction:.2f}")

print()
print("============= END =============")