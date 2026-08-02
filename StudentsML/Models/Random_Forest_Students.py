# Importing requered models

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

# Importing dataset

df = pd.read_csv("Dataset/Students.csv")
df = df.drop('parental_education_level', axis=1)

# Assigning features and lables

X = df.drop(columns=['exam_score', 'student_id'])
y = df['exam_score']

# Splitting training and testing data

x_train,x_test,y_train,y_test = train_test_split(X,y ,test_size=0.2 ,random_state=42)

# Making Preprocessor and Pipeline

preprocessor = ColumnTransformer(
                                  transformers=
                                 [
                                    ('cat',OneHotEncoder(),
                                    X.select_dtypes(include='object').columns),

                                    ('num',StandardScaler(),
                                    X.select_dtypes(include='number').columns)
                                    ]
                                )

pipeline = Pipeline([('preprocessor',preprocessor),
                     ('model', RandomForestRegressor(n_estimators=300,max_depth=30))])

# Fitting the model

pipeline.fit(x_train,y_train)

# Saving and Loading model

joblib.dump(pipeline,"Models/Random_Forest_Students.pkl")

Loaded_Pipeline = joblib.load("Models/Random_Forest_Students.pkl") 

# Metrics Check

print("=============METRICS=============")

y_pred = Loaded_Pipeline.predict(x_test)

r2 = r2_score(y_test,y_pred)
Cross_Validation = cross_val_score(Loaded_Pipeline,X,y,cv=5)
MSE = mean_squared_error(y_test,y_pred)

print('R2 Score : ',round(r2,2))
print()

print('MSE Score : ',round(MSE,2))
print()

print('Cross_Validation Score : \n\n',Cross_Validation)
print()

print("Average Cross_Validation : ",Cross_Validation.mean())
print()

#Feature Importance

print("=============FEATURE_IMPORTANCE=============")

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = pipeline.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.savefig("Outputs/Graphs/FeatureImportance.png")
plt.close()


print('=============END=============')