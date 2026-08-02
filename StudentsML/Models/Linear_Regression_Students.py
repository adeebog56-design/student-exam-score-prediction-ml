# Importing requered models

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
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
                     ('model', LinearRegression())])

# Fitting the model

pipeline.fit(x_train,y_train)

# Saving and Loading model

joblib.dump(pipeline ,"Models/Linear_Regression_Students.pkl")

Loaded_Pipeline = joblib.load("Models/Linear_Regression_Students.pkl") 

# Metrics Check

print("=============METRICS=============")
print()

y_pred = Loaded_Pipeline.predict(x_test)

r2 = r2_score(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
Cross_Validation = cross_val_score(Loaded_Pipeline,X,y,cv=5)

print('R2 Score : ',round(r2,2))
print()

print('MSE Score : ',round(MSE,2))
print()

print('Cross_Validation Score : \n\n',Cross_Validation)
print()

print("Average Cross_Validation : ",Cross_Validation.mean())
print()

print('=============END=============')