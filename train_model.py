# train_model.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Define all possible categories
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Royal Challengers Bangalore', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam',
    'Indore', 'Durban', 'Chandigarh', 'Delhi', 'Dharamsala',
    'Ahmedabad', 'Chennai', 'Ranchi', 'Nagpur', 'Mohali', 'Pune',
    'Bengaluru', 'Jaipur', 'Port Elizabeth', 'Centurion', 'Raipur',
    'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London',
    'Abu Dhabi', 'Kimberley', 'Bloemfontein'
]

# Mock data (replace with actual data)
data = {
    'batting_team': ['Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders'],
    'bowling_team': ['Mumbai Indians', 'Sunrisers Hyderabad', 'Mumbai Indians'],
    'city': ['Hyderabad', 'Mumbai', 'Kolkata'],
    'runs_left': [100, 80, 60],
    'balls_left': [60, 50, 40],
    'wickets_remaining': [5, 4, 3],
    'total_runs_x': [200, 180, 160],
    'crr': [6.0, 7.0, 8.0],
    'rrr': [8.0, 9.0, 10.0]
}
df = pd.DataFrame(data)
y = [1, 0, 1]  # Target values

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('trf', OneHotEncoder(categories=[teams, teams, cities]), ['batting_team', 'bowling_team', 'city'])
    ],
    remainder='passthrough'
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the model
pipe.fit(df, y)

# Save the model
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Model trained and saved as 'pipe.pkl'")
