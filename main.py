import base64
import streamlit as st
import pickle
import pandas as pd

@st.cache_data
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error reading the image file: {e}")
        return None

img = get_img_as_base64("background.jpg")
if img is not None:
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    width: 100%;
    height:100%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: cover;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
    # **IPL VICTORY PREDICTOR**            
""")

teams = ['--- select ---',
         'Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders',
         'Royal Challengers Bangalore', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam',
          'Indore', 'Durban', 'Chandigarh', 'Delhi', 'Dharamsala',
          'Ahmedabad', 'Chennai', 'Ranchi', 'Nagpur', 'Mohali', 'Pune',
          'Bengaluru', 'Jaipur', 'Port Elizabeth', 'Centurion', 'Raipur',
          'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London',
          'Abu Dhabi', 'Kimberley', 'Bloemfontein']

# Load the model
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("The file 'pipe.pkl' was not found. Please make sure the file is in the correct directory.")
    pipe = None
except pickle.UnpicklingError:
    st.error("The file 'pipe.pkl' is not a valid pickle file or is corrupted.")
    pipe = None
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    pipe = None

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', teams)

with col2:
    if batting_team == '--- select ---':
        bowling_team = st.selectbox('Select Bowling Team', teams)
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('Select Bowling Team', filtered_teams)

selected_city = st.selectbox('Select Venue', cities)

target = st.number_input('Target')

col1, col2, col3 = st.columns(3)

with col1:
    score = st.number_input('Score')
with col2:
    overs = st.number_input("Over Completed")
with col3:
    wickets = st.number_input("Wickets down")

if st.button('Predict Winning Probability'):
    if pipe is not None:
        try:
            runs_left = target - score
            balls_left = 120 - (overs * 6)
            wickets = 10 - wickets
            crr = score / overs
            rrr = runs_left / (balls_left / 6)

            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_remaining': [wickets],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            result = pipe.predict_proba(input_data)

            loss = result[0][0]
            win = result[0][1]
            st.header(batting_team + " = " + str(round(win * 100)) + "%")
            st.header(bowling_team + " = " + str(round(loss * 100)) + "%")
        except Exception as e:
            st.header("Some error occurred.. Please check your inputs!")
            st.error(f"Error details: {e}")
    else:
        st.error("Model not loaded. Cannot predict the winning probability.")
