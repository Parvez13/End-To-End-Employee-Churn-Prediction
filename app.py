import numpy as np
import pandas as pd
import streamlit as st
import pickle
import streamlit.components.v1 as stc

# HTML for a custom title
custom_title = """
<div style="font-size:40px;font-weight:bolder;background-color:#fff;padding:30px;
border-radius:20px;border:10px solid #464e5f;text-align:center;">
    Employee Job Satisfaction
</div>
"""

# Load the trained model and other required objects
model = pickle.load(open('saved_models/0/model/model.pkl', 'rb'))
encoder = pickle.load(open('saved_models/0/target_encoder/target_encoder.pkl', 'rb'))
transformer = pickle.load(open('saved_models/0/transformer/transformer.pkl', 'rb'))

def main():
    # Create a Streamlit web app
    stc.html(custom_title)
    # st.title("Feature Input")

    # Create input fields for user input
    education = st.selectbox('Please select education level', ('Bachelors', 'Masters', 'PhD'))

    joining_year = st.text_input('Enter Joining Year', 2020)
    joining_year = int(joining_year)

    city = st.selectbox('Please select City', ('Bangalore', 'Pune', 'New Delhi'))

    payment_tier = st.selectbox('Please select payment tier', ('Tier 1', 'Tier 2', 'Tier 3'))

    age = st.text_input("Enter Age", 25)
    age = int(age)
    
    gender = st.selectbox('Please select gender', ('Male', 'Female'))

    ever_benched = st.selectbox('Ever benched', ('Yes', 'No'))

    experience_in_current_domain = st.text_input('Experience in Current Domain', 0)
    experience_in_current_domain = int(experience_in_current_domain)

    # Create a dictionary with user input
    user_input = {
        'Education': education,
        'JoiningYear': joining_year,
        'City': city,
        'PaymentTier': payment_tier,
        'Age': age,
        'Gender': gender,
        'EverBenched': ever_benched,
        'ExperienceInCurrentDomain': experience_in_current_domain
    }

    # Create a DataFrame from user input
    input_df = pd.DataFrame(user_input, index=[0])

    # Encode and transform the input data
    input_df['Education'] = input_df['Education'].map({'Bachelors': 2, 'Masters': 1, 'PHD': 0})
    input_df['City'] = input_df['City'].map({"Bangalore": 0, "Pune": 1, "New Delhi": 2})
    input_df['PaymentTier'] = input_df['PaymentTier'].map({'Tier1': 0, 'Tier2': 1, 'Tier3': 2})
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['EverBenched'] = input_df['EverBenched'].map({'Yes': 1, 'No': 0})

    # Transform the input data
    input_df = transformer.transform(input_df)

    if st.button("Show Result"):
        # Make predictions using the model
        predicted = model.predict(input_df)

        # Display the result
        # Display the result
        if predicted[0] == 1:
            st.warning("Employee may leave")
        else:
            st.success("Employee is not likely to leave")

if __name__ == "__main__":
    main()
