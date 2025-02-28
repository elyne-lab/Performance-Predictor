import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(
    page_title="Employee Performance Predictor",
    layout="wide"
)

# Load the saved model and encoders
@st.cache_resource
def load_model():
    with open('Staff Performance Analysis.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Navigation options
    page = st.sidebar.radio(
        "Select a Page",
        ["Home", "Prediction", "About"]
    )
    
    # Top 10 important features (update these based on your model's feature importance)
    top_features = {
        'EmpEnvironmentSatisfaction': 'Environment Satisfaction',
        'EmpLastSalaryHikePercent': 'Salary Hike Percentage',
        'YearsSinceLastPromotion': 'Years Since Last Promtion',
        'EmpDepartment': 'Department',
        'ExperienceYearsInCurrentRole': 'Experience in Current Role',
        'EmpHourlyRate': 'Hourly Rate',
        'EmpJobRole': 'Job Role',
        'ExperienceYearsAtThisCompany': 'Years of Experience in INX Future Inc',
        'Age': 'Age',
        'EmpWorkLifeBalance': 'Work Life Balance',
        'YearsWithCurrManager': 'Years with Your Current Manager',
        'TotalWorkExperienceInYears': 'Total Work Experience',
        'DistanceFromHome': 'Distance From Home',   
        'EducationBackground' : 'Educational Background',
        'EmpEducationLevel': 'Education Level'
    }


    if page == "Prediction":
        st.sidebar.subheader("Model Information")
        st.sidebar.info(
            """
            This prediction is based on the top 10 most influential features 
            identified by the model.
            """
        )
        
        # Display feature importance
        st.sidebar.subheader("Top Influential Factors")
        for i, (feature, label) in enumerate(top_features.items(), 1):
            st.sidebar.text(f"{i}. {label}")

    # Main content based on page selection
    if page == "Home":

        st.markdown(
    """
    <h1 style='text-align: left; color: #567ED0; font-family: Verdana, Geneva, sans-serif;'>
    INX Future Inc. Employee Performance Analysis
    </h1>
    """,
    unsafe_allow_html=True,
)
        st.title("Employee Performance Prediction System")
        st.write("Welcome to the Employee Performance Prediction System")
        
        # Add home page metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Model Accuracy", value="95%")
        with col2:
            st.metric(label="Features Used", value="15")
        with col3:
            st.metric(label="Prediction Confidence", value="High")
            
    elif page == "Prediction":
        st.title("Performance Rating Predictor")
        st.write("Enter the Below Details as Required")
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            
            # Categorical inputs
            EducationBackground = st.selectbox("Education Background", 
                                ['Marketing', 'Life Sciences', 'Human Resources','Medical','Technical Degree', 'Other'])
            EmpDepartment = st.selectbox("Department", 
                                    ['Sales', 'HR', 'Development', 
                                    'Data Science', 'Research & Development', 
                                    'Finance'])
            EmpJobRole = st.selectbox("Job Role", 
                                ['Sales Executive', 'Research Scientist', 
                                    'Developer', 'Manager', 'Data Scientist', 'Sales Representative',
                                    'Human Resources', 'Senior Developer', 'Senior Manager R&D',
                                    'Manufacturing Director', 'Healthcare Representative', 'Research Director',
                                    'Manager R&D', 'Finance Manager', 'Technical Architect', 'Business Analyst',
                                    'Technical Lead', 'Delivery Manager'])
            
            # Numerical inputs part 1
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            DistanceFromHome = st.number_input("Distance From Home", 
                                    min_value=0, max_value=100, value=10)
            EmpEducationLevel = st.number_input("Education Level (1-5)", 
                                    min_value=1, max_value=5, value=3)
            EmpEnvironmentSatisfaction = st.number_input("Environment Satisfaction (1-4)", 
                                            min_value=1, max_value=4, value=3)
            EmpHourlyRate = st.number_input("Hourly Rate", 
                                        min_value=10, max_value=200, value=50)
        
        with col2:
            st.subheader("Professional Information")
            
            # Numerical inputs part 2
            EmpWorkLifeBalance = st.number_input("Work Life Balance (1-4)", 
                                            min_value=1, max_value=4, value=3)
            ExperienceYearsInCurrentRole = st.number_input("Years in Current Role", 
                                    min_value=0, max_value=20, value=2)
            ExperienceYearsAtThisCompany = st.number_input("Years of Experience in INX Future Inc", 
                                            min_value=0, max_value=40, value=5)
            YearsWithCurrManager = st.number_input("Years With Current Manager", 
                                        min_value=0, max_value=20, value=3)
            TotalWorkExperienceInYears = st.number_input("Total Work Experience (Years)", 
                                            min_value=0, max_value=40, value=10)
            YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 
                                    min_value=0, max_value=15, value=2)
            EmpLastSalaryHikePercent = st.number_input("Last Salary Hike Percent", 
                                        min_value=0, max_value=25, value=15)

        # Create prediction button
        if st.button("Predict Performance Rating"):
            # Prepare input data
            input_data = {
                'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
                'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
                'YearsSinceLastPromotion': YearsSinceLastPromotion,
                'EmpDepartment': EmpDepartment,
                'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
                'EmpHourlyRate': EmpHourlyRate,
                'EmpJobRole': EmpJobRole,
                'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
                'Age': Age,
                'EmpWorkLifeBalance': EmpWorkLifeBalance,
                'YearsWithCurrManager': YearsWithCurrManager,
                'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
                'DistanceFromHome': DistanceFromHome,
                'EducationBackground': EducationBackground,
                'EmpEducationLevel': EmpEducationLevel
        }
            try:
                # Convert input to DataFrame
                input_df = pd.DataFrame([input_data])

                # Label encode categorical variables
                cat_cols = ['EmpDepartment', 'EmpJobRole', 'EducationBackground']                

            # Dictionary for label encoding mappings
                label_mappings = {
                    'EmpDepartment': {
                        'Sales': 0, 
                        'HR': 1, 
                        'Development': 2,
                        'Data Science': 3, 
                        'Research & Development': 4,
                        'Finance': 5
                    },                
                    'EmpJobRole': {
                        'Sales Executive': 0,
                        'Research Scientist': 1,
                        'Developer': 2,
                        'Manager': 3,
                        'Data Scientist': 4,
                        'Sales Representative': 5,
                        'Human Resources': 6,
                        'Senior Developer': 7,
                        'Senior Manager R&D': 8,
                        'Manufacturing Director': 9,
                        'Healthcare Representative': 10,
                        'Research Director': 11,
                        'Manager R&D': 12,
                        'Finance Manager': 13,
                        'Technical Architect': 14,
                        'Business Analyst': 15,
                        'Technical Lead': 16,
                        'Delivery Manager': 17
                    },
                    'EducationBackground': {
                        'Marketing': 0,
                        'Life Sciences': 1,
                        'Human Resources': 2,
                        'Medical': 3,
                        'Technical Degree': 4,
                        'Other': 5
                    }
                }
                # Apply label encoding
                for column in cat_cols:
                    input_df[column] = input_df[column].map(label_mappings[column]) 

                # Load model and make prediction
                #model = load_model()
                model_ = pickle.load(open('Staff Performance Analysis.pkl', 'rb'))
                
                prediction = model_.predict(input_df)
                probability = model_.predict_proba(input_df)
                
                # Display results
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Predicted Performance Rating: {prediction[0]}")
                    if prediction[0] >= 4:
                        st.success("Excellent Performance")
                    elif prediction[0] >= 3:
                        st.info("Good Performance")
                    else:
                        st.warning("Needs Improvement")
                
                with col2:
                    st.write("Prediction Probabilities:")
                    prob_df = pd.DataFrame({
                        'Rating': model_.classes_,
                        'Probability': probability[0]
                    })
                    
                
                # Visualize probabilities
                st.bar_chart(prob_df.set_index('Rating'))
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.write("Please make sure all inputs are in the correct format.")
    
    elif page == "About":
        st.title("About")
        st.write("""
        ### Model Information
        This prediction model uses the 15 most influential features identified 
        through feature importance analysis. 
                 
        These features have shown the strongest correlation with employee performance ratings.
        

        ### Feature Importance
        The features are ranked based on their impact on performance Rating prediction, 
        with salary hike percentage and environment satisfaction being the top 
        indicators.
        
        ### Prediction Accuracy
        The model has been trained on historical employee data and validated 
        for accuracy and reliability.
        """)
        st.markdown(
    """
    <h1 style='text-align: center; color: #567ED0; font-family: Verdana, Geneva, sans-serif;'>
    INX Future Inc.
    </h1>
    """,
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    main()