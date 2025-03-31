import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Simple Data Cleaner", layout="wide")

# Utility function to download DataFrame as CSV
def download_df(df, filename="processed_data.csv"):
    csv = df.to_csv(index=False)
    st.download_button(label="Download Processed Data", data=csv, file_name=filename, mime="text/csv")

# Main page
st.title("Upload and Clean Your Dataset")
st.write("Upload a CSV file to automatically clean and download the processed data.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the file
        df = pd.read_csv(uploaded_file)
        
        # Automatic cleaning
        # 1. Handle missing values
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['float64', 'int64']:
                    skewness = abs(df[column].skew())
                    if skewness > 1:
                        df[column] = df[column].fillna(df[column].median())
                    else:
                        df[column] = df[column].fillna(df[column].mean())
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
        
        # 2. Handle outliers for numerical columns
        numerical_columns = df.select_dtypes(include=['number']).columns
        for column in numerical_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Cap outliers
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        # Display results
        st.success("File uploaded and automatically cleaned!")
        st.write("### Processed Dataset Preview")
        st.dataframe(df.head())
        
        # Provide download option
        download_df(df, "cleaned_data.csv")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")