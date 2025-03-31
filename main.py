import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="explortary data analysis", layout="wide")

# Utility function to download DataFrame as CSV
def download_df(df, filename="processed_data.csv"):
    csv = df.to_csv(index=False)
    st.download_button(label="Download Processed Data", data=csv, file_name=filename, mime="text/csv")

# Main page
st.title("Upload, Clean, and Visualize Your Dataset")
st.write("Upload a CSV file to automatically clean and visualize the data.")

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
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        # Display cleaned data
        st.success("File uploaded and automatically cleaned!")
        st.write("### Processed Dataset Preview")
        st.dataframe(df.head())
        
        # Automatic Visualizations
        st.write("### Automatic Visualizations")
        
        # Get column types
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numerical Visualizations
        if numerical_cols:
            st.write("#### Numerical Data Visualizations")
            # Histogram
            fig, axes = plt.subplots(1, len(numerical_cols), figsize=(5*len(numerical_cols), 4))
            if len(numerical_cols) == 1:
                axes = [axes]  # Make it iterable if only one column
            for ax, col in zip(axes, numerical_cols):
                df[col].hist(ax=ax)
                ax.set_title(f'Histogram of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Box Plot
            fig, ax = plt.subplots(figsize=(max(6, len(numerical_cols)*0.5), 4))
            df[numerical_cols].boxplot(ax=ax)
            ax.set_title('Box Plots of Numerical Columns')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation Heatmap (if more than one numerical column)
            if len(numerical_cols) > 1:
                fig, ax = plt.subplots(figsize=(6, 5))
                corr = df[numerical_cols].corr()
                cax = ax.matshow(corr, cmap='coolwarm')
                fig.colorbar(cax)
                ax.set_xticks(range(len(numerical_cols)))
                ax.set_yticks(range(len(numerical_cols)))
                ax.set_xticklabels(numerical_cols, rotation=45)
                ax.set_yticklabels(numerical_cols)
                ax.set_title('Correlation Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
        
        # Categorical Visualizations
        if categorical_cols:
            st.write("#### Categorical Data Visualizations")
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Bar Chart of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        # Mixed Visualizations (if both types exist)
        if numerical_cols and categorical_cols:
            st.write("#### Mixed Data Visualizations")
            # Use the categorical column with fewest unique values
            cat_col = min(categorical_cols, key=lambda col: df[col].nunique())
            for num_col in numerical_cols:
                fig, ax = plt.subplots(figsize=(6, 4))
                df.boxplot(column=num_col, by=cat_col, ax=ax)
                ax.set_title(f'Box Plot of {num_col} by {cat_col}')
                ax.set_xlabel(cat_col)
                ax.set_ylabel(num_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        # Download option
        download_df(df, "cleaned_data.csv")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")