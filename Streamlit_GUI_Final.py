import io
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

DATA_URL = (
    "C:/Users/Admin/OneDrive/Desktop/Freelancing_projects/Gauri_PDA_GUI/bank.csv"
)

st.title(":bank: EDA for Bank Marketing Campaign ")
st.sidebar.title("Exploratory Data Analysis")
st.markdown("This dataset contains banking marketing campaign data and we can use "
            "it to optimize marketing campaigns to attract more customers to term deposit subscription :dollar: .")
st.markdown("Our goal is finding out customer segments, using data for customers, who subscribed to term deposit. " 
            "This helps to identify the profile of a customer, "
            "who is more likely to acquire the product and develop more targeted marketing campaigns :tv:")

image = "Data_description.png"
def show_image():
    st.image(image, caption="Data Description")
    
st.write("Click to view the explanation of variables in data :chart: :")
st.button("Show Image", on_click=show_image)
    
st.sidebar.markdown("This is a Streamlit application used "
            "to analyze bank marketing campaign 🐦")


@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

st.sidebar.markdown(f"There are {data.shape[0]} rows and {data.shape[1]} columns in our dataset.")
st.sidebar.subheader("Show columns with unique datatypes")
random_tweet = st.sidebar.radio('DataType', ('Numerical', 'Categorical'))

numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = data.select_dtypes(include=numerics).columns
cat = data.select_dtypes(include='object').columns
df_num = data.select_dtypes(include=numerics)
df_cat = data.select_dtypes(include=['object'])


if random_tweet == 'Numerical':
    st.sidebar.subheader("Numeric Columns")
    st.sidebar.markdown(f"There are {df_num.shape[1]} numerical columns")
    st.sidebar.write(num.tolist())
    # st.sidebar.markdown(print(num))
    
else:
    st.sidebar.subheader("Categorical Columns")
    st.sidebar.markdown(f"There are {df_cat.shape[1]} categorical columns")
    st.sidebar.write(cat.tolist())

st.sidebar.markdown("### Data Exploration")
select = st.sidebar.selectbox('Visualization type', 
                              ['Data Statistics','Boxplot','Interaction'], key='1')

# Function to generate boxplots based on selected variable (Boxplot)
def generate_boxplots(selected_var):
    fig = px.box(data, x='deposit', y=selected_var)
    fig.update_layout(title=f"Boxplot of {selected_var} vs. Deposit")
    st.plotly_chart(fig)
    
# Function to generate scatter plot of selected variables (Interaction)
def generate_scatter_plot(x_col, y_col):
    fig = px.scatter(data, x=x_col, y=y_col)
    fig.update_layout(
        title=f"Scatter Plot of {x_col} vs. {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    st.plotly_chart(fig)

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Let's understand each Variable in data")
    if select == 'Data Statistics':
        st.header("Data Description")
        st.write(data.describe())
        st.markdown("""
                    1. The highest recorded age is 95.
                    It's possible that there might be individuals with ages exceeding 95 in the data.   
                    2. The maximum and minimum values seem reasonable across all columns.  
                    3. There are 11,162 rows of data based on the count.
                    This suggests the possibility of duplicate data or missing values.  
                    4. -1 in pdays possibly means that the client wasn't contacted before or stands for missing data.  
                    Since we are not sure exactly what -1 means I suggest to drop this column, because -1 makes more than 50% of the values of the column.
                    """
                    )
               
    elif select == 'Boxplot':
        variable_options = list(data.columns)[:-1]  # Exclude 'deposit' from options

        # Create horizontal buttons and handle clicks
        selected_var = None
        selected_var = st.selectbox("Select Y-axis Variable to plot Boxplot with Deposit:", num)

        # Display boxplots if a variable is selected
        if selected_var is not None:
            generate_boxplots(selected_var)
            
        st.markdown("The higher the customer's account balance, the greater the chance it will be converted into a deposit. "
                    "However, after a campaign is conducted to customers, it will take a relatively long time for them to make a deposit if the campaign is successful. " 
                    "Therefore, campaigns should be carried out reasonably and in moderation. " 
                    "If done excessively, it can reduce the likelihood of customers making deposits.")
            
    elif select == 'Interaction':
        numeric_cols = num
        # Create column selection menus
        col1_options, col2_options = st.columns(2)
        selected_x_col = col1_options.selectbox("Select X-axis Variable:", numeric_cols)
        selected_y_col = col2_options.selectbox("Select Y-axis Variable:", numeric_cols.copy())

        # Check for valid selection (avoiding plotting the same variable on both axes)
        if selected_x_col != selected_y_col:
            generate_scatter_plot(selected_x_col, selected_y_col)
        else:
            st.write("Please select different variables for X and Y axes.")
        
      
categorical_cols = cat
numeric_cols = num

# Function to generate frequency plot for categorical variable
def generate_frequency_plot(col_name):
    fig = px.bar(data, x=col_name, color='deposit', title=f"Frequency of {col_name}")
    fig.update_layout(xaxis_title=col_name, yaxis_title="Frequency")
    st.plotly_chart(fig)

# Function to generate distribution plot for numerical variable
def generate_distribution_plot(col_name):
    fig = px.histogram(data, x=col_name, title=f"Distribution of {col_name}",color='deposit')
    fig.update_layout(xaxis_title=col_name, yaxis_title="Density")
    st.plotly_chart(fig)

    
st.sidebar.subheader("Distribution and Frequency plots for Both DataTypes")
dist_freq = st.sidebar.radio('DataType', ('Numerical', 'Categorical'),key='4')

if not st.sidebar.checkbox("Close", True, key='8'):
    if dist_freq == 'Numerical':
        st.subheader("Distribution of Numerical Columns")
        cols = st.columns(len(numeric_cols))
        for i, col_name in enumerate(numeric_cols):
            if cols[i].button(col_name,use_container_width=True):
                generate_distribution_plot(col_name)
                
        st.markdown("""
                    1. People who subscribed for term deposit tend to have greater balance and age values.  
                    2. People who subscribed for term deposit tend to have fewer number of contacts during this campaign.
                    """)
    else:
        st.subheader("Frequency of Categorical Columns")
        selected_cat_col = st.selectbox("Select Categorical Variable:", categorical_cols)
        if selected_cat_col:
            generate_frequency_plot(selected_cat_col)
        st.markdown("""
                    1. Customers with 'blue-collar' and 'services' jobs are less likely to subscribe for term deposit.  
                    2. Married customers are less likely to subscribe for term deposit.  
                    3. Customers with 'cellular' type of contact are less likely to subscribe for term deposit.
                    """)
            

# Function to generate correlation heatmap
def generate_correlation_plot():
    corr_matrix = df_num.corr()
    fig = px.imshow(corr_matrix, title="Correlation Heatmap", zmin=-1, zmax=1)
    fig.update_xaxes(title_text="Variable")
    fig.update_yaxes(title_text="Variable")
    fig.update_traces(colorbar_title="Correlation Coefficient")
    st.plotly_chart(fig)

# Function to generate pair plots
def generate_pair_plots():
    fig = sns.pairplot(data=data,hue='deposit',corner=True)
    st.pyplot(fig)

# Function to generate plot based on selected column
def generate_plot(selected_col_1, selected_col_2, selected_col_3='deposit'):
    fig = px.bar(data, x=selected_col_1, y=selected_col_2,color = selected_col_3)  # Assuming 'index' is your index column
    fig.update_layout(title=f"Plot of {selected_col_1} Vs {selected_col_2}")
    st.plotly_chart(fig)
    

st.sidebar.subheader("Multivariate Analysis")
multi_plot = st.sidebar.selectbox('Visualization type', ['Distribution','Correlation', 'Pair Plots'], key='2')
if not st.sidebar.checkbox("Close", True, key='9'):
    if multi_plot == "Correlation":
        generate_correlation_plot()
    elif multi_plot == "Pair Plots":
        generate_pair_plots()
    elif multi_plot == 'Distribution':
        st.header("Data Distribution")
        plot_options = list(data.columns)
        
        # Create horizontal buttons and handle clicks
        selected_col = None
        selected_col_1 = st.selectbox("Select X-axis Variable:", cat)
        selected_col_2 = st.selectbox("Select Y-axis Variable:", num)
        selected_col_3 = st.selectbox("Select hue Variable:", plot_options)
 
        # Display plot if a column is selected
        if selected_col_1 is not None and selected_col_2 is not None:
            generate_plot(selected_col_1, selected_col_2, selected_col_3)
        else:   
            st.write("Please select a visualization type.")
            
st.sidebar.subheader("Recommendations")
recommend = st.sidebar.button("Click for Recommendations", key='10')

if not st.sidebar.checkbox("Close", True, key='close'):
    if recommend:
        st.markdown("### Recommendations")
        st.markdown("""
                    **Age** :  
                        1. Target relatively older people.  
                        2.Convey peace of mind, Safe investment and steady income source as the value proposition.  
                    **Duration** :  
                        1. Try to engage customers and have longer calls.
                        2. Try to connect with customers through social media and influencer marketing.  
                    **Campaign** :   
                        1. Prioritize those customers to who were part of the previous marketing campaigns.
                    """)