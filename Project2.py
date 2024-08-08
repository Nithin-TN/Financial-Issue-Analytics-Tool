import streamlit as st
import nltk
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from streamlit_chat import message as st_message

nltk.download('punkt')

# Load and preprocess the dataset
@st.cache_data
def load_data():
    file_path = "C:/Users/Nithins/Downloads/New folder/Consumer_Complaints.csv"
    return pd.read_csv(file_path, nrows=500)

complaint_data = load_data()

# Convert date fields to datetime
complaint_data['Date received'] = pd.to_datetime(complaint_data['Date received'], errors='coerce')
complaint_data['Date sent to company'] = pd.to_datetime(complaint_data['Date sent to company'], errors='coerce')

# Drop rows with NaT in the date columns
complaint_data = complaint_data.dropna(subset=['Date received', 'Date sent to company'])

# Categorize complaint types and resolutions
complaint_data['Product'] = complaint_data['Product'].astype('category')
complaint_data['Company response to consumer'] = complaint_data['Company response to consumer'].astype('category')

# Function to analyze sentiment
def analyze_sentiment(text):
    if pd.isna(text):
        return None
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis
complaint_data['Sentiment'] = complaint_data['Consumer complaint narrative'].apply(analyze_sentiment)

# Drop rows where sentiment is None
complaint_data = complaint_data.dropna(subset=['Sentiment'])

# Function to categorize sentiment
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the categorization
complaint_data['Sentiment Category'] = complaint_data['Sentiment'].apply(categorize_sentiment)

# Define the Streamlit app layout
st.title('Consumer Complaint Analysis Dashboard')

# Sidebar for file upload
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    complaint_data = pd.read_csv(uploaded_file, nrows=500)
    complaint_data['Date received'] = pd.to_datetime(complaint_data['Date received'], errors='coerce')
    complaint_data['Date sent to company'] = pd.to_datetime(complaint_data['Date sent to company'], errors='coerce')
    complaint_data = complaint_data.dropna(subset=['Date received', 'Date sent to company'])
    complaint_data['Product'] = complaint_data['Product'].astype('category')
    complaint_data['Company response to consumer'] = complaint_data['Company response to consumer'].astype('category')
    complaint_data['Sentiment'] = complaint_data['Consumer complaint narrative'].apply(analyze_sentiment)
    complaint_data = complaint_data.dropna(subset=['Sentiment'])
    complaint_data['Sentiment Category'] = complaint_data['Sentiment'].apply(categorize_sentiment)

# Display dataset
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(complaint_data)

# Define available options
options = [
    "Show dashboard for a specific product",
    "Show issues related to a specific product",
    "Show complaints by state for a specific product",
    "Show sentiment distribution for a specific product",
    "Generate a general report"
]

# Chatbot Integration
st.header("Chatbot")

def generate_dashboard(option, product_name=None):
    if "product" in option.lower():
        if product_name:
            return generate_product_dashboard(product_name)
        return "Please specify a product."
    elif "state" in option.lower():
        return generate_state_dashboard()
    elif "sentiment" in option.lower():
        return generate_sentiment_dashboard()
    elif "report" in option.lower():
        return generate_report()
    else:
        return "I don't understand the option. Please select a valid one."

def generate_product_dashboard(product_name):
    filtered_data_product = complaint_data[complaint_data['Product'] == product_name]
    
    if filtered_data_product.empty:
        return "No data available for the selected product."

    product_issues = filtered_data_product['Issue'].value_counts()
    product_states = filtered_data_product['State'].value_counts()
    product_sentiment = filtered_data_product['Sentiment Category'].value_counts()
    
    # Issue Distribution
    fig_issue = px.bar(product_issues, x=product_issues.index, y=product_issues.values,
                       labels={'x':'Issue', 'y':'Number of Complaints'},
                       title=f'Issues for {product_name}')
    
    # State Distribution
    fig_state = px.bar(product_states, x=product_states.index, y=product_states.values,
                       labels={'x':'State', 'y':'Number of Complaints'},
                       title=f'Complaints by State for {product_name}')
    
    # Sentiment Distribution
    fig_sentiment = px.pie(product_sentiment, values=product_sentiment.values, names=product_sentiment.index,
                           title=f'Sentiment Distribution for {product_name}', hole=0.4)
    
    return fig_issue, fig_state, fig_sentiment

def generate_state_dashboard():
    state_counts = complaint_data['State'].value_counts()
    fig_state = px.bar(state_counts, x=state_counts.index, y=state_counts.values,
                       labels={'x':'State', 'y':'Number of Complaints'},
                       title='Complaints by State')
    return fig_state

def generate_sentiment_dashboard():
    sentiment_counts = complaint_data['Sentiment Category'].value_counts()
    fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                           title='Sentiment Distribution', hole=0.4)
    return fig_sentiment

def generate_report():
    report = complaint_data.groupby('Product').agg({
        'Sentiment': ['mean', 'std'],
        'Resolution time': ['mean', 'std']
    }).reset_index()
    report.to_csv('complaint_analysis_report.csv', index=False)
    return "Report generated successfully!"

# User interface for chatbot
selected_option = st.selectbox("Choose an option:", options)

if selected_option == "Show dashboard for a specific product":
    product_name = st.selectbox("Select a product:", complaint_data['Product'].unique())
else:
    product_name = None

if st.button('Submit'):
    st_message("You selected: " + selected_option, is_user=True)
    
    if "product" in selected_option.lower():
        response = generate_dashboard(selected_option, product_name)
        if isinstance(response, tuple):
            st.plotly_chart(response[0])  # Issue Distribution
            st.plotly_chart(response[1])  # State Distribution
            st.plotly_chart(response[2])  # Sentiment Distribution
        else:
            st_message(response, is_user=False)
    else:
        response = generate_dashboard(selected_option)
        if isinstance(response, str) and response.startswith('fig_'):
            st.plotly_chart(eval(response))
        elif response:
            st_message(response, is_user=False)

# Footer
st.sidebar.header("About")
st.sidebar.info("This is a Streamlit app for analyzing consumer complaints, performing sentiment analysis, and predicting complaint resolution times. It also includes a basic chatbot for assistance.")
