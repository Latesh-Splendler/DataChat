# Required installations
# pip install pandasai  
# pip install pandasai[excel]
# pip install pandasai[connectors]

import os, csv, streamlit as st, pandas as pd, tiktoken, matplotlib
from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
from pandasai.connectors.yahoo_finance import YahooFinanceConnector
from pandasai.llm import OpenAI, GoogleGemini
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.responses.response_parser import ResponseParser
from google.cloud import storage

class OutputParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
    
    def parse(self, result):
        if result['type'] == "dataframe":
            st.dataframe(result['value'])
        elif result['type'] == 'plot':
            st.image(result["value"])
        else:
            st.write(result['value'])
        return

def setup():
    st.header("Chat with your small and large datasets!", anchor=False, divider="red")
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

def get_tasks():
    st.sidebar.header("Select a task", divider="rainbow")
    return st.sidebar.radio("Choose one:",
                            ("Load from local drive, <200MB",
                             "Load from local drive, 200MB+",
                             "Load from Google Storage",
                             "Yahoo Finance"))

def get_llm():
    st.sidebar.header("Select an LLM", divider='rainbow')
    return st.sidebar.radio("Choose a LLM:", ("OpenAI", "Google Gemini"))

def write_read(bucket_name, blob_name, projectid):
    storage_client = storage.Client(project=projectid)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    with blob.open('r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
    
    return pd.DataFrame([{header[i]: row[i] for i in range(len(header))} for row in data])

def calculate_cost(df):  
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  
    cost_per_token = 0.0005  
    
    total_tokens = sum(len(encoding.encode(' '.join(row.values.astype(str)))) for _, row in df.iterrows())
    st.write('Tokens:', total_tokens)  
    st.write('Cost:', total_tokens * cost_per_token / 1000)

def main():
    setup()
    task = get_tasks()
    
    if task.startswith("Load from local drive"):
        dataset = st.file_uploader("Upload your CSV or XLSX file", type=['csv', 'xlsx'])
        if not dataset: st.stop()
        
        try:
            file_extension = dataset.name.split(".")[-1].lower()
            df = pd.read_csv(dataset, low_memory=False) if file_extension == "csv" else pd.read_excel(dataset)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
        
        calculate_cost(df)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        col_desc = st.radio("Do you want to provide column descriptors?", ("Yes", "No"))
        addon = st.text_input("Enter your column description (e.g., 'col1': 'unique id')") if col_desc == "Yes" else "None"
        
        if addon:
            llm_choice = get_llm()
            llm = OpenAI(api_token=OPENAI_API_KEY) if llm_choice == "OpenAI" else GoogleGemini(api_key=GOOGLE_API_KEY)
            sdf = SmartDataframe(PandasConnector({"original_df": df}, field_descriptions=addon),
                                 {"enable_cache": False},
                                 config={"llm": llm, "conversational": False, "response_parser": OutputParser})
            prompt = st.text_input("Enter your question/prompt.")
            if not prompt: st.stop()
            
            st.write("Response")
            with get_openai_callback() as cb if llm_choice == "OpenAI" else None:
                response = sdf.chat(prompt)
                st.write(response)
                st.divider()
                st.write("🧞‍♂️ Under the hood, the code that was executed:")
                st.code(sdf.last_code_executed)
                if cb:
                    st.divider()
                    st.write("💰 Tokens used and your cost:")
                    st.write(cb)
    
    elif task == "Load from Google Storage":
        bucket_name = st.text_input("Provide your Google Cloud Storage bucket name")
        blob_name = st.text_input("Provide the blob/object name")
        if not bucket_name or not blob_name: st.stop()
        df_bq = write_read(bucket_name, blob_name, projectid)
        st.write("Data Preview:")
        st.dataframe(df_bq.head())
    
    elif task == "Yahoo Finance":
        stock_symbol = st.text_input("Enter a stock symbol (e.g., MSFT)")
        if not stock_symbol: st.stop()
        yahoo_df = SmartDataframe(YahooFinanceConnector(stock_symbol), config={"response_parser": OutputParser})
        prompt = st.text_input("Enter your prompt")
        if not prompt: st.stop()
        st.write("Response")
        response = yahoo_df.chat(prompt)
        st.divider()
        st.code(yahoo_df.last_code_executed)
    
if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    projectid = os.environ.get('GOOG_PROJECT')
    matplotlib.use("Agg", force=True)
    main()
