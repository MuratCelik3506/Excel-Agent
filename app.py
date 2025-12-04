import streamlit as st
import pandas as pd
from agent import ExcelAgent
import os

st.set_page_config(page_title="Excel Agent", page_icon="📊", layout="wide")

st.title("📊 Excel Agent")
st.markdown("Upload an Excel file and ask questions about your data!")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini", "Local"])
    
    api_key = None
    base_url = None
    
    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_name = st.text_input("Model Name", value="gpt-4o")
    elif provider == "Gemini":
        api_key = st.text_input("Gemini API Key", type="password")
        model_name = st.text_input("Model Name", value="gemini-1.5-flash")
    elif provider == "Local":
        base_url = st.text_input("Base URL", value="http://localhost:1234/v1")
        model_name = st.text_input("Model Name", value="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF")
        api_key = "lm-studio" # Placeholder, often not checked by local servers

    st.divider()
    st.markdown("### About")
    st.markdown("This agent uses an LLM to generate Pandas code to analyze your data.")

# Main Interface
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file and (api_key or provider == "Local"):
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    uploaded_file.seek(0) # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("Could not decode CSV file. Please ensure it is UTF-8 or Latin-1 encoded.")
                st.stop()
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Loaded {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns.")
        
        # Show Data Preview
        with st.expander("Preview Data"):
            st.dataframe(df.head())

        # Initialize Agent
        agent = ExcelAgent(df, api_key=api_key, provider=provider, model_name=model_name, base_url=base_url)

        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "code" in message:
                    with st.expander("Show Code"):
                        st.code(message["code"], language="python")
                if "result" in message:
                    st.write("Result:")
                    st.write(message["result"])
                if "fig" in message and message["fig"] is not None:
                    st.pyplot(message["fig"])

        if prompt := st.chat_input("Ask something about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                try:
                    # 1. Plan
                    with st.status("Generating Plan...", expanded=False) as status:
                        plan = agent.generate_plan(prompt)
                        status.write(plan)
                        status.update(label="Plan Generated", state="complete")
                    
                    # 2. Code
                    with st.status("Generating Code...", expanded=False) as status:
                        code = agent.generate_code(prompt, plan)
                        status.code(code, language="python")
                        status.update(label="Code Generated", state="complete")
                    
                    # 3. Execute
                    with st.status("Executing Code...", expanded=False) as status:
                        exec_result = agent.execute_code(code)
                        if exec_result["success"]:
                            status.update(label="Execution Successful", state="complete")
                        else:
                            status.update(label="Execution Failed", state="error")
                            st.error(f"Error: {exec_result['error']}")
                            st.code(exec_result['traceback'])
                            st.stop()

                    # 4. Explain
                    if exec_result["success"]:
                        final_response = agent.explain_result(prompt, exec_result)
                        message_placeholder.markdown(final_response)
                        
                        # Display explicit result if available (dataframe or number)
                        if exec_result["result"] is not None:
                            st.write(exec_result["result"])
                        
                        # Display plot if available
                        if exec_result["fig"] is not None:
                            st.pyplot(exec_result["fig"])

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_response,
                            "code": code,
                            "result": exec_result["result"],
                            "fig": exec_result["fig"]
                        })
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred loading the file: {str(e)}")

elif uploaded_file and not (api_key or provider == "Local"):
    st.warning("Please enter your API Key in the sidebar to proceed.")
else:
    st.info("Please upload a file to get started.")
