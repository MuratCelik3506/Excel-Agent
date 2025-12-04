import openai
import sys

# Configuration matching the user's curl command and app defaults
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"

print(f"Testing connection to {BASE_URL} with model {MODEL}...")

client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful data analysis assistant. You are given a pandas DataFrame named 'df'."},
            {"role": "user", "content": "Calculate the mean of column A."}
        ],
        temperature=0,
        stream=False 
    )
    
    print("\n--- Response Object ---")
    print(response)
    
    print("\n--- Content ---")
    content = response.choices[0].message.content
    print(f"Content: {content!r}")
    
    if content:
        print("\nSUCCESS: Received content from Local LLM.")
    else:
        print("\nWARNING: Received empty content.")

except openai.APIConnectionError as e:
    print(f"\nERROR: Could not connect to server: {e}")
    print("Check if LM Studio is running and the server is started on port 1234.")
except openai.APIStatusError as e:
    print(f"\nERROR: Server returned status code {e.status_code}")
    print(f"Response: {e.response}")
except Exception as e:
    print(f"\nERROR: An unexpected error occurred: {e}")
