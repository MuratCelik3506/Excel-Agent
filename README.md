# 📊 Excel Agent

**Excel Agent** is an intelligent data analysis tool that empowers you to interact with your Excel and CSV files using natural language. Built with Streamlit and powered by Large Language Models (LLMs), it acts as your personal data analyst, generating code, executing it, and explaining the results in plain English.

## 🚀 Features

-   **Natural Language Queries**: Ask questions like "What is the average sales per region?" or "Plot the trend of revenue over time".
-   **Multi-Format Support**: Upload and analyze both `.xlsx` and `.csv` files.
-   **Transparent Execution**: View the **Plan**, **Generated Code**, and **Execution Result** to understand exactly how the agent arrived at the answer.
-   **Flexible LLM Support**:
    -   **OpenAI**: Use GPT-4o or other OpenAI models.
    -   **Google Gemini**: Use Gemini 1.5 Flash or Pro.
    -   **Local LLMs**: Connect to local models (e.g., Llama 3 via LM Studio) using an OpenAI-compatible endpoint.
-   **Interactive Visualizations**: Automatically generates and displays matplotlib/seaborn plots based on your queries.
-   **Robust CSV Handling**: Automatically detects and handles various file encodings (UTF-8, Latin-1, etc.).

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/excel-agent.git
    cd excel-agent
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

2.  **Configure the Agent**:
    -   Open the app in your browser (usually `http://localhost:8501`).
    -   In the sidebar, select your **LLM Provider** (OpenAI, Gemini, or Local).
    -   Enter your **API Key** (or Base URL for local models).

3.  **Analyze Data**:
    -   Upload your Excel or CSV file.
    -   Type your question in the chat box and hit enter!

## 🧩 Architecture

The agent follows a 4-step process for every query:
1.  **Plan**: Generates a high-level plan using the LLM.
2.  **Code**: Writes Pandas/Python code to implement the plan.
3.  **Execute**: Runs the code in a secure local scope, capturing results and plots.
4.  **Explain**: Uses the LLM to interpret the result and provide a friendly response.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT](LICENSE)
