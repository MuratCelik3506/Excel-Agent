import pandas as pd
import os
import io
import sys
import traceback

class ExcelAgent:
    def __init__(self, dataframe: pd.DataFrame, api_key: str = None, provider: str = 'openai', model_name: str = None, base_url: str = None):
        self.df = dataframe
        self.api_key = api_key or "dummy" # Local LLMs might not need a real key
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        
        # Set up the LLM client
        if self.provider == 'openai' or self.provider == 'local':
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.model = self.model_name or ("gpt-4o" if self.provider == 'openai' else "local-model")
        elif self.provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = self.model_name or "gemini-1.5-flash"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _call_llm(self, prompt: str) -> str:
        """Helper to call the configured LLM."""
        try:
            if self.provider == 'openai' or self.provider == 'local':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful data analysis assistant. You are given a pandas DataFrame named 'df'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                print(response)
                content = response.choices[0].message.content
                if content is None:
                    return "Error: LLM returned no content."
                return content
            elif self.provider == 'gemini':
                import google.generativeai as genai
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return response.text
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
        return "Error: Unknown provider or failed execution."

    def generate_plan(self, user_query: str) -> str:
        """Generates a natural language plan."""
        columns = ", ".join(self.df.columns.tolist())
        head = self.df.head(3).to_markdown()
        
        prompt = f"""
        User Query: {user_query}
        
        Dataframe Context:
        Columns: {columns}
        Sample Data:
        {head}
        
        Task: Explain step-by-step how you would solve this query using Pandas. 
        Keep it concise and high-level. Do not write code yet.
        """
        return self._call_llm(prompt)

    def generate_code(self, user_query: str, plan: str) -> str:
        """Generates the Pandas code."""
        columns = ", ".join(self.df.columns.tolist())
        dtypes = str(self.df.dtypes)
        
        prompt = f"""
        You have a pandas DataFrame named `df`.
        Columns: {columns}
        Data Types:
        {dtypes}
        
        User Query: {user_query}
        Plan: {plan}
        
        Task: Write Python code to solve the query.
        
        Constraints:
        1. The code must be valid Python.
        2. Assume `df` is already loaded.
        3. If the user asks for a plot, use matplotlib or seaborn and save the figure to a variable named `fig` if possible, or just `plt.show()`.
        4. If the result is a number or dataframe, assign it to a variable named `result`.
        5. RETURN ONLY THE CODE. No markdown backticks, no explanation.
        """
        
        response = self._call_llm(prompt)
        if not response or "Error:" in response:
            return f"print('{response}')"
            
        # Try to extract code from markdown blocks
        import re
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code_match = re.search(r"```(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                # Fallback: assume the whole response is code, but strip backticks if any
                code = response.replace("```python", "").replace("```", "").strip()
        
        return code

    def execute_code(self, code: str):
        """Executes the code in a local scope."""
        local_scope = {"df": self.df, "pd": pd}
        
        # Add plotting libraries to scope
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            local_scope["plt"] = plt
            local_scope["sns"] = sns
        except ImportError:
            pass

        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            print("burada")
            import ast
            tree = ast.parse(code)
            last_stmt = tree.body[-1] if tree.body else None
            
            # If the last statement is an expression, we want to return its value
            if isinstance(last_stmt, ast.Expr):
                # Execute everything before the last expression
                exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), filename="<string>", mode="exec"), {}, local_scope)
                # Evaluate the last expression
                result = eval(compile(ast.Expression(body=last_stmt.value), filename="<string>", mode="eval"), {}, local_scope)
                local_scope["result"] = result
            else:
                # Just execute the whole thing
                exec(code, {}, local_scope)
                result = local_scope.get("result", None)
                print(result)
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
            
            fig = local_scope.get("fig", None)
            
            # If no explicit result variable, but there is stdout, use that
            if result is None and output:
                result = output
            
            return {
                "success": True,
                "result": result,
                "output": output,
                "fig": fig
            }
        except Exception as e:
            sys.stdout = old_stdout
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def explain_result(self, user_query: str, execution_result: dict) -> str:
        """Explains the result to the user."""
        result_preview = str(execution_result.get("result"))[:500] # Truncate for token limit
        
        prompt = f"""
        User Query: {user_query}
        Result: {result_preview}
        
        Task: Provide a friendly, human-readable answer based on the result.
        """
        return self._call_llm(prompt)
