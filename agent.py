import pandas as pd
import os
import io
import sys
import traceback
import json
import re

class ExcelAgent:
    def __init__(self, dataframe: pd.DataFrame, api_key: str = None, provider: str = 'openai', model_name: str = None, base_url: str = None):
        self.df = dataframe
        self.api_key = api_key or "dummy"
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
                        {"role": "system", "content": "You are a helpful data analysis assistant. You output ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
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

    def generate_json_plan(self, user_query: str) -> list:
        """Generates a JSON plan for the query."""
        columns = ", ".join(self.df.columns.tolist())
        head = self.df.head(3).to_markdown()
        dtypes = str(self.df.dtypes)
        
        prompt = f"""
        You are a data analyst. You have a pandas DataFrame named `df`.
        Columns: {columns}
        Data Types:
        {dtypes}
        Sample Data:
        {head}
        
        User Query: {user_query}
        
        Task: Create a JSON plan to answer the query using the following supported functions:
        
        1. `filter(dataframe, column, operator, value)`: Filter rows. Operators: ==, !=, >, <, >=, <=, contains.
        2. `sort(dataframe, column, ascending)`: Sort rows. ascending is boolean.
        3. `group_by(dataframe, by, agg_col, agg_func)`: Group by column 'by', and aggregate 'agg_col' using 'agg_func' (mean, sum, count, min, max).
        4. `calculate(dataframe, column, func)`: Calculate on a column. func: mean, sum, min, max, count, unique, value_counts.
        5. `select_columns(dataframe, columns)`: Select specific columns. 'columns' is a list of strings.
        6. `head(dataframe, n)`: Get first n rows.
        7. `plot(dataframe, x, y, kind, title)`: Plot data. kind: line, bar, scatter, hist, box.
        
        Output Format:
        A JSON list of steps. Each step is an object:
        {{
            "id": 1,
            "function": "function_name",
            "args": {{ "arg_name": "arg_value", ... }},
            "output": "variable_name_to_store_result"
        }}
        
        Rules:
        - The first step usually takes "df" as input in 'dataframe' arg.
        - Subsequent steps can use the 'output' variable name of previous steps as input.
        - The final result should be in a variable named "final_result".
        - Do NOT write any text or markdown outside the JSON.
        """
        
        response = self._call_llm(prompt)
        
        # Extract JSON from response
        try:
            # Try to find JSON block
            json_match = re.search(r"```json\n(.*?)```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"```(.*?)```", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response.strip()
            
            parsed = json.loads(json_str)
            
            # Handle case where LLM wraps list in a dict
            if isinstance(parsed, dict):
                # Check if it's a single step
                if "function" in parsed:
                    parsed = [parsed]
                elif "steps" in parsed:
                    parsed = parsed["steps"]
                elif "plan" in parsed:
                    parsed = parsed["plan"]
                else:
                    # Try to find any list value
                    for v in parsed.values():
                        if isinstance(v, list):
                            parsed = v
                            break
            
            if not isinstance(parsed, list):
                print(f"Parsed JSON is not a list: {parsed}")
                return []
                
            return parsed
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {response}")
            return []

    def execute_json_plan(self, plan: list):
        """Executes the JSON plan."""
        context = {"df": self.df}
        execution_log = []
        final_result = None
        result = None
        fig = None
        
        if not isinstance(plan, list):
             return {
                "success": False,
                "error": f"Invalid plan format: expected list, got {type(plan)}",
                "log": []
            }
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            pass

        for step in plan:
            if not isinstance(step, dict):
                execution_log.append({
                    "status": "error",
                    "error": f"Invalid step format: expected dict, got {type(step)}"
                })
                continue
                
            func_name = step.get("function")
            args = step.get("args", {})
            output_var = step.get("output")
            
            # Resolve variable references in args
            resolved_args = {}
            for k, v in args.items():
                if isinstance(v, str) and v in context:
                    resolved_args[k] = context[v]
                else:
                    resolved_args[k] = v
            
            try:
                result = None
                
                
                if func_name == "filter":
                    df_in = resolved_args.get("dataframe", context["df"])
                    col = resolved_args.get("column")
                    op = resolved_args.get("operator")
                    val = resolved_args.get("value")
                    
                    if op == "==": result = df_in[df_in[col] == val]
                    elif op == "!=": result = df_in[df_in[col] != val]
                    elif op == ">": result = df_in[df_in[col] > val]
                    elif op == "<": result = df_in[df_in[col] < val]
                    elif op == ">=": result = df_in[df_in[col] >= val]
                    elif op == "<=": result = df_in[df_in[col] <= val]
                    elif op == "contains": result = df_in[df_in[col].astype(str).str.contains(str(val), case=False, na=False)]
                
                elif func_name == "sort":
                    df_in = resolved_args.get("dataframe", context["df"])
                    col = resolved_args.get("column")
                    asc = resolved_args.get("ascending", True)
                    result = df_in.sort_values(by=col, ascending=asc)
                
                elif func_name == "group_by":
                    df_in = resolved_args.get("dataframe", context["df"])
                    by = resolved_args.get("by")
                    agg_col = resolved_args.get("agg_col")
                    agg_func = resolved_args.get("agg_func")
                    
                    # Perform groupby
                    grouped = df_in.groupby(by)[agg_col].agg(agg_func)
                    
                    # If result is a Series (single agg column), rename it to avoid collision on reset_index
                    if isinstance(grouped, pd.Series):
                        if grouped.name == by:
                            grouped.name = f"{agg_col}_{agg_func}"
                    
                    result = grouped.reset_index()
                
                elif func_name == "calculate":
                    df_in = resolved_args.get("dataframe", context["df"])
                    col = resolved_args.get("column")
                    func = resolved_args.get("func")
                    series = df_in[col]
                    if func == "mean": result = series.mean()
                    elif func == "sum": result = series.sum()
                    elif func == "min": result = series.min()
                    elif func == "max": result = series.max()
                    elif func == "count": result = series.count()
                    elif func == "unique": result = series.unique()
                    elif func == "value_counts": result = series.value_counts()
                
                elif func_name == "select_columns":
                    df_in = resolved_args.get("dataframe", context["df"])
                    cols = resolved_args.get("columns")
                    result = df_in[cols]
                
                elif func_name == "head":
                    df_in = resolved_args.get("dataframe", context["df"])
                    n = int(resolved_args.get("n", 5))
                    result = df_in.head(n)
                
                elif func_name == "plot":
                    df_in = resolved_args.get("dataframe", context["df"])
                    x = resolved_args.get("x")
                    y = resolved_args.get("y")
                    kind = resolved_args.get("kind", "bar")
                    title = resolved_args.get("title", f"{kind} plot of {y} by {x}")
                    
                    plt.figure(figsize=(10, 6))
                    if kind == "bar":
                        sns.barplot(data=df_in, x=x, y=y)
                    elif kind == "line":
                        sns.lineplot(data=df_in, x=x, y=y)
                    elif kind == "scatter":
                        sns.scatterplot(data=df_in, x=x, y=y)
                    elif kind == "hist":
                        sns.histplot(data=df_in, x=x)
                    elif kind == "box":
                        sns.boxplot(data=df_in, x=x, y=y)
                    
                    plt.title(title)
                    plt.xticks(rotation=45)
                    fig = plt.gcf()
                    result = "Plot generated"

                else:
                    raise ValueError(f"Unknown function: {func_name}")

                # Store result
                if output_var:
                    context[output_var] = result
                    if output_var == "final_result":
                        final_result = result
                
                execution_log.append({
                    "step_id": step.get("id"),
                    "status": "success",
                    "result_preview": str(result)[:100]
                })

            except Exception as e:
                execution_log.append({
                    "step_id": step.get("id"),
                    "status": "error",
                    "error": str(e)
                })
                return {
                    "success": False,
                    "error": str(e),
                    "log": execution_log
                }

        return {
            "success": True,
            "result": final_result if final_result is not None else result, # Fallback to last result
            "log": execution_log,
            "fig": fig
        }

    def explain_result(self, user_query: str, execution_result: dict) -> str:
        """Explains the result to the user."""
        result_preview = str(execution_result.get("result"))[:500]
        
        prompt = f"""
        User Query: {user_query}
        Result: {result_preview}
        
        Task: Provide a friendly, human-readable answer based on the result.
        """
        return self._call_llm(prompt)
