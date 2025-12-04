import pandas as pd
from agent import ExcelAgent
import unittest
from unittest.mock import MagicMock
import json

class TestExcelAgent(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'Region': ['East', 'West', 'East']})
        # Mock the API key and provider to avoid real calls
        self.agent = ExcelAgent(self.df, api_key="dummy", provider="openai")
        self.agent._call_llm = MagicMock()

    def test_execute_json_plan_calculation(self):
        # Test a simple calculation plan
        plan = [
            {
                "id": 1,
                "function": "calculate",
                "args": {"dataframe": "df", "column": "A", "func": "mean"},
                "output": "final_result"
            }
        ]
        
        exec_result = self.agent.execute_json_plan(plan)
        self.assertTrue(exec_result['success'])
        self.assertEqual(exec_result['result'], 2.0)

    def test_execute_json_plan_filter_and_calc(self):
        # Test filtering then calculating
        plan = [
            {
                "id": 1,
                "function": "filter",
                "args": {"dataframe": "df", "column": "Region", "operator": "==", "value": "East"},
                "output": "filtered_df"
            },
            {
                "id": 2,
                "function": "calculate",
                "args": {"dataframe": "filtered_df", "column": "A", "func": "sum"},
                "output": "final_result"
            }
        ]
        
        exec_result = self.agent.execute_json_plan(plan)
        self.assertTrue(exec_result['success'])
        # Rows 0 and 2 are East. A values are 1 and 3. Sum is 4.
        self.assertEqual(exec_result['result'], 4)

    def test_generate_json_plan_mock(self):
        # Mock LLM response with valid JSON
        mock_json = json.dumps([
            {"id": 1, "function": "calculate", "args": {"dataframe": "df", "column": "A", "func": "mean"}, "output": "final_result"}
        ])
        self.agent._call_llm.return_value = f"```json\n{mock_json}\n```"
        
        plan = self.agent.generate_json_plan("mean of A")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['function'], "calculate")

    def test_execute_json_plan_default_dataframe(self):
        # Test that dataframe defaults to context['df'] if missing
        plan = [
            {
                "id": 1,
                "function": "calculate",
                "args": {"column": "A", "func": "mean"}, # Missing dataframe arg
                "output": "final_result"
            }
        ]
        exec_result = self.agent.execute_json_plan(plan)
        self.assertTrue(exec_result['success'])
        self.assertEqual(exec_result['result'], 2.0)

    def test_execute_json_plan_groupby_collision(self):
        # Test grouping by a column and aggregating the same column (e.g. count)
        # This can cause "cannot insert A, already exists" error on reset_index
        plan = [
            {
                "id": 1,
                "function": "group_by",
                "args": {"dataframe": "df", "by": "A", "agg_col": "A", "agg_func": "count"},
                "output": "final_result"
            }
        ]
        exec_result = self.agent.execute_json_plan(plan)
        self.assertTrue(exec_result['success'])
        # Result should have columns 'A' and 'A_count' or similar, or just be a dataframe
        self.assertIsInstance(exec_result['result'], pd.DataFrame)
        self.assertIn('A', exec_result['result'].columns)

if __name__ == '__main__':
    unittest.main()
