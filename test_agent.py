import pandas as pd
from agent import ExcelAgent
import unittest
from unittest.mock import MagicMock

class TestExcelAgent(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        # Mock the API key and provider to avoid real calls
        self.agent = ExcelAgent(self.df, api_key="dummy", provider="openai")
        self.agent._call_llm = MagicMock()

    def test_generate_code_execution(self):
        # Mock LLM response to return valid code
        self.agent._call_llm.return_value = "result = df['A'].mean()"
        
        # Test code generation (mocked)
        code = self.agent.generate_code("mean of A", "plan")
        self.assertEqual(code, "result = df['A'].mean()")
        
        # Test execution
        exec_result = self.agent.execute_code(code)
        self.assertTrue(exec_result['success'])
        self.assertEqual(exec_result['result'], 2.0)

    def test_plotting_execution(self):
        # Mock LLM response for plotting
        code = "import matplotlib.pyplot as plt\nfig = plt.figure()\nplt.plot(df['A'], df['B'])"
        
        exec_result = self.agent.execute_code(code)
        self.assertTrue(exec_result['success'])
        self.assertIsNotNone(exec_result['fig'])

    def test_implicit_result_capture(self):
        # Test that an expression without assignment is captured
        code = "df['A'].mean()"
        exec_result = self.agent.execute_code(code)
        self.assertTrue(exec_result['success'])
        self.assertEqual(exec_result['result'], 2.0)

if __name__ == '__main__':
    unittest.main()
