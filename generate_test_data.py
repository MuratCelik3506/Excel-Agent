import pandas as pd
import numpy as np

# Create Sales Data (CSV)
sales_data = {
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C'], 100),
    'Sales': np.random.randint(100, 1000, 100),
    'Quantity': np.random.randint(1, 20, 100)
}
df_sales = pd.DataFrame(sales_data)
df_sales.to_csv('test_sales.csv', index=False)
print("Created test_sales.csv")

# Create Inventory Data (Excel)
inventory_data = {
    'ProductID': range(1, 51),
    'ProductName': [f'Item {i}' for i in range(1, 51)],
    'Category': np.random.choice(['Electronics', 'Furniture', 'Office'], 50),
    'Stock': np.random.randint(0, 500, 50),
    'Price': np.random.uniform(10.0, 500.0, 50).round(2)
}
df_inventory = pd.DataFrame(inventory_data)
df_inventory.to_excel('test_inventory.xlsx', index=False)
print("Created test_inventory.xlsx")
