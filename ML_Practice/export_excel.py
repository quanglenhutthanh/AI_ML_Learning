import pandas as pd
import numpy as np
import time

# Function to generate random data
def generate_data(num_rows, num_cols):
    # Generate random data for the DataFrame
    data = np.random.rand(num_rows, num_cols)
    columns = [f"Column_{i+1}" for i in range(num_cols)]
    return pd.DataFrame(data, columns=columns)

def main():
    num_rows = 500000
    num_cols = 60
    output_file = 'large_excel_file.xlsx'
    
    # Start time measurement
    start_time = time.time()
    
    # Generate data and create DataFrame
    df = generate_data(num_rows, num_cols)
    
    # Export to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    # End time measurement
    end_time = time.time()
    
    # Calculate and print execution time
    execution_time = end_time - start_time
    print(f"Time taken to generate and export Excel file: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
