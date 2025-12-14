import pandas as pd
import os

# Define the file path for the master file
master_file_path = r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\Master\master_fourier.csv"

# Load the CSV file
df = pd.read_csv(master_file_path)

# Extract the columns relevant to experimental conditions
condition_columns = ['Plant_Type', 'Treatment', 'Temp_Hum_Control', 'Light_Cycle', 'Replicate_Number', 'Run_Date']

# Get unique combinations of the experimental conditions
unique_conditions = df[condition_columns].drop_duplicates().reset_index(drop=True)

# Define the output file path in the same folder as the master file
output_dir = os.path.dirname(master_file_path)
output_file_path = os.path.join(output_dir, 'unique_experimental_conditions.csv')

# Save the table to a CSV file
unique_conditions.to_csv(output_file_path, index=False)

# Print confirmation
print(f"Unique experimental conditions saved to: {output_file_path}")

# Optionally, print the tabulated experimental conditions in markdown format for readability
print("\nUnique Experimental Conditions:")
print(unique_conditions.to_markdown(index=True))
