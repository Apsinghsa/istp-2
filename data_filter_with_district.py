import os
import pandas as pd

# Define input and output directories
input_dir = "data/raw"
output_dir = "data/filtered"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the raw folder
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        # Read the CSV file
        input_path = os.path.join(input_dir, filename)
        df = pd.read_csv(input_path)
        
        # Keep only records where District is 'MANDI' (case-insensitive, trimmed)
        df_filtered = df[df['District'].fillna('').str.strip().str.upper() == 'MANDI']
        
        # Save the filtered CSV to the output folder
        output_path = os.path.join(output_dir, filename)
        df_filtered.to_csv(output_path, index=False)
        
        print(f"Processed: {filename}")

print("Filtering complete!")