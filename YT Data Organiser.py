import pandas as pd
from tkinter import Tk, filedialog
import os

def generate_output_path(input_path, suffix="_processed"):
    # Get the folder and base file name
    folder = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(folder, f"{base_name}{suffix}.csv")
    
    # Add a version number if the file already exists
    version = 1
    while os.path.exists(output_path):
        output_path = os.path.join(folder, f"{base_name}{suffix}_v{version}.csv")
        version += 1

    return output_path

def process_csv(input_path):
    # Read the CSV file with the option to preserve "N/A" as is
    df = pd.read_csv(input_path, keep_default_na=False)
    
    # Check if "Channel Name" and "View Count" columns exist
    if "Channel Name" not in df.columns or "View Count" not in df.columns:
        print("The CSV file does not contain the necessary columns ('Channel Name' and 'View Count').")
        return

    # Step 1: Add "Number of Video Published" column
    channel_counts = df['Channel Name'].value_counts()
    df['Number of Video Published'] = df['Channel Name'].map(channel_counts)

    # Generate the primary output file path
    output_path = generate_output_path(input_path, "_processed")
    df.to_csv(output_path, index=False)
    print(f"Processed file saved as: {output_path}")

    # Step 2: Create the secondary file with unique channels
    # Group by 'Channel Name' and select the row with the highest 'View Count' for each channel
    unique_df = df.loc[df.groupby('Channel Name')['View Count'].idxmax()]

    # Generate the secondary output file path
    unique_output_path = generate_output_path(input_path, "_unique_channels")
    unique_df.to_csv(unique_output_path, index=False)
    print(f"Unique channel file saved as: {unique_output_path}")

def upload_and_process_file():
    # Open file dialog to select CSV file
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        process_csv(file_path)

# Initialize Tkinter GUI
root = Tk()
root.withdraw()  # Hide the main window
print("Please select the CSV file for processing...")
upload_and_process_file()
