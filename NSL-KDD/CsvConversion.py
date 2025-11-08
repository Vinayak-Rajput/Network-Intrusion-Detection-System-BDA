import pandas as pd
from pathlib import Path  # Import the pathlib library

# --- Define Column Names ---
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]
column_names_full = column_names + ['difficulty_score']

# --- Main Script ---

# Get the directory where this Python script is located
script_dir = Path(__file__).parent

# Define file paths *relative to the script's directory*
input_file_path = script_dir / 'KDDTest+.TXT'
output_csv_path = script_dir / 'KDDTest_with_headers.csv'
output_excel_path = script_dir / 'KDDTest_presentation.xlsx'

# 2. Load the .TXT file using the full, absolute path
print(f"Loading '{input_file_path}'...")
df = pd.read_csv(input_file_path, header=None, names=column_names_full)

# 3. Save to a new CSV file (in the same script directory)
df.to_csv(output_csv_path, index=False)
print(f"Successfully saved to '{output_csv_path}'")

# 4. Save to an Excel file (in the same script directory)
df.head(500).to_excel(output_excel_path, index=False)
print(f"Successfully saved a sample to '{output_excel_path}'")

print("Done.")