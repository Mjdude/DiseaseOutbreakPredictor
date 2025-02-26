import pandas as pd

# Load dataset properly
df = pd.read_csv("data/ILINet.csv", delimiter=",", skiprows=1)

# Print first few rows to check structure
print("Original Data:\n", df.head())

# Ensure column names are stripped of whitespace
df.columns = df.columns.str.strip()

# Create a new 'date' column using YEAR and WEEK
df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-W' + df['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')

# Fill missing values with 0
df.fillna(0, inplace=True)

# Save cleaned dataset
df.to_csv("data/cleaned_ILINet.csv", index=False)

print("✅ Data Preprocessing Done! Cleaned file saved.")

