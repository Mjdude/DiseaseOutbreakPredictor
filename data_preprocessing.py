import pandas as pd

# Load dataset properly (skip the first row if it's a header)
df = pd.read_csv("data/ILINet.csv", delimiter=",", skiprows=1)

# Print first few rows to check structure
print("📌 Original Data Preview:\n", df.head())

# Ensure column names are stripped of whitespace
df.columns = df.columns.str.strip()

# Check required columns
required_columns = {"YEAR", "WEEK"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing required columns! Expected {required_columns}, found {set(df.columns)}")

# Convert YEAR and WEEK into a proper date format
df["date"] = pd.to_datetime(df["YEAR"].astype(str) + "-W" + df["WEEK"].astype(str) + "-1", format="%Y-W%W-%w", errors="coerce")

# Remove rows where date conversion failed
df.dropna(subset=["date"], inplace=True)

# Fill missing values with 0
df.fillna(0, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save cleaned dataset
df.to_csv("data/cleaned_ILINet.csv", index=False)

print("✅ Data Preprocessing Done! Cleaned file saved as 'cleaned_ILINet.csv'.")
