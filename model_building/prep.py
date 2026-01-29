# for data manipulation
import pandas as pd
import sklearn
import dotenv
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from huggingface_hub import hf_hub_download


# Define the repository ID and the filename
REPO_ID = "arulmozhiselvan/tourism-gl-arul" 
FILENAME = "data/tourism.csv" 
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# Download the file to a local cache and get the local file path
csv_file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)


# Read dataset from Hugging Face
print("Tourism dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------------------------
# Encode categorical columns
# ---------------------------
label_encoder = LabelEncoder()

categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    else:
        print(f"Warning: Column '{col}' not found in dataset")

# ---------------------------
# Target column
# ---------------------------
target_col = "ProdTaken"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found")

# ---------------------------
# Train-test split
# ---------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y

# ---------------------------
out_dir = "tourism_project/data"
os.makedirs(out_dir, exist_ok=True)

Xtrain_path = f"{out_dir}/Xtrain.csv"
Xtest_path = f"{out_dir}/Xtest.csv"
ytrain_path = f"{out_dir}/ytrain.csv"
ytest_path = f"{out_dir}/ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("Train-test split complete. Files saved locally:")
print(Xtrain_path, Xtest_path, ytrain_path, ytest_path)

# ---------------------------
# Upload files to Hugging Face dataset repo
# ---------------------------
files = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),   # uploads just filename
        repo_id="arulmozhiselvan/tourism-gl-arul",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to HF dataset repo.")
