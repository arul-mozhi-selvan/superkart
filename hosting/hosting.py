from huggingface_hub import HfApi
import os

# Authenticate using your HF_TOKEN (must be added in GitHub Actions secrets OR env vars)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the entire Streamlit deployment folder to your Hugging Face Space
api.upload_folder(
    folder_path="tourism-gl/deployment",         # local folder containing app.py + requirements.txt
    repo_id="arulmozhiselvan/tourism-gl",  # 🚨 your HF Space name
    repo_type="space",                            # this is a Streamlit Space
    path_in_repo=""                               # upload to root of the Space
)

print("Deployment files uploaded successfully to Hugging Face Space!")
