

import huggingface_hub
from huggingface_hub import HfApi, login

HF_TOKEN = "********" 
login(token=HF_TOKEN)

# 2. Configuration
local_folder_path = "/Users/jahanvigajera/M25CSA012/model_epoch5" 
repo_id = "Jahanvi16/tinybert_model"      

# 3. Initialize the HfApi
api = HfApi()

try:
    # 3. Create the repo (exists_ok=True prevents the 409 Conflict error)
    print(f"Creating/Verifying repository: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model")

    # 4. Upload the folder
    print(f"🚀 Uploading files from {local_folder_path}...")
    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial model and tokenizer upload"
    )
    print(f"✅ Success! Your model is at: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"❌ Error: {e}")