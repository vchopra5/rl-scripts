from huggingface_hub import snapshot_download

# Download the pre-trained Qwen 0.8B model
snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B-Base",
    local_dir="./models/Qwen3.5-0.8B-Base"
)
