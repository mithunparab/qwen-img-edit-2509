from huggingface_hub import snapshot_download

print('Checking if base model exists...')
try:
    snapshot_download(
        'Qwen/Qwen-Image-Edit-2509',
        cache_dir='/home/user/app/models',
        local_files_only=True
    )
    print('✓ Base model found in image!')
except:
    print('✗ Base model not found, downloading...')
    snapshot_download(
        'Qwen/Qwen-Image-Edit-2509',
        cache_dir='/home/user/app/models'
    )

print('Downloading LoRA...')
snapshot_download(
    'lightx2v/Qwen-Image-Lightning',
    cache_dir='/home/user/app/models',
    allow_patterns=['Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors', '*.json', 'README.md']
)
print('✓ All downloads complete!')