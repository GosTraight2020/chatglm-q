from huggingface_hub import snapshot_download

snapshot_download(repo_id="K024/chatglm2-6b", cache_dir='/home/shanlin/pytorch-vulkan/weight_chatglm_fp32')