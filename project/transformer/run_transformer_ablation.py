import os

CONFIGS = [
    ("absolute", "layernorm"),
    ("absolute", "rmsnorm"),
]

for pos, norm in CONFIGS:
    cmd = (
        f"python run_transformer_from_scratch.py "
        f"--pos_emb {pos} "
        f"--norm {norm} "
        f"--epochs 5"
    )
    print("=" * 80)
    print(cmd)
    os.system(cmd)
