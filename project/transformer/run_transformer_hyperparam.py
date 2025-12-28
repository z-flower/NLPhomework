import os

BATCHES = [16, 32]
LRS = [1e-4, 3e-4]

for bs in BATCHES:
    for lr in LRS:
        cmd = (
            f"python run_transformer_from_scratch.py "
            f"--batch_size {bs} "
            f"--lr {lr} "
            f"--epochs 5"
        )
        print("=" * 80)
        print(cmd)
        os.system(cmd)
