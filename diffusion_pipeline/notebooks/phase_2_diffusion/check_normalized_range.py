import numpy as np
from pathlib import Path

# Path to processed diffusion data
data_path = Path("/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/data/processed/diffusion/train_data.npz")

if not data_path.exists():
    print(f"File not found: {data_path}")
    # Try finding any .npz file in that directory
    parent = data_path.parent
    print(f"Listing {parent}:")
    for p in parent.glob("*.npz"):
        print(f" - {p.name}")
else:
    print(f"Loading {data_path}...")
    data = np.load(data_path)
    X = data['X']
    print(f"Shape: {X.shape}")
    print(f"Min: {X.min()}")
    print(f"Max: {X.max()}")
    print(f"Mean: {X.mean()}")
    print(f"Std: {X.std()}")

    # Check approximate distribution
    print("\nPercentiles:")
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        print(f"{p}%: {np.percentile(X, p)}")
