import os

BASE = "pbma"

structure = {
    "configs": ["mnist_vae.yaml", "mnist_realnvp.yaml", "cifar10_realnvp.yaml"],
    "datasets": [],
    "models": ["__init__.py", "vae.py", "realnvp.py", "layers.py"],
    "training": ["__init__.py", "trainer.py", "checkpoint_manager.py", "ema.py"],
    "pbma": ["__init__.py", "checkpoint_loader.py", "weighting.py", "aggregator.py", "metrics.py"],
    "scripts": ["train.py", "validate_checkpoints.py", "run_pbma_eval.py", "sample_pbma.py"],
    "utils": ["__init__.py", "seed.py", "logger.py", "plotting.py"],
}

ROOT_FILES = ["README.md", "requirements.txt", ".gitignore", "setup.py"]

def create_structure():
    os.makedirs(BASE, exist_ok=True)
    print(f"Created root: {BASE}/")

    for folder, files in structure.items():
        folder_path = os.path.join(BASE, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"  Created: {folder_path}/")
        for f in files:
            fp = os.path.join(folder_path, f)
            with open(fp, "w") as fh:
                if f.endswith(".py"):
                    fh.write(f"# {f}\n")
                elif f.endswith(".yaml"):
                    fh.write(f"# Config: {f}\n")
                else:
                    fh.write("")
            print(f"    Created: {fp}")

    for f in ROOT_FILES:
        fp = os.path.join(BASE, f)
        with open(fp, "w") as fh:
            if f == "README.md":
                fh.write("# Pathwise Bayesian Model Averaging (PBMA)\n")
            elif f == ".gitignore":
                fh.write("__pycache__/\n*.pyc\n*.pt\n*.pth\n.env\nPBMA-ENV/\n")
            elif f == "requirements.txt":
                fh.write("torch\ntorchvision\nnumpy\nmatplotlib\npyyaml\ntqdm\n")
            else:
                fh.write("")
        print(f"  Created: {fp}")

    print("\nDone! Directory structure created successfully.")

if __name__ == "__main__":
    create_structure()
