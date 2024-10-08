import os
import subprocess
import sys

from torchbenchmark.util.framework.huggingface.patch_hf import (
    cache_model,
    patch_transformers,
)


def pip_install_requirements():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
    )


if __name__ == "__main__":
    pip_install_requirements()
    patch_transformers()
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    cache_model(model_name)
