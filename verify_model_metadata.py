"""Verify metadata on a TFLite model file.

Usage:
  python verify_model_metadata.py
  python verify_model_metadata.py assets/your_model.tflite
"""

import sys
from pathlib import Path


DEFAULT_MODEL_PATH = Path("assets/best_float32.tflite")


def verify_metadata(model_path: Path) -> None:
    try:
        from tflite_support.metadata import MetadataDisplayer
    except ImportError:
        print(
            "tflite_support is not installed. Install it first:\n"
            "pip install tflite-support"
        )
        return

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    try:
        displayer = MetadataDisplayer.with_model_file(str(model_path))
        print(f"Metadata for: {model_path}")
        print(displayer.get_metadata_json())
        print("\nAssociated Files:")
        print(displayer.get_packed_associated_file_list())
    except Exception as error:
        print(f"Unable to read metadata from {model_path}: {error}")


if __name__ == "__main__":
    selected_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    verify_metadata(selected_path)