"""Print candidate release hashes for operator review; never changes trust pins.

Run only in a trusted training/release environment, before artifact deployment.
"""
import hashlib
import json
from .model import MODEL_FILE, CARD_FILE

if __name__ == "__main__":
    print(json.dumps({
        "model_sha256": hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        "card_sha256": hashlib.sha256(CARD_FILE.read_bytes()).hexdigest(),
    }, indent=2))
