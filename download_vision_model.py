"""Download the pinned, small ADE20K SegFormer ONNX checkpoint locally."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


REVISION = "312a6466f90f7057b711cf550d5659ed3a18d508"
BASE_URL = ("https://huggingface.co/optimum/segformer-b0-finetuned-ade-512-512/"
            f"resolve/{REVISION}/")
EXPECTED = {
    "model.onnx": "3a89102115fe3c16230502437b894844ba50cde6f7c800f9884e87c360bcbfc9",
    "config.json": "8faac3a12302746f63d8a3a8c0bbf7f7f49bf459e5a3533bf71c067cb824e799",
    "preprocessor_config.json": "b5fa2a5da2f4483d6f2f8ac96c2a81ff91a91a0815cfbc77cc2ab30fd7c7b45b",
}


def download_model(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in EXPECTED.items():
        target = destination / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash:
            continue
        with urlopen(BASE_URL + name, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Pinned model checksum mismatch for {name}; file was not saved.")
        target.write_bytes(data)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "models" / "segformer-b0-ade20k")
    args = parser.parse_args()
    print(f"Verified model in {download_model(args.output)}")


if __name__ == "__main__":
    main()
