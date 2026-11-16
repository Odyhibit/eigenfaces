"""Command-line entry point. Defaults are relative to the project root."""
import argparse
import sys

from src.model import DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_OUTPUT, PAYLOAD_SIZES


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact grayscale PCA face encoding")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train PCA and shared quantization ranges")
    train.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    train.add_argument("--samples", type=int, default=5000)
    train.add_argument("--components", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)
    compare = commands.add_parser("compare", help="Compare held-out images at all five sizes")
    compare.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    compare.add_argument("--samples", type=int, default=10)
    compare.add_argument("--seed", type=int, default=43)
    compare.add_argument("--output", default=DEFAULT_OUTPUT)
    encode = commands.add_parser("encode", help="Encode a JPEG into exactly N bytes")
    encode.add_argument("image")
    encode.add_argument("--bytes", dest="size", type=int, choices=PAYLOAD_SIZES, default=64)
    encode.add_argument("--output", default=DEFAULT_OUTPUT / "face.bin")
    decode = commands.add_parser("decode", help="Reconstruct a PNG using only a binary payload and model")
    decode.add_argument("payload")
    decode.add_argument("--output", default=DEFAULT_OUTPUT / "reconstructed.png")
    for command in (train, compare, encode, decode):
        command.add_argument("--model", default=DEFAULT_MODEL, help="Shared trusted joblib model path")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    command = options.pop("command")
    options["model_path"] = options.pop("model")
    try:
        if command == "train":
            from src.train import train_model
            train_model(**options)
        elif command == "compare":
            from src.compare import compare_images
            compare_images(**options)
        elif command == "encode":
            from src.encode import encode_image
            options["image_path"] = options.pop("image")
            payload = encode_image(**options)
            print(f"Saved exactly {len(payload)} bytes to {args.output}")
        else:
            from src.decode import decode_file
            options["payload_path"] = options.pop("payload")
            decode_file(**options)
            print(f"Saved reconstructed image to {args.output}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
