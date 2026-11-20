"""Command-line entry point. Defaults are relative to the project root."""
import argparse
import sys

from src.model import DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_OUTPUT, PAYLOAD_SIZES, ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact grayscale PCA and neural face encoding")
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
    DEFAULT_NEURAL = ROOT / "models" / "neural64" / "best.pt"
    DEFAULT_DECODER = ROOT / "models" / "neural64" / "decoder.pt"
    neural_train = commands.add_parser("neural-train", help="Train quantization-aware convolutional autoencoder")
    neural_train.add_argument("--dataset", default=DEFAULT_DATASET)
    neural_train.add_argument("--output", default=ROOT / "models" / "neural64")
    neural_train.add_argument("--samples", type=int, default=20000)
    neural_train.add_argument("--validation-samples", type=int, default=2000)
    neural_train.add_argument("--latent", type=int, choices=PAYLOAD_SIZES, default=64)
    neural_train.add_argument("--epochs", type=int, default=50)
    neural_train.add_argument("--batch-size", type=int, default=128)
    neural_train.add_argument("--learning-rate", type=float, default=.001)
    neural_train.add_argument("--max-grad-norm", type=float, default=1.,
                              help="Clip global gradient norm after AMP unscale (default: 1.0; 0 disables)")
    neural_train.add_argument("--l1-weight", type=float, default=1.)
    neural_train.add_argument("--ssim-weight", type=float, default=.2)
    neural_train.add_argument("--patience", type=int, default=10)
    neural_train.add_argument("--seed", type=int, default=42)
    neural_train.add_argument("--workers", type=int, default=None)
    neural_train.add_argument("--amp", action="store_true")
    neural_train.add_argument("--resume")
    neural_train.add_argument("--partition", help="Official list_eval_partition.txt, including outside image directory")
    neural_train.add_argument("--split-manifest", help="Reuse an existing saved split.json")
    neural_train.add_argument("--pca-model", default=DEFAULT_MODEL)
    neural_train.add_argument("--evaluation-manifest", default=DEFAULT_OUTPUT / "evaluation" / "verification.json")
    neural_encode = commands.add_parser("neural-encode", help="Encode exactly one byte per neural latent value")
    neural_encode.add_argument("image_path")
    neural_encode.add_argument("--model", dest="model_path", default=DEFAULT_NEURAL)
    neural_encode.add_argument("--output", default=DEFAULT_OUTPUT / "neural_face.bin")
    neural_decode = commands.add_parser("neural-decode", help="Decode using only bytes and shared decoder weights")
    neural_decode.add_argument("payload_path")
    neural_decode.add_argument("--model", dest="model_path", default=DEFAULT_DECODER)
    neural_decode.add_argument("--output", default=DEFAULT_OUTPUT / "neural_face.png")
    neural_export = commands.add_parser("neural-export-decoder", help="Export decoder weights without encoder/optimizer")
    neural_export.add_argument("--model", dest="model_path", default=DEFAULT_NEURAL)
    neural_export.add_argument("--output", default=DEFAULT_DECODER)
    neural_compare = commands.add_parser("neural-compare", help="Compare independently trained neural budgets with frozen PCA")
    neural_compare.add_argument("--dataset", default=DEFAULT_DATASET)
    neural_compare.add_argument("--models", nargs="+", default=[DEFAULT_NEURAL])
    neural_compare.add_argument("--pca-model", default=DEFAULT_MODEL)
    neural_compare.add_argument("--output", default=DEFAULT_OUTPUT / "neural_evaluation")
    neural_compare.add_argument("--samples", type=int, default=20)
    neural_compare.add_argument("--seed", type=int, default=43)
    neural_compare.add_argument("--evaluation-manifest", default=DEFAULT_OUTPUT / "evaluation" / "verification.json")
    for command in (neural_train, neural_encode, neural_decode, neural_export, neural_compare):
        command.add_argument("--device", default="auto" if command is neural_train else "cpu")
    continuation = commands.add_parser("neural-continue", help="Run the controlled two-arm Phase 4 continuation")
    continuation.add_argument("--dataset", default=DEFAULT_DATASET)
    continuation.add_argument("--source", default=ROOT / "models/neural64_retry/last.pt")
    continuation.add_argument("--reference", default=DEFAULT_NEURAL.parent.with_name("neural64_retry") / "best.pt")
    continuation.add_argument("--device", default="cuda")
    continuation.add_argument("--workers", type=int, default=4)
    continuation.add_argument("--additional-epochs", type=int, default=50)
    continuation.add_argument("--patience", type=int, default=10)
    continuation.add_argument("--resume", action="store_true")
    vq_train = commands.add_parser("vq-train", help="Train the separate 64-byte spatial-codebook codec")
    vq_train.add_argument("--dataset", default=DEFAULT_DATASET)
    vq_train.add_argument("--output", default=ROOT / "models/vq64")
    vq_train.add_argument("--samples", type=int, default=50000)
    vq_train.add_argument("--validation-samples", type=int, default=2000)
    vq_train.add_argument("--epochs", type=int, default=100)
    vq_train.add_argument("--batch-size", type=int, default=128)
    vq_train.add_argument("--learning-rate", type=float, default=.0002)
    vq_train.add_argument("--patience", type=int, default=15)
    vq_train.add_argument("--seed", type=int, default=42)
    vq_train.add_argument("--workers", type=int)
    vq_train.add_argument("--amp", action="store_true")
    vq_train.add_argument("--resume")
    vq_train.add_argument("--split-manifest")
    vq_train.add_argument("--base-split", default=ROOT / "models/neural64_retry/split.json")
    vq_train.add_argument("--new-split", dest="base_split", action="store_const", const=None,
                          help="Explicitly create a new deterministic split instead of extending previous training")
    for flag, default in (("l1-weight", 1.), ("ssim-weight", .2), ("edge-weight", .2),
                          ("codebook-weight", 1.), ("commitment-weight", .25), ("max-grad-norm", 1.)):
        vq_train.add_argument("--" + flag, type=float, default=default)
    vq_train.add_argument("--collapse-grace", type=int, default=5)
    vq_train.add_argument("--collapse-patience", type=int, default=3)
    vq_train.add_argument("--collapse-min-active", type=int, default=8)
    vq_train.add_argument("--collapse-min-perplexity", type=float, default=4.)
    vq_encode = commands.add_parser("vq-encode", help="Encode a face into 64 row-major codebook indices")
    vq_encode.add_argument("image_path")
    vq_encode.add_argument("--model", dest="model_path", default=ROOT / "models/vq64/best.pt")
    vq_encode.add_argument("--output", default=DEFAULT_OUTPUT / "vq_face.bin")
    vq_decode = commands.add_parser("vq-decode", help="Reconstruct using only the VQ decoder/codebook checkpoint")
    vq_decode.add_argument("payload_path")
    vq_decode.add_argument("--model", dest="model_path", default=ROOT / "models/vq64/decoder.pt")
    vq_decode.add_argument("--output", default=DEFAULT_OUTPUT / "vq_face.png")
    vq_export = commands.add_parser("vq-export-decoder", help="Export codebook and decoder without encoder or optimizer")
    vq_export.add_argument("--model", dest="model_path", default=ROOT / "models/vq64/best.pt")
    vq_export.add_argument("--output", default=ROOT / "models/vq64/decoder.pt")
    vq_compare = commands.add_parser("vq-compare", help="Compare VQ, PCA and neural A on identical varied held-out faces")
    vq_compare.add_argument("--dataset", default=DEFAULT_DATASET)
    vq_compare.add_argument("--model", dest="model_path", default=ROOT / "models/vq64/best.pt")
    vq_compare.add_argument("--neural-model", default=ROOT / "models/neural64_continue_A/best.pt")
    vq_compare.add_argument("--pca-model", default=DEFAULT_MODEL)
    vq_compare.add_argument("--output", default=DEFAULT_OUTPUT / "vq_evaluation")
    vq_compare.add_argument("--samples", type=int, default=25)
    vq_compare.add_argument("--seed", type=int, default=43)
    vq_compare.add_argument("--attributes", help="Official CelebA list_attr_celeba.txt for varied semantic examples")
    vq_compare.add_argument("--selection", help="Curated JSON list of common held-out relative filenames")
    for command in (vq_train, vq_encode, vq_decode, vq_compare):
        command.add_argument("--device", default="auto" if command is vq_train else "cpu")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    command = options.pop("command")
    if "model" in options:
        options["model_path"] = options.pop("model")
    try:
        if command == "vq-train":
            from src.vq_train import train_vq
            train_vq(**options)
        elif command == "vq-encode":
            from src.vq_codec import encode_image
            payload = encode_image(**options)
            print(f"Saved exactly {len(payload)} VQ bytes to {args.output}")
        elif command == "vq-decode":
            from src.vq_codec import decode_file
            decode_file(**options)
            print(f"Saved VQ reconstruction to {args.output}")
        elif command == "vq-export-decoder":
            from src.vq_codec import export_decoder
            export_decoder(**options)
        elif command == "vq-compare":
            from src.vq_evaluate import compare_vq
            compare_vq(**options)
        elif command == "neural-continue":
            from src.neural_continuation import run_experiment
            run_experiment(**options)
        elif command == "neural-train":
            from src.neural_train import train_neural
            train_neural(**options)
        elif command == "neural-encode":
            from src.neural_codec import encode_image
            payload = encode_image(**options)
            print(f"Saved exactly {len(payload)} bytes to {args.output}")
        elif command == "neural-decode":
            from src.neural_codec import decode_file
            decode_file(**options)
            print(f"Saved reconstructed image to {args.output}")
        elif command == "neural-export-decoder":
            from src.neural import export_decoder
            export_decoder(**options)
        elif command == "neural-compare":
            from src.neural_compare import compare_neural
            compare_neural(**options)
        elif command == "train":
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
