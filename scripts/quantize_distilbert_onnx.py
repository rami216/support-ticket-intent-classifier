from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


ONNX_DIR = Path("artifacts/distilbert/onnx")

INPUT_MODEL = ONNX_DIR / "model.onnx"
OUTPUT_MODEL = ONNX_DIR / "model_quantized.onnx"


def main():
    quantize_dynamic(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QInt8,
    )

    print("Quantization completed.")
    print(f"Saved to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()