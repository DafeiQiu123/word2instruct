import argparse
import json

from transformers import T5ForConditionalGeneration, T5Tokenizer


def pretty_print_json(text: str):
    try:
        obj = json.loads(text)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    input_text = f"Instruction: {args.text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=4,
        early_stopping=True,
    )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Input:")
    print(args.text)
    print("\nPredicted JSON:")
    pretty_print_json(prediction)


if __name__ == "__main__":
    main()