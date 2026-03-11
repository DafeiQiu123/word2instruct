import argparse
import json

from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def repair_json_text(text: str) -> str:
    text = text.strip()

    if not text:
        return text

    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"

    return text


def normalize_json_string(s: str) -> str:
    s = repair_json_text(s)
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--show_correct", type=int, default=10)
    parser.add_argument("--show_errors", type=int, default=10)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    rows = load_jsonl(args.test_file)

    total = len(rows)
    exact_match = 0
    valid_json = 0
    error_cases = []

    print("=" * 80)
    print("Sample predictions")
    print("=" * 80)

    shown_correct = 0

    for idx, row in enumerate(rows):
        instruction = row["instruction"]
        gold = row["output"]

        input_text = f"Instruction: {instruction}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )

        pred_raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred = repair_json_text(pred_raw)

        pred_norm = normalize_json_string(pred)
        gold_norm = normalize_json_string(gold)

        is_exact = pred_norm == gold_norm
        is_valid = pred_norm != ""

        if is_exact:
            exact_match += 1
        else:
            error_cases.append({
                "instruction": instruction,
                "gold": gold,
                "prediction_raw": pred_raw,
                "prediction_repaired": pred,
                "prediction_normalized": pred_norm,
            })

        if is_valid:
            valid_json += 1

        if shown_correct < args.show_correct:
            print(f"\nCase {shown_correct + 1}")
            print(f"Instruction: {instruction}")
            print(f"Gold:        {gold}")
            print(f"Prediction:  {pred}")
            print(f"Match:       {is_exact}")
            shown_correct += 1

    print("\n" + "=" * 80)
    print("Final metrics")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Exact match: {exact_match / total:.4f}")
    print(f"Valid JSON rate: {valid_json / total:.4f}")

    print("\n" + "=" * 80)
    print("Incorrect cases")
    print("=" * 80)

    if not error_cases:
        print("No incorrect cases found.")
    else:
        for i, case in enumerate(error_cases[:args.show_errors]):
            print(f"\nError Case {i + 1}")
            print(f"Instruction:            {case['instruction']}")
            print(f"Gold:                   {case['gold']}")
            print(f"Prediction (raw):       {case['prediction_raw']}")
            print(f"Prediction (repaired):  {case['prediction_repaired']}")
            print(f"Prediction (normalized):{case['prediction_normalized']}")


if __name__ == "__main__":
    main()