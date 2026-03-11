import argparse
import inspect
import json
import os
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_dataset(path: str) -> Dataset:
    rows = load_jsonl(path)
    inputs = [f"Instruction: {row['instruction']}" for row in rows]
    targets = [row["output"] for row in rows]
    return Dataset.from_dict({"input_text": inputs, "target_text": targets})


def preprocess_function(examples, tokenizer, max_input_length=128, max_target_length=128):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


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


def compute_metrics_builder(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_norms = [normalize_json_string(p) for p in decoded_preds]
        label_norms = [normalize_json_string(l) for l in decoded_labels]

        exact_match = accuracy_score(label_norms, pred_norms)

        json_valid_flags = [1 if p != "" else 0 for p in pred_norms]
        json_valid_rate = float(np.mean(json_valid_flags)) if json_valid_flags else 0.0

        return {
            "exact_match": float(exact_match),
            "json_valid_rate": json_valid_rate,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/val.jsonl")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--output_dir", type=str, default="outputs/t5-small-json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    train_dataset = build_dataset(args.train_file)
    val_dataset = build_dataset(args.val_file)

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        ),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args_kwargs = dict(
    output_dir=args.output_dir,
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    greater_is_better=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
    )

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()

    # Save model in a way that is usually more stable for reload
    trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)

    eval_metrics = trainer.evaluate()
    print("Validation metrics:")
    print(eval_metrics)


if __name__ == "__main__":
    main()