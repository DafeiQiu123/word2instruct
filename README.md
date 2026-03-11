# Natural Language to JSON Action Generation

This project fine-tunes a lightweight sequence-to-sequence model (T5-small) to convert natural language instructions into structured JSON actions.

The task simulates a simplified embodied intelligence pipeline: a user provides a command in natural language, and the system converts it into a machine-readable action representation.

---

## Project Goal

The goal is to train a generative model that maps free-form natural language instructions to strict JSON outputs.

Example:

Input:
Turn on the fan in the bedroom.

Output:
{"action":"open","device":"fan","location":"bedroom"}

---

## Dataset Generation

The dataset is synthetically generated using templates and paraphrases.

Each sample contains:
instruction – natural language command
output – target JSON action

Example:

{"instruction":"Set the living room light to full brightness.","output":"{\\"action\\":\\"set_brightness\\",\\"device\\":\\"light\\",\\"location\\":\\"living_room\\",\\"value\\":100}"}

Generate the dataset:

python generate_dataset.py

---

## Training

Run training:

python src/train.py --train_file data/train.jsonl --val_file data/val.jsonl --model_name t5-small --output_dir outputs/t5-small-json --epochs 5 --batch_size 8 --lr 3e-4

The script loads the dataset, tokenizes instructions and targets, fine-tunes T5, and evaluates after each epoch.

---

## Inference

Run inference:

python src/infer.py --model_dir outputs/t5-small-json --text "Turn on the fan in the bedroom."

Example output:

{"action":"open","device":"fan","location":"bedroom"}

---

## Evaluation

Run evaluation:

python src/evaluate.py --model_dir outputs/t5-small-json --test_file data/test.jsonl

Metrics:
Exact Match – prediction exactly matches target JSON
Valid JSON Rate – prediction can be parsed as JSON

Example result:

Exact Match: 0.995
Valid JSON Rate: 1.000

---

## Challenges & Solutions

Challenge – JSON formatting errors

Early in training, the model sometimes generated outputs without the outer braces:

"action":"open","device":"fan","location":"bedroom"

Solution

A lightweight post-processing repair function was added during evaluation and inference to automatically add missing braces and ensure valid JSON.

---


## Results

After training on the expanded dataset:

Exact Match: 0.995
Valid JSON Rate: 1.000

The model correctly generates structured JSON actions for almost all test instructions.

---

## Error Analysis

Although the model performed very strongly overall, a small number of errors remained on the test set.

### Failure Case 1

**Input**
In the hallway, dim the light.

**Expected**

{"action":"set_brightness","device":"light","location":"hallway","value":30}

**Predicted**

{"action":"close","device":"light","location":"hallway","value":30}

The model correctly identified the device, location, and value, but predicted the wrong action type. It appears to have confused a brightness adjustment command with an on/off-style control action.

### Failure Case 2

**Input**

For the TV in the kitchen, switch to the kids channel.

**Expected**

{"action":"set_channel","device":"tv","location":"kitchen","value":"kids"}

**Predicted**

{"action":"open","device":"tv","location":"kitchen","value":"kids"}

The model correctly extracted the target device, location, and channel value, but generated the wrong action label. This suggests that the model still occasionally confuses action categories with similar control semantics.

## Inference Error Analysis

After fine-tuning the model, I tested it using several natural language instructions that were **not present in the training or test datasets**. This experiment was designed to evaluate the model's ability to generalize to new phrasings and unseen instruction patterns. Most of the outputs were correct and remained in valid JSON format. However, the main type of error I observed was still **action label confusion**.

**Input**

Switch the TV in the kitchen over to kids.

**Predicted**

{"action":"set_volume","device":"tv","location":"kitchen","value":"kids"}

**Expected**

{"action":"set_channel","device":"tv","location":"kitchen","value":"kids"}

The unseen-instruction inference results show that the model generalizes well to new phrasing patterns. Most predictions were correct and formatted as valid JSON. Among the tested examples, the only incorrect case involved the `action` field, while the `device`, `location`, and `value` fields were still predicted correctly. This suggests that the action label is the most difficult part of the schema for the model to learn robustly. The error likely comes from semantic overlap between related control actions such as `set_channel`, `set_volume`, and `set_brightness`. 
