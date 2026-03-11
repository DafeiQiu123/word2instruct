import json
import os
import random

random.seed(42)

LOCATIONS = [
    ("living room", "living_room"),
    ("bedroom", "bedroom"),
    ("kitchen", "kitchen"),
    ("study", "study_room"),
    ("bathroom", "bathroom"),
    ("dining room", "dining_room"),
    ("hallway", "hallway"),
]

SAMPLES = []


def add_sample(instruction: str, output_dict: dict):
    instruction = " ".join(instruction.strip().split())
    SAMPLES.append({
        "instruction": instruction,
        "output": json.dumps(output_dict, ensure_ascii=False, separators=(",", ":"))
    })


def build_light_samples():
    brightness_phrases = [
        ("set the brightness to maximum", 100),
        ("set the light to full brightness", 100),
        ("make the light brighter", 70),
        ("brighten the light", 70),
        ("dim the light", 30),
        ("make the light dimmer", 30),
        ("set the brightness to 50 percent", 50),
        ("set the brightness to 70 percent", 70),
        ("set the brightness to 30 percent", 30),
        ("turn the brightness up", 80),
        ("turn the brightness down", 20),
        ("increase the brightness", 80),
        ("decrease the brightness", 20),
    ]

    templates = [
        "Set the {location} light to {phrase}.",
        "Please {phrase} in the {location}.",
        "Adjust the {location} light and {phrase}.",
        "In the {location}, {phrase}.",
        "Can you {phrase} in the {location}?",
        "I want you to {phrase} in the {location}.",
        "For the {location} light, {phrase}.",
        "Could you {phrase} for the light in the {location}?",
        "{phrase} in the {location} light.",
        "Please set the light in the {location} and {phrase}.",
    ]

    for location_text, location_json in LOCATIONS:
        for phrase, value in brightness_phrases:
            for template in templates:
                instruction = template.format(location=location_text, phrase=phrase)
                add_sample(instruction, {
                    "action": "set_brightness",
                    "device": "light",
                    "location": location_json,
                    "value": value
                })


def build_ac_samples():
    temperature_phrases = [
        ("set the temperature to 18 degrees", 18),
        ("set the temperature to 19 degrees", 19),
        ("set the temperature to 20 degrees", 20),
        ("set the temperature to 21 degrees", 21),
        ("set the temperature to 22 degrees", 22),
        ("set the temperature to 24 degrees", 24),
        ("set the temperature to 25 degrees", 25),
        ("set the temperature to 26 degrees", 26),
        ("lower the temperature to 19 degrees", 19),
        ("raise the temperature to 26 degrees", 26),
        ("make it cooler to 20 degrees", 20),
        ("make it warmer to 24 degrees", 24),
        ("cool it down to 18 degrees", 18),
        ("heat it up to 25 degrees", 25),
    ]

    templates = [
        "Set the {location} air conditioner to {phrase}.",
        "Please {phrase} in the {location} air conditioner.",
        "Adjust the {location} AC and {phrase}.",
        "In the {location}, {phrase} on the air conditioner.",
        "Can you {phrase} for the {location} AC?",
        "For the air conditioner in the {location}, {phrase}.",
        "Please adjust the AC in the {location} and {phrase}.",
        "{phrase} on the {location} air conditioner.",
        "Could you {phrase} using the air conditioner in the {location}?",
        "Set the AC in the {location} and {phrase}.",
    ]

    for location_text, location_json in LOCATIONS:
        for phrase, value in temperature_phrases:
            for template in templates:
                instruction = template.format(location=location_text, phrase=phrase)
                add_sample(instruction, {
                    "action": "set_temperature",
                    "device": "air_conditioner",
                    "location": location_json,
                    "value": value
                })


def build_curtain_samples():
    phrase_map = [
        ("open", "open"),
        ("close", "close"),
        ("pull open", "open"),
        ("pull shut", "close"),
        ("draw open", "open"),
        ("draw closed", "close"),
        ("open up", "open"),
        ("shut", "close"),
    ]

    templates = [
        "{phrase} the {location} curtain.",
        "Please {phrase} the curtain in the {location}.",
        "In the {location}, {phrase} the curtains.",
        "Can you {phrase} the {location} curtain?",
        "For the {location}, {phrase} the curtain.",
        "Please {phrase} the curtains in the {location}.",
        "{phrase} the curtains located in the {location}.",
        "Could you {phrase} the curtain for the {location}?",
    ]

    for location_text, location_json in LOCATIONS:
        for phrase, action_name in phrase_map:
            for template in templates:
                instruction = template.format(location=location_text, phrase=phrase)
                add_sample(instruction, {
                    "action": action_name,
                    "device": "curtain",
                    "location": location_json
                })


def build_tv_channel_samples():
    channel_phrases = [
        ("switch to the news channel", "news"),
        ("switch to the sports channel", "sports"),
        ("switch to the movie channel", "movie"),
        ("switch to the kids channel", "kids"),
        ("change to the news channel", "news"),
        ("change to the sports channel", "sports"),
        ("put it on the movie channel", "movie"),
        ("set it to the kids channel", "kids"),
    ]

    templates = [
        "In the {location}, {phrase} on the TV.",
        "Please {phrase} for the TV in the {location}.",
        "Set the {location} TV to {phrase}.",
        "Change the channel in the {location} and {phrase}.",
        "Could you {phrase} on the television in the {location}?",
        "For the TV in the {location}, {phrase}.",
        "{phrase} on the {location} TV.",
        "Please use the TV in the {location} and {phrase}.",
    ]

    for location_text, location_json in LOCATIONS:
        for phrase, channel in channel_phrases:
            for template in templates:
                instruction = template.format(location=location_text, phrase=phrase)
                add_sample(instruction, {
                    "action": "set_channel",
                    "device": "tv",
                    "location": location_json,
                    "value": channel
                })


def build_tv_volume_samples():
    volume_phrases = [
        ("set the volume to 10", 10),
        ("set the volume to 15", 15),
        ("set the volume to 20", 20),
        ("set the volume to 25", 25),
        ("set the volume to 30", 30),
        ("turn the volume down a little", 15),
        ("turn the volume up a little", 35),
        ("lower the volume", 15),
        ("raise the volume", 35),
        ("make it quieter", 10),
        ("make it louder", 40),
        ("reduce the volume", 15),
        ("increase the volume", 35),
    ]

    templates = [
        "In the {location}, {phrase} on the TV.",
        "Please {phrase} for the {location} TV.",
        "Adjust the TV in the {location} and {phrase}.",
        "Set the {location} television and {phrase}.",
        "Could you {phrase} on the TV in the {location}?",
        "For the television in the {location}, {phrase}.",
        "{phrase} on the {location} TV.",
        "Please use the TV in the {location} and {phrase}.",
    ]

    for location_text, location_json in LOCATIONS:
        for phrase, value in volume_phrases:
            for template in templates:
                instruction = template.format(location=location_text, phrase=phrase)
                add_sample(instruction, {
                    "action": "set_volume",
                    "device": "tv",
                    "location": location_json,
                    "value": value
                })


def build_fan_samples():
    fan_phrases = [
        ("turn on", "open"),
        ("turn off", "close"),
        ("switch on", "open"),
        ("switch off", "close"),
        ("power on", "open"),
        ("power off", "close"),
        ("start", "open"),
        ("stop", "close"),
    ]

    templates = [
        "{phrase} the fan in the {location}.",
        "Please {phrase} the {location} fan.",
        "In the {location}, {phrase} the fan.",
        "Can you {phrase} the fan located in the {location}?",
        "For the {location}, {phrase} the fan.",
        "{phrase} the fan in the {location} for me.",
        "Please use the fan in the {location} and {phrase} it.",
        "Could you {phrase} the {location} fan?",
    ]

    for location_text, location_json in LOCATIONS:
        for phrase, action_name in fan_phrases:
            for template in templates:
                instruction = template.format(location=location_text, phrase=phrase)
                add_sample(instruction, {
                    "action": action_name,
                    "device": "fan",
                    "location": location_json
                })


def deduplicate_samples(samples):
    seen = set()
    unique = []
    for item in samples:
        key = (item["instruction"], item["output"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def split_and_save(samples, output_dir="data"):
    samples = deduplicate_samples(samples)
    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]

    os.makedirs(output_dir, exist_ok=True)

    for filename, dataset in [
        ("train.jsonl", train_data),
        ("val.jsonl", val_data),
        ("test.jsonl", test_data),
    ]:
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total samples: {total}")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")


if __name__ == "__main__":
    build_light_samples()
    build_ac_samples()
    build_curtain_samples()
    build_tv_channel_samples()
    build_tv_volume_samples()
    build_fan_samples()

    split_and_save(SAMPLES, output_dir="data")