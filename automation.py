
import os
import json
import time
from pathlib import Path
import yaml
import requests
from datetime import datetime
from jinja2 import Template
import xml.etree.ElementTree as ET

CONFIG_PATH = "config.yaml"

from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader("."),  # or your full prompt directory
    autoescape=select_autoescape(enabled_extensions=("jinja2",))
)

def extract_relevant_xml(xml_text):
    try:
        root = ET.fromstring(xml_text)
        # Find all <pou> elements
        pous = root.findall(".//{*}pou")
        for pou in pous:
            # Remove <body>, <documentation>, etc. inside each POU
            for tag in ["body", "documentation"]:
                elem = pou.find(f"./{{*}}{tag}")
                if elem is not None:
                    pou.remove(elem)
        # Return the cleaned XML
        return ET.tostring(root, encoding="unicode")
    except Exception as e:
        print(f"XML parsing failed: {e}")
        return xml_text  # fallback to original

def load_template(path):
    return env.get_template(path)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

# def load_template(path):
#     with open(path) as f:
#         return Template(f.read())

def read_file(path):
    with open(path) as f:
        return f.read()

def send_to_lmstudio(prompt, cfg):
    url = "LLM URL" # add your local LLM url here
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.4),
        "top_p": cfg.get("top_p", 0.95),
        "n": cfg.get("n", 1),
        "stop": cfg.get("stop"),
        "max_tokens": cfg.get("max_tokens", 1024)
    }

    start = time.time()
    response = requests.post(url, headers=headers, json=payload)
    duration = time.time() - start

    try:
        data = response.json()
        total_tokens = data.get("usage", {}).get("total_tokens", None)
        tps = round(total_tokens / duration, 2) if total_tokens else None
    except Exception:
        data = {"error": response.text}
        tps = None

    return data, duration, tps

def log_result(log_path, metadata):
    with open(log_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")

# This method iterates through the folder containg the programs in dedicated folders each having an XML file for source code
# and a CSV file for the tests. For each program it creates a results folder and there it saves output logs for each prompt
# performed. Each file is named so the according to the prompt technique + method + run index, eg. 0s-add-log1 is a result
# for the first run of a zero shot ADD prompt. 
def main():
    cfg = load_config(CONFIG_PATH)
    program_root = Path(cfg["program_root"])
    for program_folder in program_root.iterdir():
        if not program_folder.is_dir():
            continue

        xml_files = list(program_folder.glob("*.xml"))
        csv_files = list(program_folder.glob("*.csv"))

        # require exactly one XML and one CSV
        if not xml_files:
            print(f"Error: Missing .xml source code in {program_folder}")
            continue
        if not csv_files:
            print(f"Error: Missing .csv test cases in {program_folder}")
            continue

        # source_code = read_file(xml_files[0])
        raw_xml = read_file(xml_files[0])
        source_code = extract_relevant_xml(raw_xml)

        test_cases = read_file(csv_files[0])

        for task in cfg["tasks"]:
            for mode in task["prompt_modes"]:
                template = load_template(mode["template_path"])
                for run_index in range(1, cfg.get("repeats_per_task", 5) + 1):
                    prompt = template.render(
                        source_code=source_code,
                        test_cases=test_cases or "",
                        task_description=task["name"]
                    )

                    response, duration, tps = send_to_lmstudio(prompt, cfg)
                    log_data = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "task": task["name"],
                        "prompt_type": mode["type"],
                        "run_index": run_index,
                        "model": cfg["model"],
                        "temperature": cfg["temperature"],
                        "top_p": cfg["top_p"],
                        "n": cfg["n"],
                        "stop": cfg["stop"],
                        "duration": duration,
                        "tokens_per_second": tps,
                        "prompt": prompt,
                        "response": response,
                    }

                    shot = "fs" if mode["type"] == "few_shot" else "0s"
                    if "generation" in task["name"]:
                        task_code = "gen"
                    elif "add" in task["name"]:
                        task_code = "add"
                    elif "mod" in task["name"]:
                        task_code = "mod"
                    else:
                        task_code = "unk"

                    # Create results subfolder if it doesn't exist
                    results_folder = program_folder / "results"
                    results_folder.mkdir(exist_ok=True)

                    # Build filename
                    filename = f"{shot}-{task_code}-log{run_index}.json"
                    log_path = results_folder / filename

                    # Save single file per run
                    with open(log_path, "w") as f:
                        json.dump(log_data, f, indent=2)
                    print(f"Logged {task['name']} ({mode['type']}) run {run_index} for {program_folder.name}")


if __name__ == "__main__":
    main()