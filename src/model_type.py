import json

path = "./models/smart_job_model/config.json"
with open(path, "r") as f:
    cfg = json.load(f)

cfg["model_type"] = "bert"

with open(path, "w") as f:
    json.dump(cfg, f, indent=4)

print("✅ model_type: bert eklendi")