import os

_base_config = os.path.join(
    os.path.dirname(__file__),
    "sparsedrive_r50_stage2_4gpu_bs24_planrefine3.py",
)
with open(_base_config, "r") as f:
    exec(f.read())

log_config["hooks"][1]["init_kwargs"]["name"] = "sparsedrive_r50_stage2_4gpu_nomap_planrefine3"

task_config["with_map"] = False
model["head"]["task_config"] = task_config

eval_mode["with_map"] = False
