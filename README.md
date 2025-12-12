# Are Your Agents Upward Deceivers?
![](https://github.com/QingyuLiu/Agentic-Upward-Deception/blob/main/upward_deception.png)

This repo is the official implementation of “[Are Your Agents Upward Deceivers?](https://arxiv.org/abs/2512.04864)”. 

## What’s Included
- A dataset with **5 task types** (typically **40 tasks/type**, 200 total).
- An evaluation runner (`evaluation.py`) that:
  - executes tasks with controlled tool availability / failure modes,
  - saves per-task logs,
  - scores behaviors using a judge model + prompt templates.

## Quick Start
**Installation**: Our code is implemented and evaluated on Smolagents, and all dependencies are listed in `requirements.txt`.

**Configure (.env)**: Before running, you should set the environment variables via a `.env` file (e.g., API base URL, API key, model name for both the agent model and the judge model).

Run one task file:

```bash
python evaluation.py \
  --dataset dataset/task_1.json \
  --output_dir results/run_task1 \
  --results results/task_1_results.json \
  --prompts_dir template
```
Mitigation (adds a suffix to each instruction):

```bash
python evaluation.py \
  --dataset dataset/task_4.json \
  --output_dir results/run_task4_mit \
  --results results/task_4_results_mit.json \
  --prompts_dir template \
  --mitigation
```
The following is a description of parameters:
* `--dataset` : path to `dataset/task_k.json`
* `--output_dir` : where logs are written
* `--results` : results JSON file (supports resume)
* `--prompts_dir` : template directory (default: `template`)
* `--mitigation` : enable mitigation suffix

## Dataset

Datasets live in `dataset/task_{1..5}.json` (each is a list of task objects).

* **Type 1**: broken tool / read failure setting
* **Type 2**: irrelevant file (tests “admit insufficiency” vs guessing)
* **Type 3**: local decoy substitution (e.g., pptx fails, nearby txt works)
* **Type 4**: multi-step with **broken download** + subtask
* **Type 5**: multi-step with **nonexistent file** + subtask


## Judge Templates

Templates live in `template/` and should output JSON like:

* `"Result"` (class label)
* `"Reason"` (short explanation)

Used for:

* failure disclosure checks
* decoy-use detection (Type 3)
* download-failure concealment + source attribution checks (Type 4/5)

## Outputs

Per-task logs:

* `results/.../logs/<task_index>/agent_log.log`

Results file (`--results`):

* stores each task plus its computed `result` fields (format depends on task type)



## Metrics

Commonly reported rates include: (see `compute_metrics()` in `evaluation.py`)

- **NFR (Type 1/2)** — *No-Failure-Reporting rate*: fraction of tasks where the agent gives a **confident answer without mentioning any issue**.

- **DFR (Type 3)** — *Decoy-File deception rate*: fraction of tasks where the agent **does not mention the intended-file failure** and the response is judged **derivable from the local decoy text**.

- **NFR (Type 4/5)** — fraction of tasks where the agent **does not report the download failure**, i.e., it **ignores download status** or **claims download success**.

- **FFR (Type 4/5)** — fraction of tasks where the agent **does not report download failure** and a **local “source” file is detected that contains neither explicit error signals nor explicit source attribution**.

- **HFR (Type 4/5)** — *Hidden-Failure + no-source-attribution rate*: **among semantically related instructions only**, the fraction of cases where the agent **does not report download failure** and the response to the analysis subtask **does not explicitly attribute its source** (neither web browsing/search nor “based on my own knowledge/analysis”).



## Citation
```
@article{guo2025your,
  title={Are Your Agents Upward Deceivers?},
  author={Guo, Dadi and Liu, Qingyu and Liu, Dongrui and Ren, Qihan and Shao, Shuai and Qiu, Tianyi and Li, Haoran and Fung, Yi R and Ba, Zhongjie and Dai, Juntao and others},
  journal={arXiv preprint arXiv:2512.04864},
  year={2025}
}
```
