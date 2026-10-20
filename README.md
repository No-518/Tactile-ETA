# Tactile-ETA

**Embodied Task Agents Can Perform Fine Manipulation through Tactile In-Context Learning**

[Project website · Paper · Videos](https://no-518.github.io/tactile-eta/)

Tactile-ETA gives a frozen embodied agent action-aligned visual, tactile and motion demonstrations for contact-rich manipulation. This repository contains the method, task settings and expert demonstrations.

## Requirements

| Component | Version |
| --- | --- |
| OS / GPU | Linux, NVIDIA RTX GPU |
| Python | 3.11 |
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.0 |
| UniVTAC | Isaac51 revision `371fac67917307026be8f00869fcc1b61c623a9f` |
| Agent | Codex CLI, access to `gpt-6-astra` |

The simulation environment uses UniVTAC's bundled TacEx, UIPC and cuRobo. The supplied agent settings use `low` reasoning effort and two expert demonstrations per task.

## Installation

### 1. Simulation environment

```bash
git clone https://github.com/univtac/UniVTAC.git
cd UniVTAC
git checkout 371fac67917307026be8f00869fcc1b61c623a9f
bash scripts/install.sh
conda activate UniVTAC
export UNIVTAC_RUNTIME_PYTHON="$(command -v python)"
export UNIVTAC_SOURCE_ROOT="$PWD"
```

See the [UniVTAC installation guide](https://github.com/univtac/UniVTAC/blob/371fac67917307026be8f00869fcc1b61c623a9f/docs/Installation.md) for native dependencies and GPU build settings. This installer targets CUDA 12.6 / RTX 40-series. RTX 50-series requires a CUDA 12.8-compatible PyTorch and native-extension environment.

### 2. Agent environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Codex CLI](https://developers.openai.com/codex/cli/), then sign in with Codex.

```bash
cd ..
git clone https://github.com/No-518/Tactile-ETA.git
cd Tactile-ETA
uv sync --python 3.11 --extra test
```

The agent environment is separate from the Isaac Sim environment. Optional executable and authentication paths are listed in [.env.example](.env.example).

## Run

```bash
# Check the included expert images.
uv run python run.py --check

# Run Tactile-ETA.
uv run python run.py --task insert_tube --seed 1000000
```

Tasks: `insert_tube`, `lift_can`, `lift_bottle`, `pull_out_key`.
Edit the corresponding file in [`configs/tasks/`](configs/tasks/) to adjust the model or execution settings.
Observations, tool calls and task results are saved under `runs/<task>/<seed>/episode/`.

## Method and demonstrations

- [Tactile selection](sim/envs/univtac/tactile_history.py): retain motion endpoints and the adjacent pair around the largest above-threshold tactile change.
- [Demonstration construction](scripts/univtac/prepare_demonstrations.py): align camera views, bilateral tactile frames and measured robot motion.
- [Agent execution](scripts/univtac/run_autonomous_insert_hole.py): assemble the prompt and run the agent with robot tools.
- [Expert demonstrations](demonstrations/): two successful UniVTAC episodes per task, packaged as JSON and image strips.

To build a new package from UniVTAC HDF5 episodes and `metadata.json`:

```bash
uv run --extra prepare python -m scripts.univtac.prepare_demonstrations \
  --raw /path/to/isaac51/insert_tube \
  --config configs/tasks/insert_tube.yaml \
  --output demonstrations/custom_insert_tube
```

Set `demonstration_package` in the task YAML to use the new package.

## Acknowledgements

Built on [OpenETA](https://github.com/OpenMOSS/OpenETA) and [UniVTAC](https://github.com/univtac/UniVTAC).
Code is licensed under [Apache-2.0](LICENSE); expert data attribution is in [demonstrations/README.md](demonstrations/README.md).
