# Expert demonstrations

Each task includes official successful episodes **0 and 1** from the UniVTAC Isaac Sim 5.1 dataset, represented as action-aligned camera pairs, bilateral tactile strips and measured robot motion.

- Source: [UniVTAC dataset](https://huggingface.co/datasets/byml/UniVTAC), `isaac51/<task>/`.
- Upstream: [UniVTAC](https://github.com/univtac/UniVTAC).
- Data license: MIT, as declared by the upstream dataset card.
- Processing: action segmentation, boundary camera selection, tactile change-based frame selection and image-strip assembly. The implementation is in [`prepare_demonstrations.py`](../scripts/univtac/prepare_demonstrations.py).

`expert_demonstrations.json` contains the context text and the corresponding paths under `images/`.
