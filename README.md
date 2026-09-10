# VLA for Autonomous Driving — A Systematic Learning & Research Series

> **🗂️ Status:** Active — learning series in progress &nbsp;|&nbsp; Phase 0 (theory) → Phase 1 (use cases)
> **👤 Maintainer:** Xianbiao (XB) Hu · Smart Mobility Lab, The Pennsylvania State University
> **🧭 Focus:** Not chasing SOTA — characterizing *where and why* VLA driving policies fail, with an eye toward embodied transportation applications (ATMA, CDA, work-zone automation).

---

## Overview

This repository documents a structured, hands-on study of **Vision-Language-Action (VLA)** models for autonomous driving, following the same "theory first, then runnable use cases on open datasets" rhythm used in the lab's earlier LLM and VLM series.

The learning series runs on *generic* open driving data; applying VLA to the lab's own embodied scenarios (ATMA leader-follower, cooperative driving automation, work-zone automation) is a downstream research phase that will require lab-collected data and is intentionally out of scope here.

---

## Learning Roadmap

### Phase 0 — Theory (the foundation)

Concept-level groundwork before any code:

- **Positioning** — VLA as a *branch of the VLM lineage* (LLM → VLM → VLA), not a new foundational paradigm.
- **Embodied vs. agentic vs. physical AI** — why VLA requires a moving, sensor-bearing agent, and how that differs from virtual LLM agents.
  - *Physical AI ≈ Embodied AI* in practice; the narrative framing differs but the referent largely overlaps — not a distinction worth over-indexing on.
  - *Agentic* and *Embodied* are intersecting, not nested, circles. Agentic-but-not-embodied: an LLM coding agent. Embodied-but-weakly-agentic: a reflex-only driving controller. Their intersection — agents that plan, use tools, *and* have a body — is the real frontier VLA targets.
  - *Two orthogonal axes*: LLM → VLM → VLA is the **model lineage** axis; agentic / embodied / physical is the **deployment paradigm** axis. Both a virtual LLM agent and a VLA-driven AV are called "agents," but in different senses.
  - VLA is best understood as a *method* for building embodied/physical AI systems, not a subclass of them (subclasses are "robot," "AV," etc.).
  - The body must move and close a sensorimotor perception-action loop; the world need not be physical — VLA trains and evaluates fine in simulation (NAVSIM, CARLA). Real hardware vs. sim is a deployment question, not a VLA prerequisite.
- **Landscape & moat** — Open-source, runnable VLA models (OpenDriveVLA, AutoVLA, LMDrive, OpenEMMA, …) are overwhelmingly academic; the strongest deployed systems (Waymo EMMA, Wayve LINGO/GAIA, NVIDIA OmniDrive/Alpamayo, Tesla FSD) are industry-built and closed. The moat has shifted: VLM backbones (e.g., Qwen2.5-VL) commoditized architecture innovation, letting academia flood open-loop benchmarks like nuScenes — but industry's edge moved to proprietary real-world data, closed-loop simulation, and fleet deployment, none of which academia can access. Net: VLA-for-AV has a low entry cost for academia — which is both the opportunity and the source of noise.
- **Evaluation philosophy** — open-loop vs. closed-loop vs. pseudo-closed-loop (NAVSIM), and the "ego status is all you need" critique of open-loop metrics.
- **Action representation** — two orthogonal axes: *output level* (trajectory vs. low-level control) and *decoding mechanism* (discrete tokenization vs. continuous regression vs. diffusion), unified by the core problem of **future multimodality**.

Theory notes live under [`docs/`](docs/).

### Phase 1 — Use Cases (three clusters)

The series spine: **VLA = VLM + (action representation) + (closed perception-action loop).** Each cluster maps to one part of that spine.

**Cluster A — Action representation** *(what VLA adds, part 1)*

| UC | Goal | Model / Data |
|----|------|--------------|
| **A1** | Run baseline inference; understand the I/O contract; compute open-loop L2 / collision | OpenDriveVLA-0.5B / nuScenes |
| **A2** | Build an action codebook by hand; compare token vs. regression vs. diffusion decoding on the **same** trajectory GT; quantify multimodal coverage (does regression mode-collapse?) | AutoVLA codebook tooling + custom heads / nuScenes |
| **A3** | LoRA fine-tune the action representation; test whether it learns planning or **memorizes** templated planning QA | AutoVLA / DriveLM |

**Cluster B — Closed loop & the evaluation illusion** *(what VLA adds, part 2 — the methodological spine)*

| UC | Goal | Model / Data |
|----|------|--------------|
| **B1** | Ego-state-only MLP baseline → quantify how much open-loop L2 is mere kinematic extrapolation; turn the "open-loop illusion" into hard evidence | Custom MLP / nuScenes |
| **B2** | NAVSIM PDMS consequence-aware evaluation → show that open-loop-good can be closed-loop-bad | AutoVLA / NAVSIM (CARLA/Bench2Drive optional) |

**Cluster C — Capability boundaries & failure characterization** *(the lab's signature angle; extends prior VLM findings)*

| UC | Goal | Extends prior finding |
|----|------|-----------------------|
| **C1** | Is reasoning causal or decorative? Toggle CoT (fast/slow) and check whether the trajectory changes | temporal-language illusion |
| **C2** | Command following: counterfactually swap the driver command (straight ↔ left) and test whether the action follows | text-compliance dependence |
| **C3** | Perception grounding: mask / perturb perception inputs and measure action change (scene-blind test) | geometric-reasoning ceiling |
| **C4** | **Reasoning–action coherence**: detect "says one thing, does another" (verbally yields but trajectory does not) — capstone, strongest publication potential | chain-coherence collapse |

---

## Models

| Model | Role | Notes |
|-------|------|-------|
| **OpenDriveVLA-0.5B** | Entry / on-ramp | Checkpoint on Hugging Face; single-GPU inference. Input is **3D structured perception tokens** (UniAD / mmdet3d upstream), not raw images; **open-loop only**. AAAI 2026. |
| **AutoVLA** | Primary workhorse | Action codebook + CoT (fast/slow) + RFT (GRPO); raw visual input; preprocessing consumes DriveLM `v1_1_train_nus.json`; supports NAVSIM PDMS and CARLA. NeurIPS 2025. |

Rule of thumb for this series: **OpenDriveVLA gets you in the door (A1); AutoVLA does most of the work (A2–C4).**

---

## Datasets & Benchmarks

| Dataset | Role | Loop type | Notes |
|---------|------|-----------|-------|
| **nuScenes** | Core | Open-loop | L2 / collision; primary substrate for A1, A2, B1, C1–C4 |
| **DriveLM-nuScenes** | Core | Open-loop | Planning QA is heavily templated — relevant to the A3 memorization test |
| **NAVSIM** | Evaluation | Pseudo-closed-loop | Real data + lightweight rollout (PDMS); no rendering; much lighter than CARLA |
| **CoVLA** | Optional | Open-loop | Real-world trajectory + caption; more diverse than templated DriveLM |
| **CARLA / Bench2Drive** | Optional | Closed-loop | Full simulator; heavy (GPU rendering); reserved for an overflow/HPC environment |

> **No open ATMA / CDA / work-zone VLA dataset exists.** The learning series builds VLA capability on generic urban-driving data; embodied transportation applications are a separate, later phase.

---

## Repository Structure

```
.
├── docs/                              # Phase 0 theory notes, design docs
│
├── sourcecode/                        # Use-case scripts (usecaseNX_*.py)
│   ├── clusterA_action_representation/
│   ├── clusterB_evaluation_loop/
│   └── clusterC_capability_boundaries/
│
├── datasets/                          # nuScenes / DriveLM / NAVSIM (download instructions below)
│   └── README.md
│
├── models/                            # Pointers / checkouts for OpenDriveVLA, AutoVLA
│
├── outputs/                           # Experiment outputs (mirrors sourcecode/ layout)
│
├── notebooks/                         # Exploratory analysis & result visualization
│
├── environment.yml                    # Conda environment
├── requirements.txt                   # Python dependencies
├── LICENSE
└── README.md
```

Script naming convention: `usecaseNX_description_model_dataset.py` (e.g., `usecaseA1_baseline_inference_opendrivevla_nuscenes.py`).

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/[github-username]/vla-mobility.git
cd vla-mobility
```

### 2. Environment

```bash
conda env create -f environment.yml
conda activate vla-mobility
```

> Tested on Ubuntu with NVIDIA GPUs (CUDA 12.x). OpenDriveVLA-0.5B inference fits on a single 12 GB GPU; AutoVLA fine-tuning is intended for a multi-GPU workstation (e.g., dual RTX 6000 Ada) or an HPC cluster.

### 3. Data

| Source | Where |
|--------|-------|
| nuScenes | https://www.nuscenes.org (registration required) |
| DriveLM-nuScenes | https://github.com/OpenDriveLab/DriveLM |
| NAVSIM | https://github.com/autonomousvision/navsim |

Place data under `datasets/` and see [`datasets/README.md`](datasets/README.md) for the expected layout and preprocessing notes (including the DriveLM image-path fix).

### 4. Models

```bash
# OpenDriveVLA-0.5B checkpoint (Hugging Face)
hf download DriveVLA/OpenDriveVLA-0.5B --local-dir models/opendrivevla-0.5b

# AutoVLA
git clone https://github.com/ucla-mobility/AutoVLA.git models/autovla
```

### 5. Run a use case

```bash
python sourcecode/clusterA_action_representation/usecaseA1_baseline_inference_opendrivevla_nuscenes.py \
  --config sourcecode/configs/A1.yaml
```

---

## Citation & Upstream Work

This is a research-in-progress learning series; a citation will be added if it leads to a publication. The work builds directly on:

```bibtex
@inproceedings{jiang2025survey,
  title     = {A Survey on Vision-Language-Action Models for Autonomous Driving},
  author    = {Jiang, Sicong and Huang, Zilin and Qian, Kangan and Luo, Ziang and Zhu, Tianze and Zhong, Yang and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  year      = {2025},
  eprint    = {2506.24044}
}

@misc{zhou2025opendrivevla,
  title         = {OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model},
  author        = {Zhou, Xingcheng and Han, Xuyuan and Yang, Feng and Ma, Yunpu and Tresp, Volker and Knoll, Alois},
  year          = {2025},
  eprint        = {2503.23463},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{zhou2025autovla,
  title   = {AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning},
  author  = {Zhou, Zewei and Cai, Tianhui and Zhao, Seth Z. and Zhang, Yun and Huang, Zhiyu and Zhou, Bolei and Ma, Jiaqi},
  journal = {arXiv preprint arXiv:2506.13757},
  year    = {2025}
}
```

---

## Related Resources

- 🏠 **Smart Mobility Lab:** [sites.psu.edu/xbhu](https://sites.psu.edu/xbhu/)
- 📖 **Research Atlas:** [atlas.mobilitypsu.com](https://atlas.mobilitypsu.com)
- 📚 **VLA4AD survey & resource list:** [github.com/JohnsonJiang1996/Awesome-VLA4AD](https://github.com/JohnsonJiang1996/Awesome-VLA4AD)

---

## License

Code: [MIT License](LICENSE)
