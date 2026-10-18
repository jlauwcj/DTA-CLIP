# DTA-CLIP: Decoupled Temporal Adaptation for Generalizable Video Foundation Models

Official PyTorch implementation of **DTA-CLIP** (Decoupled Temporal Adaptation for Generalizable Video Foundation Models).

## 🚀 Introduction

Adapting vision-language models (like CLIP) for video understanding often distorts their pre-trained semantic manifolds, causing catastrophic forgetting and compromising zero-shot generalization. **DTA-CLIP** fundamentally breaks this adaptation-generalization trade-off through explicit architectural decoupling and orthogonal optimization.

By physically separating spatial semantics from temporal dynamics, DTA-CLIP achieves state-of-the-art performance on motion-heavy datasets (e.g., Something-Something V2) while strictly preserving its cross-domain zero-shot transferability (e.g., UCF-101).

### ✨ Core Modules

*   **Scale-Aligned Temporal Decoding (SATD)**: Solves the scale collapse of positional encodings via pre-normalization and constant-scale encoding, reviving sequence-aware, object-centric temporal aggregation.
*   **Motion Difference Branch (MDB)**: Explicitly extracts inter-frame physical displacements, bypassing redundant spatial computation and forcing the network to focus on true motion, thus eliminating static appearance biases.
*   **Manifold-Preserving Decoupled Optimization**: Employs a forward-gated gradient scaling mechanism to physically shield the pre-trained CLIP manifold from aggressive task-specific temporal updates.

## 🛠️ Architecture
![DTA-CLIP Architecture](docs/architecture.png)
<!-- TODO: Upload your final architecture image to a `docs/` folder and link it here -->

## 📦 Requirements & Installation

1. Create a conda environment and activate it:
```bash
conda create -n dtaclip python=3.11
conda activate dtaclip
