# Topological Evaluation of LLM-Driven PCG via R-GCN

**Associated Publication:** Under review — title and venue withheld for double-anonymous peer review.

This repository contains the official PyTorch implementation, dataset generation scripts, and evaluation framework for verifying Large Language Model (LLM) narrative consistency in procedural environments using Relational Graph Convolutional Networks (R-GCN).

## Overview and Architecture

Large Language Models frequently suffer from logical "hallucinations" when generating Procedural Content Generation (PCG) events. Over prolonged temporal horizons, these errors compound into cascading world-state continuity failures (e.g., teleportation, looting items from locked rooms, or interacting with absent NPCs).

This framework replaces subjective, computationally expensive LLM-as-a-judge evaluators with a mathematically deterministic topology gatekeeper.

1. **Heterogeneous Graphing:** Maps a dynamic 2048-node game state into distinct adjacency matrices: Spatial, Social, and Inventory.
2. **Context Aggregation:** Utilizes a 3-layer R-GCN to project entity features into a 512-dimensional latent workspace.
3. **Link Prediction:** Evaluates proposed LLM actions as topological edges via an MLP to isolate anomalies.
4. **Real-Time Inference:** Executes server-side in under 5 milliseconds on consumer hardware.

## Repository Structure

```
Model_triner/
  Dataset_generator/
    data_generator.py              — Simulates a 2048-node RPG environment and generates
                                     balanced 50/50 temporal datasets via Hard-Negative sampling.
    Unseen_Test_data_Generator.py  — Generates an unseen inference dataset using an O(1)
                                     hash blocklist to guarantee unique testing events.
  Model_Training.py                — Core PyTorch training architecture: 512-D hidden workspace,
                                     Automatic Mixed Precision (AMP), One-Cycle Cosine Annealing.
  optimal_LR_finder.py             — Calculates the optimal maximum learning rate before
                                     gradient explosion.

Evaluation/
  Batch_Evaluation.py              — Inference engine: Accuracy, Precision, Recall, F1-Score
                                     via threshold tuning (threshold = -0.5).
  model_evaluation.py              — Standard quantitative evaluation on a single model.
  Laptop_latancy_test.py           — High-precision CUDA/CPU benchmarking for 60 FPS viability
                                     on local edge hardware.
  colab_latancy_test.py            — Latency benchmarking for cloud-compute environments
                                     (e.g., Google Colab with T4 GPUs).
  Ablation Studies/
    ablation_models.py             — Modified baseline architectures (Standard GCN,
                                     Dot-Product, Variable Depth).
    Run-Ablation-Study.py          — Trains all ablation baselines from scratch and evaluates
                                     against the test dataset to reproduce paper Table III.
```

## Installation and Requirements

This framework requires **Python 3.10+** and a CUDA-enabled GPU (NVIDIA RTX 3050 or higher is recommended for full graph re-aggregations).

```bash
# 1. Clone the repository
git clone [ANONYMIZED — link provided upon acceptance]
cd Topological-PCG-Evaluator

# 2. Install core dependencies (CUDA 11.8 configuration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm matplotlib scikit-learn
```

## Reproducing the Experiments

To independently verify the 92.52% peak F1-Score, ablation studies, and sub-5 ms latency metrics reported in the manuscript, follow these sequential steps from the root directory.

### Step 1: Generate the Training Data

```bash
python Model_triner/Dataset_generator/data_generator.py
```

*Output:* `temporal_pcg_dataset_2048_hard.json` (100,000 deterministic events)

### Step 2: Train the R-GCN Framework

```bash
python Model_triner/Model_Training.py
```

*Output:* Epoch checkpoints and a t-SNE embedding visualization (`brain_embeddings.png`).

### Step 3: Generate the Unseen Inference Data

```bash
python Model_triner/Dataset_generator/Unseen_Test_data_Generator.py
```

*Output:* `temporal_pcg_dataset_2048_new.json` (10,000 events blocked against the training set).

### Step 4: Execute the Batch Evaluator

```bash
python Evaluation/Batch_Evaluation.py
```

*Output:* `batch_evaluation_results.json` — Confusion Matrices, Accuracy, and F1-Scores across all epochs.

### Step 5: Execute the Ablation Studies

```bash
python "Evaluation/Ablation Studies/Run-Ablation-Study.py"
```

*Output:* Terminal table reproducing the manuscript's ablation study (Table III).

### Step 6: Verify Inference Latency

```bash
python Evaluation/Laptop_latancy_test.py
```

*Output:* Hardware execution speeds for Isolated Actions vs. Full World Re-aggregations.

## Datasets and Checkpoints

Due to file size constraints, the 100k-event JSON datasets and trained `.pth` model weights are not hosted directly in this repository. Reviewers can reproduce all datasets locally using the provided generator scripts.

Pre-compiled ground-truth files and pretrained Epoch 4 weights are available at the anonymized links below, provided for reviewer access only:

- **Dataset:** [ANONYMIZED — available on request through submission portal]
- **Model weights:** [ANONYMIZED — available on request through submission portal]

*These links will be replaced with permanent public repository URLs in the camera-ready version.*

## License and Citation

This project is licensed under the MIT License — see the `LICENSE` file for details.

A citation block will be provided upon acceptance and de-anonymization of the manuscript.
