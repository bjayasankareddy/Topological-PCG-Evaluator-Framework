# **Topological Evaluation of LLM-Driven PCG via R-GCN**

**Author:** Basireddy Jaya Sankar Reddy

**Associated Publication:** *Journal of Artificial Intelligence Research (JAIR) \[Submitted 2026\]*

This repository contains the official PyTorch implementation, dataset generation scripts, and evaluation framework for verifying Large Language Model (LLM) narrative consistency in procedural environments using Relational Graph Convolutional Networks (R-GCN).

## **Overview and Architecture**

Large Language Models frequently suffer from logical "hallucinations" when generating Procedural Content Generation (PCG) events. Over prolonged temporal horizons, these errors compound into catastrophic world-state continuity paradoxes (e.g., teleportation, looting items from locked rooms, or interacting with absent NPCs) known as the "Snowball Effect."

This framework replaces subjective, computationally expensive LLM-as-a-judge evaluators with a mathematically deterministic topology gatekeeper.

1. **Heterogeneous Graphing:** Maps a dynamic 2048-node game state into distinct adjacency matrices: Spatial, Social, and Inventory.  
2. **Context Aggregation:** Utilizes a 3-layer R-GCN to project entity features into a 512-dimension latent workspace.  
3. **Link Prediction:** Evaluates proposed LLM actions as topological edges, utilizing an Advanced MLP (with 0.4 Dropout) to isolate anomalies.  
4. **Zero-Trust Anti-Cheat:** Executes server-side in under 5 milliseconds (on consumer hardware), acting inherently as a robust anti-cheat gatekeeper.

## **Repository Structure**

* dataset\_generator\_2048\_hard.py: Simulates a 2048-node RPG environment and generates perfectly balanced 50/50 temporal datasets using deterministic Hard-Negative sampling.  
* dataset\_generator\_test\_10k.py: Generates an unseen inference dataset utilizing an O(1) cryptographic blocklist to guarantee unique testing events.  
* rgcn\_mlp\_training\_v2.py: The core PyTorch training architecture featuring a 512-D hidden workspace, Automatic Mixed Precision (AMP), and a One-Cycle Cosine Annealing scheduler.  
* batch\_evaluate\_v2.py: The inference engine that calculates Accuracy, Precision, Recall, and F1-Score using mathematical threshold tuning (Threshold \= \-0.5).  
* latency\_benchmark\_6gb.py: A high-precision CUDA/CPU benchmarking tool to verify real-time 60 FPS integration viability on local edge hardware.  
* plot\_batch\_results.py: Generates publication-ready academic learning curves from the batch evaluation outputs.  
* interactive\_playtest.py: A live, human-in-the-loop terminal simulator allowing users to manually test the gatekeeper against the pre-trained neural network.

## **Installation and Requirements**

This framework requires **Python 3.10+** and a CUDA-enabled GPU (NVIDIA RTX 3050 or higher is recommended for full graph re-aggregations).

\# 1\. Clone the repository  
git clone \[https://github.com/YourUsername/Topological-PCG-Evaluator.git\](https://github.com/YourUsername/Topological-PCG-Evaluator.git)  
cd Topological-PCG-Evaluator

\# 2\. Install core dependencies (CUDA 11.8 configuration)  
pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/cu118\](https://download.pytorch.org/whl/cu118)  
pip install numpy tqdm matplotlib scikit-learn

## **Reproducing the Experiments (JAIR Checklist)**

To independently verify the 91.5% peak F1-Score and sub-5ms latency metrics reported in the manuscript, follow these sequential steps:

### **Step 1: Generate the Training Data**

python dataset\_generator\_2048\_hard.py

*Outputs:* temporal\_pcg\_dataset\_2048\_hard.json (100,000 deterministic events)

### **Step 2: Train the R-GCN Framework**

python rgcn\_mlp\_training\_v2.py

*Outputs:* Epoch checkpoints inside /checkpoints\_mlp\_512 and a t-SNE 2D brain visualization (brain\_embeddings.png).

### **Step 3: Generate the Unseen Inference Data**

python dataset\_generator\_test\_10k.py

*Outputs:* temporal\_pcg\_dataset\_2048\_new.json (10,000 completely unique events blocked against the training set).

### **Step 4: Execute the Batch Evaluator**

python batch\_evaluate\_v2.py

*Outputs:* batch\_evaluation\_results\_v2.json containing the detailed Confusion Matrices, Accuracy, and optimal F1-Scores across all epochs.

### **Step 5: Generate Publication Graphs**

python plot\_batch\_results.py

*Outputs:* model\_performance\_curves.png

### **Step 6: Verify Inference Latency**

python latency\_benchmark\_6gb.py

*Outputs:* Real-time hardware execution speeds for Isolated Actions vs. Full World Re-aggregations in the terminal.

## **Datasets and Checkpoints**

Due to standard version control file size limits, the massive 100k+ event JSON datasets and final .pth model weights are not hosted directly in this Git repository.

Reviewers and researchers can natively generate the exact topological datasets locally using the provided generator scripts. Alternatively, the pre-compiled Ground Truth files and pretrained Epoch 4 weights can be downloaded from our open-source Zenodo/HuggingFace repository here:
## **Model**
**jayasankarrr/Topological-Evaluation-of-LLM-Driven-PCG**
## **Dataset**
**jayasankarrr/Topological-Evaluation-of-LLM-Driven-PCG**

## **License and Citation**

This project is licensed under the MIT License \- see the LICENSE file for details. Free usage is explicitly granted for academic reproducibility and general research purposes in compliance with JAIR guidelines.

If you utilize this framework or code in your own research, please cite our paper:

@article{reddy2026topological,  
  title={Topological Evaluation of LLM-Driven Procedural Content Generation via Relational Graph Convolutional Networks},  
  author={Reddy, Basireddy Jaya Sankar},  
  journal={Journal of Artificial Intelligence Research},  
  year={2026}  
}  
