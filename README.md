# **Topological Evaluation of LLM-Driven PCG via R-GCN**

**Author:** Basireddy Jaya Sankar Reddy

**Associated Publication:** Journal of Artificial Intelligence Research (JAIR)

![][image1]This repository contains the official PyTorch implementation, dataset generation scripts, and evaluation framework for verifying Large Language Model (LLM) narrative consistency in procedural environments using Relational Graph Convolutional Networks (R-GCN).

## **Overview and Architecture**

Large Language Models frequently suffer from logical "hallucinations" when generating Procedural Content Generation (PCG) events. Over prolonged temporal horizons, these errors compound into catastrophic world-state continuity paradoxes (e.g., teleportation, looting items from locked rooms, or interacting with absent NPCs) known as the "Snowball Effect."

This framework replaces subjective, computationally expensive LLM-as-a-judge evaluators with a mathematically deterministic topology gatekeeper.

1. **Heterogeneous Graphing:** Maps a dynamic 2048-node game state into distinct adjacency matrices: Spatial, Social, and Inventory.  
2. **Context Aggregation:** Utilizes a 3-layer R-GCN to project entity features into a 512-dimension latent workspace.  
3. **Link Prediction:** Evaluates proposed LLM actions as topological edges, utilizing an Advanced MLP (with 0.4 Dropout) to isolate anomalies.  
4. **Zero-Trust Anti-Cheat:** Executes server-side in under 5 milliseconds (on consumer hardware), acting inherently as a robust anti-cheat gatekeeper.

## **Repository Structure**

* Model\_triner/  
  * Dataset\_generator/data\_generator.py: Simulates a 2048-node RPG environment and generates perfectly balanced 50/50 temporal datasets using deterministic Hard-Negative sampling.  
  * Dataset\_generator/Unseen\_Test\_data\_Generator.py: Generates an unseen inference dataset utilizing an O(1) cryptographic blocklist to guarantee unique testing events.  
  * Model\_Training.py: The core PyTorch training architecture featuring a 512-D hidden workspace, Automatic Mixed Precision (AMP), and a One-Cycle Cosine Annealing scheduler.  
  * optimal\_LR\_finder.py: Calculates the optimal maximum learning rate before gradient explosion.  
* Evaluation/  
  * Batch\_Evaluation.py: The inference engine that calculates Accuracy, Precision, Recall, and F1-Score using mathematical threshold tuning (Threshold \= \-0.5).  
  * model\_evaluation.py: Runs standard quantitative evaluation metrics on a single isolated model.  
  * Laptop\_latancy\_test.py: A high-precision CUDA/CPU benchmarking tool to verify real-time 60 FPS integration viability on local edge hardware.  
  * colab\_latancy\_test.py: Latency benchmarking optimized for cloud-compute environments (e.g., Google Colab with T4 GPUs).  
  * Ablation Studies/  
    * ablation\_models.py: Contains the modified baseline neural network architectures (Standard GCN, Dot-Product, Variable Depth).  
    * Run-Ablation-Study.py: An automated execution script that trains all ablation baselines from scratch and evaluates them against the test dataset to generate the paper's ablation metrics.

## **Installation and Requirements**

This framework requires **Python 3.10+** and a CUDA-enabled GPU (NVIDIA RTX 3050 or higher is recommended for full graph re-aggregations).

\# 1\. Clone the repository  
git clone \[https://github.com/bjayasankareddy/Topological-PCG-Evaluator-Framework.git](https://github.com/bjayasankareddy/Topological-PCG-Evaluator-Framework.git)  
cd Topological-PCG-Evaluator

\# 2\. Install core dependencies (CUDA 11.8 configuration)  
pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/cu118\](https://download.pytorch.org/whl/cu118)  
pip install numpy tqdm matplotlib scikit-learn

## **Reproducing the Experiments (JAIR Checklist)**

To independently verify the 92.52% peak F1-Score, ablation studies, and sub-5ms latency metrics reported in the manuscript, follow these sequential steps from the root directory:

### **Step 1: Generate the Training Data**

python Model\_triner/Dataset\_generator/data\_generator.py

*Outputs:* temporal\_pcg\_dataset\_2048\_hard.json (100,000 deterministic events)

### **Step 2: Train the R-GCN Framework**

python Model\_triner/Model\_Training.py

*Outputs:* Epoch checkpoints and a t-SNE 2D brain visualization (brain\_embeddings.png).

### **Step 3: Generate the Unseen Inference Data**

python Model\_triner/Dataset\_generator/Unseen\_Test\_data\_Generator.py

*Outputs:* temporal\_pcg\_dataset\_2048\_new.json (10,000 completely unique events blocked against the training set).

### **Step 4: Execute the Batch Evaluator**

python Evaluation/Batch\_Evaluation.py

*Outputs:* batch\_evaluation\_results.json containing the detailed Confusion Matrices, Accuracy, and optimal F1-Scores across all epochs.

### **Step 5: Execute the Ablation Studies**

python "Evaluation/Ablation Studies/Run-Ablation-Study.py"

*Outputs:* A terminal-printed table directly mirroring the manuscript's ablation study, providing exact F1-Scores for the Standard GCN, Dot-Product, and variable depth network variants.

### **Step 6: Verify Inference Latency**

python Evaluation/Laptop\_latancy\_test.py

*Outputs:* Real-time hardware execution speeds for Isolated Actions vs. Full World Re-aggregations in the terminal.

## **Datasets and Checkpoints**

Due to standard version control file size limits, the massive 100k+ event JSON datasets and final .pth model weights are not hosted directly in this Git repository.

Reviewers and researchers can natively generate the exact topological datasets locally using the provided generator scripts. Alternatively, the pre-compiled Ground Truth files and pretrained Epoch 4 weights can be downloaded from our open-source Hugging Face repository here:

**Dataset**
[https://huggingface.co/datasets/jayasankarrr/Topological-PCG-Evaluator-Framework_Dataset]
**Model**
[https://huggingface.co/jayasankarrr/Topological-Evaluation-of-LLM-Driven-PCG]
## **License and Citation**

This project is licensed under the MIT License \- see the LICENSE file for details. Free usage is explicitly granted for academic reproducibility and general research purposes in compliance with JAIR guidelines.

If you utilize this framework or code in your own research, please cite our paper:

@article{reddy2026topological,  
  title={Topological Evaluation of LLM-Driven Procedural Content Generation via Relational Graph Convolutional Networks},  
  author={Reddy, Basireddy Jaya Sankar},  
  journal={Journal of Artificial Intelligence Research},  
  year={2026}  
}  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAtCAYAAAATDjfFAAAFkklEQVR4Xu3c32scVRjG8V0SpaKiUfOjyWZnN1FTE7Vq1Jq2QgULFRuNSqVaqWIvqqAILVYMgYaWQKEoJa0oWFol+ANTUAhEpdLGVjRUEL3wRu8k4LV/gNTn2XknDAPV3JiAfj/wcs6cOXNm0quHMzstlQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID/hSRJHq5Wq293d3d3qv+56riGy8V5Gc19WXN+rlQqXcVzS9HV1VXR9d9kx7rv3T09PdXsuL29/Urd40F1m7Oxv6PnuK1Wq+32uv47VCe1/o9qt5bSv6Ps86px1+Dg4GUx9qz/VtVXqimtc53X87+H6iPVB6qx0hKfAwAA4F9Rr9fbFWy+cNjxsfrDCik7ivPyFLAe0ZwZBZ5VxXNL1NTb29vtTmtr61Vaa1b33Zid9Ppa+5PSEoOSrm1R2LpX65zQdY96TP0HVAta61a1Q6pTGi6rPao522NsWkHxGgc49T9VHfMcrfeOxyLQfZ8FOQAAgBWhcDKo+k21wccKKWtUNXXLCnHXq3ttNtc7XxFuJlVjCnt9Djxx3SqFsDZ1mxzCIsw5mLVl457n6x0Sfd5rK1Ddr7XmHaz6+vqu9jkdv6sa9f39HNk1+XXM94n5Q+p3qH1DdcDnYhdvIQLoqO71vsd1/JqOj+u6tep/ree/2eOeo5pra2vzek/43l4/uxcAAMCKceBK0leCFx2UFGw6PK7+Q7ETdd6vK/M7YRq/4NCjkNWr4/cc9lSvaPyQ2s9U23R+3sdqb9HxfvXHK5XKFTreo/6LaqfUrvNultpzqhGdv0nHz6h+iWC1vpTuik3HOlvVPu2xWGPU16g947H836XzWzT+k4Obw1ohsM3lw1g816xqr873eK76d6oOeqxUWBsAAGAlNDkQqX5XTfb391+epK8U16tOR6jz+bOdnZ03qD3v1hc63CjkHKrX67erPanj5zyuOT8oCN3ofoSkCb8GVX/A4c0V8ybV3509SP4+ce2AaoPWT9Tu07m71G5W+23swDUn6evORTH3S617h4//IbA5EL6qOuydPJ0fVP/xmOv+r36mmAsAALC8/DpQQWZTdhxhxh8cNDhMqV6K/g4HMnX9G6+sbdH1F1TbI8idzcJNhKjm2L2aqaYfETR+M6f55/wK1Nfr3GkHo9w9F+8Tx3tUj3mXLz4W8Jg/FJhw3x8+qD+fXa+1O3T9EbU1z4/wOVEIbI1n8z00/pTqBfWb/MrXH15oznDMdWD7Q+e3ZOsDAAAsK4cjV+74LYWUze7HK9AZBZh7ShHSIkxtc+jxnPj92RnvdGlso/qz2c5VLXbNkvTH/XMKan0aG3G40/GHDkMRiBzYWvyhQbxibdzHv1dzm61jDn8OaA5fSeyCJelO4KzG1vnjAPUPqL9G7Wqt9aSfK/cRg+dPJGkIdVgb0fmdnuvdQB2/7ucvBDZ22AAAwIrxq8A3HWRq6e/Bhtx6PHf+sOqgA5ra71THInBNqZ53m31dmqS/TRt3369QHXbc19imJA1l+/xKVP1d6n+s2q8a0PGcr62mQdHB8EiSfjywN17FOnjtrKVfbI7V0h204Qhtu5J0F3Am1vaXnhdzteDnixDm38rdp7ETXjee68/C/NH4W6YVMNeqPZpEuIt/EwAAgGVV9itAt94hUzBZXZxg2Zeh6pb9FadbVVP0F/nVY/bKsii3xuJxKUJQLf1aNP/fg+Tv05D91xu5OY3rst083zt/7hLKxa9ML8Vr+9+Dr0QBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPgv+guDmUgrtqJgBQAAAABJRU5ErkJggg==>
