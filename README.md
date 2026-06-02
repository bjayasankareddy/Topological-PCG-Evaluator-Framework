<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topological Evaluation of LLM-Driven PCG via R-GCN</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #34495e;
            --accent-color: #e74c3c;
            --background-color: #f8f9fa;
            --text-color: #333333;
            --code-bg: #ecf0f1;
            --border-color: #bdc3c7;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }

        header {
            background-color: var(--primary-color);
            color: white;
            padding: 3rem 1rem;
            text-align: center;
            border-bottom: 5px solid var(--accent-color);
        }

        header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }

        header p.author {
            font-size: 1.2rem;
            margin-top: 1rem;
            font-weight: 300;
        }

        header p.publication {
            font-style: italic;
            color: #bdc3c7;
            margin-top: 0.5rem;
        }

        .badges {
            margin-top: 1.5rem;
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .badges img {
            height: 24px;
        }

        main {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-radius: 8px;
            margin-top: -2rem;
        }

        h2 {
            color: var(--primary-color);
            border-bottom: 2px solid var(--code-bg);
            padding-bottom: 0.5rem;
            margin-top: 2.5rem;
        }

        h3 {
            color: var(--secondary-color);
            margin-top: 1.5rem;
        }

        p {
            margin-bottom: 1rem;
        }

        ul {
            margin-bottom: 1.5rem;
            padding-left: 1.5rem;
        }

        li {
            margin-bottom: 0.5rem;
        }

        code {
            font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
            background-color: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
            color: #c0392b;
        }

        pre {
            background-color: #282c34;
            color: #abb2bf;
            padding: 1.5rem;
            border-radius: 6px;
            overflow-x: auto;
            margin-bottom: 1.5rem;
        }

        pre code {
            background-color: transparent;
            color: inherit;
            padding: 0;
            font-size: 0.95em;
        }

        .highlight-box {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 1rem;
            margin: 1.5rem 0;
            border-radius: 0 4px 4px 0;
        }

        footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            font-size: 0.9rem;
        }

        a {
            color: #2980b9;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>

    <header>
        <h1>Topological Evaluation of LLM-Driven PCG via R-GCN</h1>
        <p class="author"><strong>Author:</strong> Basireddy Jaya Sankar Reddy</p>
        <p class="publication">Associated Publication: Journal of Artificial Intelligence Research (JAIR) [Submitted 2026]</p>
        
        <div class="badges">
            <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
            <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
            <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
        </div>
    </header>

    <main>
        <p class="highlight-box">This repository contains the official PyTorch implementation, dataset generation scripts, and evaluation framework for verifying Large Language Model (LLM) narrative consistency in procedural environments using Relational Graph Convolutional Networks (R-GCN).</p>

        <h2>Overview and Architecture</h2>
        <p>Large Language Models frequently suffer from logical "hallucinations" when generating Procedural Content Generation (PCG) events. Over prolonged temporal horizons, these errors compound into catastrophic world-state continuity paradoxes (e.g., teleportation, looting items from locked rooms, or interacting with absent NPCs) known as the "Snowball Effect."</p>
        <p>This framework replaces subjective, computationally expensive LLM-as-a-judge evaluators with a mathematically deterministic topology gatekeeper.</p>
        <ul>
            <li><strong>Heterogeneous Graphing:</strong> Maps a dynamic 2048-node game state into distinct adjacency matrices: Spatial, Social, and Inventory.</li>
            <li><strong>Context Aggregation:</strong> Utilizes a 3-layer R-GCN to project entity features into a 512-dimension latent workspace.</li>
            <li><strong>Link Prediction:</strong> Evaluates proposed LLM actions as topological edges, utilizing an Advanced MLP (with 0.4 Dropout) to isolate anomalies.</li>
            <li><strong>Zero-Trust Anti-Cheat:</strong> Executes server-side in under 5 milliseconds (on consumer hardware), acting inherently as a robust anti-cheat gatekeeper.</li>
        </ul>

        <h2>Repository Structure</h2>
        <ul>
            <li><code>dataset_generator_2048_hard.py</code>: Simulates a 2048-node RPG environment and generates perfectly balanced 50/50 temporal datasets using deterministic Hard-Negative sampling.</li>
            <li><code>dataset_generator_test_10k.py</code>: Generates an unseen inference dataset utilizing an O(1) cryptographic blocklist to guarantee unique testing events.</li>
            <li><code>rgcn_mlp_training_v2.py</code>: The core PyTorch training architecture featuring a 512-D hidden workspace, Automatic Mixed Precision (AMP), and a One-Cycle Cosine Annealing scheduler.</li>
            <li><code>batch_evaluate_v2.py</code>: The inference engine that calculates Accuracy, Precision, Recall, and F1-Score using mathematical threshold tuning (Threshold = -0.5).</li>
            <li><code>latency_benchmark_6gb.py</code>: A high-precision CUDA/CPU benchmarking tool to verify real-time 60 FPS integration viability on local edge hardware.</li>
            <li><code>plot_batch_results.py</code>: Generates publication-ready academic learning curves from the batch evaluation outputs.</li>
            <li><code>interactive_playtest.py</code>: A live, human-in-the-loop terminal simulator allowing users to manually test the gatekeeper against the pre-trained neural network.</li>
        </ul>

        <h2>Installation and Requirements</h2>
        <p>This framework requires <strong>Python 3.10+</strong> and a CUDA-enabled GPU (NVIDIA RTX 3050 or higher is recommended for full graph re-aggregations).</p>
<pre><code># 1. Clone the repository
git clone https://github.com/YourUsername/Topological-PCG-Evaluator.git
cd Topological-PCG-Evaluator

# 2. Install core dependencies (CUDA 11.8 configuration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm matplotlib scikit-learn</code></pre>

        <h2>Reproducing the Experiments (JAIR Checklist)</h2>
        <p>To independently verify the 91.5% peak F1-Score and sub-5ms latency metrics reported in the manuscript, follow these sequential steps:</p>

        <h3>Step 1: Generate the Training Data</h3>
<pre><code>python dataset_generator_2048_hard.py</code></pre>
        <p><em>Outputs:</em> <code>temporal_pcg_dataset_2048_hard.json</code> (100,000 deterministic events)</p>

        <h3>Step 2: Train the R-GCN Framework</h3>
<pre><code>python rgcn_mlp_training_v2.py</code></pre>
        <p><em>Outputs:</em> Epoch checkpoints inside <code>/checkpoints_mlp_512</code> and a t-SNE 2D brain visualization (<code>brain_embeddings.png</code>).</p>

        <h3>Step 3: Generate the Unseen Inference Data</h3>
<pre><code>python dataset_generator_test_10k.py</code></pre>
        <p><em>Outputs:</em> <code>temporal_pcg_dataset_2048_new.json</code> (10,000 completely unique events blocked against the training set).</p>

        <h3>Step 4: Execute the Batch Evaluator</h3>
<pre><code>python batch_evaluate_v2.py</code></pre>
        <p><em>Outputs:</em> <code>batch_evaluation_results_v2.json</code> containing the detailed Confusion Matrices, Accuracy, and optimal F1-Scores across all epochs.</p>

        <h3>Step 5: Generate Publication Graphs</h3>
<pre><code>python plot_batch_results.py</code></pre>
        <p><em>Outputs:</em> <code>model_performance_curves.png</code></p>

        <h3>Step 6: Verify Inference Latency</h3>
<pre><code>python latency_benchmark_6gb.py</code></pre>
        <p><em>Outputs:</em> Real-time hardware execution speeds for Isolated Actions vs. Full World Re-aggregations in the terminal.</p>

        <h2>Datasets and Checkpoints</h2>
        <p>Due to standard version control file size limits, the massive 100k+ event JSON datasets and final <code>.pth</code> model weights are not hosted directly in this Git repository.</p>
        <p>Reviewers and researchers can natively generate the exact topological datasets locally using the provided generator scripts. Alternatively, the pre-compiled Ground Truth files and pretrained Epoch 4 weights can be downloaded from our open-source Zenodo/HuggingFace repository here:<br>
        <strong>[Insert Link to Zenodo/HuggingFace Here]</strong></p>

        <h2>License and Citation</h2>
        <p>This project is licensed under the MIT License - see the <code>LICENSE</code> file for details. Free usage is explicitly granted for academic reproducibility and general research purposes in compliance with JAIR guidelines.</p>
        <p>If you utilize this framework or code in your own research, please cite our paper:</p>
<pre><code>@article{reddy2026topological,
  title={Topological Evaluation of LLM-Driven Procedural Content Generation via Relational Graph Convolutional Networks},
  author={Reddy, Basireddy Jaya Sankar},
  journal={Journal of Artificial Intelligence Research},
  year={2026}
}</code></pre>
    </main>

    <footer>
        <p>&copy; 2026 Basireddy Jaya Sankar Reddy. All rights reserved.</p>
    </footer>

</body>
</html>
