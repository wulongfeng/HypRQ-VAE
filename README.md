# HypRQ-VAE: Long-Tail-Aware Item Indexing for Generative Recommender Systems
This repository contains the official implementation for the paper "HypRQ-VAE: Long-Tail-Aware Item Indexing for Generative Recommender Systems."

# 📖 Introduction
Generative recommender systems, which use large language models (LLMs) to model user behavior, often struggle with a fundamental challenge: bridging the gap between an LLM's text-based vocabulary and the discrete item IDs used in recommender systems. This mismatch can lead to **hallucinations** and poor performance, especially when dealing with the **long-tail distribution** of real-world item catalogs, where a small number of popular "head" items coexist with a vast number of less popular "tail" items.

**HypRQ-VAE** is the first framework to address this by learning item indexing in **hyperbolic space**.  Unlike traditional Euclidean models, hyperbolic geometry's exponential expansion volume naturally aligns with the power-law structure of user-item interactions. This allows HypRQ-VAE to encode rich textual semantics while preserving the fidelity of **tail items** without aggressive compression. Our experiments on three benchmark datasets demonstrate that HypRQ-VAE significantly improves recommendation performance, particularly for long-tail items, outperforming Euclidean baselines.

# ⚙️ Requirements and Setup
To set up the environment, clone the repository and run the following commands.
* Create the Conda environment from the provided `environment.yml` file:

  ```conda env create -f environment.yml ```
* Activate the newly created environment:

   ```conda activate hyper_rqvae ```

# 🚀 Getting Started
This project involves a two-stage process: first, generating the hyperbolic semantic IDs, and second, using these IDs to fine-tune a generative recommender model.

## 🛠 Stage 1 -- Generation of Hyperbolic Semantic IDs
This stage involves training the HypRQ-VAE model to learn the hyperbolic item embeddings and generate the hyperbolic semantic IDs.

1. Train the HypRQ-VAE Model
Run the following script to train the tokenizer (the HypRQ-VAE model):
 
    ```bash train_tokenizer.sh ```

2. Generate Hyperbolic Semantic IDs
After training, use the following command to generate the final item IDs:

    ```bash generate_tokenizer.sh ```

## 💻 Stage 2 -- Generative Recommender with Hyperbolic Semantic IDs 
This stage uses the generated hyperbolic semantic IDs to fine-tune a generative recommender model for the recommendation task. You can choose between an encoder-decoder or a decoder-only architecture.

### Fine-tuning
1. Encoder-Decoder Model (e.g., T5)

   ```bash fine-tuning/run_train_t5.sh ```

2. Decoder-only Model (e.g., Llama)
   
    ```bash fine-tuning/run_train.sh ```

### Testing
1. Encoder-Decoder Model (e.g., T5)

    ```bash fine-tuning/run_test_t5.sh ```

2. Decoder-only Model (e.g., Llama)

    ```bash fine-tuning/run_test.sh ```

  
