# Chain-of-Thought Circuits in Gemma-2-2B-IT

## Overview

This project investigates how **chain-of-thought (CoT) prompting** affects the internal addition-only arithmetic circuits of the **Gemma-2-2B-IT** language model. While CoT prompts are known to improve model performance, it is unclear whether they induce the model to use the same circuits as standard (Direct) prompts.  

Using **Circuit Tracer** and **intervention-based analysis**, we construct minimal circuits for addition tasks under both Direct and CoT prompts, then compare them through feature sets and weighted graphs. Our findings suggest that CoT and Direct circuits are largely distinct, indicating that CoT prompts engage different model components.

---

## Features

- Generation of **Direct** and **CoT addition prompts**.
- Extraction of **per-digit attribution graphs** using Circuit Tracer.
- **Feature selection and effect calculation** via activation patching.
- Construction of **minimal circuits** for Direct and CoT prompts.
- **Faithfulness evaluation** of circuits using Total Variation distance.
- Circuit comparison using **Jaccard and Weighted Jaccard similarity**.
- Baseline comparison across **subtraction, word addition, and unrelated numeric tasks**.

---

## Requirements

- Python >= 3.10  
- PyTorch  
- GPU 

---

## Structure

Scripts and files are organized as follows:
- finding_arithmetic_circuts.ipynb main notebook - with the important logic 
- `data/` – prompt sets and counterfactual prompts  
- `artifacts/` – low-weighted features and intermediate files from analysis  
- `scripts/` – additional scripts Used for comparison
- `results/` – figures, tables, and similarity metrics

---
