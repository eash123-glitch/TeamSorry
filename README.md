# 🧠 TeamSorry — Grounded Video Question Answering

The project focuses on building a unified model that performs both **temporal localization** and **question answering**, while leveraging **large language models (LLMs)** for efficient and scalable data generation.

---

## 🚀 Overview

We address the challenge of **Video–Language Grounded QA**, which involves:
- Understanding long **egocentric (first-person)** videos,  
- **Grounding** queries to the correct temporal segments, and  
- Generating accurate **open-ended** or **close-ended** answers to natural language questions.

Our system jointly handles:
1. **OpenQA** — Free-form answer generation  
2. **CloseQA** — Multiple-choice question answering  
3. **VLG (Video–Language Grounding)** — Temporal localization of question-relevant video segments

---
## Video Features

The precomputed video features of the Edo4D dataset can be found at https://ego4d-data.org/docs/data/features/  and can be accessed throught their commandline interface, the model training expects that the features be in an .hdf5 file and be placed in the data directory.
 

## 📂 Repository Structure
```
.
|-- data/            # Narrations and generated question–answer pairs
|-- eval.py          # Evaluation script for QA datasets
|-- eval_nlq.py      # Evaluation script for NLQ benchmarks
|-- model/           # Model architecture, dataset loaders, and training modules
|-- requirements.txt # Python dependencies
|-- train_run.py     # Entry point for training and evaluation
`-- utils/           # Data generation utilities

---
```
## ⚙️ Environment Setup

**Recommended setup:**
- Ubuntu 20.04 or newer  
- CUDA 12.2+  
- At least one GPU with ≥12 GB VRAM  

```bash
# Create and activate environment
conda create -n teamsorry python=3.9 -y
conda activate teamsorr

# Install dependencies
pip install -r requirements.txt
    
```

## 🧩 Training
Training can be customized for various dataset combinations and task objectives.

```bash
# Train unified model on QA-Ego4D
python train_run.py

```
All experiments were performed on 6 lab GPUs (12 GB each), completing in approximately 7 hours.

## 🧪 Evaluation
```bash
## Evaluate model on QA-Ego4D
python eval.py

```
💡 Data Generation
Our data generation pipeline employs LLMs to automatically create QA pairs from egocentric video narrations.

🔹 Generate OpenQA Data
```bash
cd utils/generate_open_qa

# Run generation on two GPUs in parallel
CUDA_VISIBLE_DEVICES=0 python generate.py -start 0 -end 5000
CUDA_VISIBLE_DEVICES=1 python generate.py -start 5000 -end 11000

# Merge and normalize generated results
python merge.py
```
(Adjust the number of GPUs based on availability.)

🔹 Generate CloseQA Data
```bash
cd utils/generate_close_qa
python generate.py
```

Note:

The Hugging Face API tokens are required for both generate_open_qa and generate_close_qa scripts.And also the ego4d website.
The variable token is defined inside these files but left empty — please insert your own valid token there.
This is because GitHub does not allow sensitive credentials such as access tokens to be pushed publicly.

Without a valid token, LLM-based data generation will not execute successfully.

## 🧾 Dataset Access
The Ego4D narrations used for generating QA pairs are already included in the data/ directory.
However, the raw video data is not part of this repository due to its large size and licensing restrictions.

To access the full Ego4D video dataset:

Visit https://ego4d-data.org/

The official Ego4D website provides better visualization tools, annotations, and download options for the complete dataset.

📊 Results Overview
| Model                 | Data                         | Task                     | Sentence Sim. | ROUGE | METEOR | Accuracy      |
|-----------------------|------------------------------|---------------------------|---------------|-------|--------|---------------|
| Unified               | QA-Ego4D + EgoTimeQA         | OpenQA + CloseQA + VLG    | 57.7          | 30.2  | 21.2   | 48.7 ± 0.4    |


The inclusion of EgoTimeQA, an automatically generated large-scale dataset, significantly enhances both QA accuracy and grounding capability.


