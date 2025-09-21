# Tiny_CLIP-X

Here is a self-built Model [Only 0.58MB].

---

# Cognitive Distillation: Interpretable and Deployable Lightweight Vision-Language Models

![Project Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red?style=for-the-badge&logo=pytorch) ![GitHub stars](https://img.shields.io/github/stars/your-github-username/your-repository-name?style=for-the-badge&color=yellow)

---

### 🔑 Keywords
`Vision-Language Models` · `Knowledge Distillation` · `Interpretability` · `Edge AI` · `Lightweight Models` · `TinyEval` · `Trustworthy AI`

---

## 📖 Project Overview

This repository contains the **official implementation** of the paper:

> **Cognitive Distillation: Interpretable and Deployable Lightweight Vision-Language Models with TinyEval**

We introduce a novel **Cognitive Distillation** strategy that enables training **lightweight Vision-Language Models (VLMs)** with:
- 🧩 **Interpretability** — Student models inherit semantic attribution abilities from teachers.
- ⚡ **Efficiency** — Optimized for edge-device inference.
- 🔒 **Trustworthiness** — Designed for safe and transparent AI deployment.

---

## 🚀 Key Features

- **Phased Training Pipeline**
  Modularized into multiple stages (`train_stage1.py`, `train_stage2.py`, `train_stage2b.py`, `train_stage3.py`) for progressive distillation from CLIP → TinyCLIP.

- **Semantic Alignment Loss**
  Implemented in `semantic_alignment_loss.py`. Ensures semantic attribution consistency between teacher and student models.

- **Modular & Reusable Design**
  Teacher (`CLIP_teacher.py`), Student (`TinyCLIP_student.py`), and Loss (`semantic_alignment_loss.py`) are decoupled for flexibility and extensibility.

---

## 📂 File Structure
```bash
.

├── CLIP_teacher.py              # CLIP teacher model definition
├── README.md                    # This file
├── TinyCLIP_student.py          # TinyCLIP student model definition
├── semantic_alignment_loss.py   # Core semantic alignment loss
├── train_stage1.py              # Training script - Stage 1
├── train_stage2.py              # Training script - Stage 2
├── train_stage2b.py             # Training script - Stage 2b
├── train_stage3.py              # Training script - Stage 3
└── ...
```

### 🛠️ How to Use

#### 1. Environment Setup

- Ensure you have **Python 3.8+** and **PyTorch** installed in your environment.

#### 2. Dataset Preparation

- This project uses the **COCO** dataset. Please download the COCO 2017 `train2017` images and `annotations` files from the official website and organize them in the `datasets/` folder to match the paths in the `coco.py` script.

#### 3. Running the Training

- **Configure Training Parameters**:
  - Training parameters (e.g., learning rate, batch size, loss weights) are defined in YAML files within the `configs/` folder. Create these configuration files according to your experimental needs.
- **Execute Training**:
  - Follow the phased approach described in the paper by running the training scripts sequentially. For example, to run Stage 2b:
   ```bash
   python train_stage2b.py
   ```
- The script will automatically load the checkpoint from the previous stage and begin the cognitive distillation training.

---

### 🙏 Acknowledgments

We are grateful for the high-quality visual data provided by the COCO Dataset team and for the accessible pretrained models from the CLIP open-source community.

---

### 📌 Official Code Release

This repository is the official code release for:
**Cognitive Distillation: Interpretable and Deployable Lightweight Vision-Language Models with TinyEval**
---
