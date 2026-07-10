# BanglarMukh: A Physics-Aware Multilingual Multimodal Vision Language Benchmark for Evaluating Cultural and Traditional Grounding

[![Paper](https://img.shields.io/badge/Paper-NeurIPS/CVPR-red.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-BanglarMukh-blue.svg)](https://github.com/rasel1510/BanglarMukh)
[![Code Grade](https://img.shields.io/badge/Code--Grade-A*--Level-brightgreen.svg)](#)
[![OS](https://img.shields.io/badge/OS-Windows/Linux-orange.svg)](#)

---

## 🌟 Overview

**BanglarMukh** is a novel, physics-aware, multilingual, and multimodal vision-language benchmark designed to evaluate and enhance the cultural and traditional grounding of large Vision-Language Models (VLMs) on the heritage of Bangladesh. By integrating structural physical characteristics with deep semantic representations, **BanglarMukh** addresses the limitations of standard VLMs in recognizing regional variations, local attires, domestic wildlife, national history, and regional dialects.

### Key Contributions:
1. **Physics-Aware Visual Fusion**: Utilizes a custom `PhysicsFusedVisionTower` combining a `PhysicsFeatureExtractor` and a `PhysicsMotionEncoder` to inject physical spatial/structural priors into base visual representations (Qwen2-VL, PaliGemma) through cross-conscious fusion attention.
2. **Multilingual Baseline Evaluation**: Support for **6 primary languages** (Bengali, English, Chinese, French, Hindi, Urdu) for visual captioning and VQA reasoning tasks.
3. **Dialect Gradient Adaptation**: Evaluates model performance across **6 local dialects of Bangladesh** (Barishal, Chittagong, Noakhali, Rajshahi, Rangpur, Sylhet) using Gemini 1.5 Pro-driven translation pipelines and Parameter-Efficient Fine-Tuning (PEFT/LoRA).
4. **Comprehensive Cultural Grounding**: Evaluated across **15 distinct cultural and heritage classes** representing the traditional essence of Bangladesh.

---

## 📸 Methodology

The core architecture of BanglarMukh integrates physics-aware features (like texture gradients, edge contours, and structural shapes) with visual tokens, followed by deep cross-conscious fusion before being processed by the language model.

![Main Methodology](./Main%20Methodology.png)

---

## 📁 Dataset Structure

The dataset is organized hierarchically. It separates the raw image repository from the semantic annotations in 6 primary languages and 6 regional dialects across **15 distinct target classes**:

```
BanglarMukh/
└── Data/
    ├── images/                          # Raw image repositories organized by the 15 classes
    │   ├── attires/                     # Local attires (e.g., Sharee, Panjabi, Lungi)
    │   ├── crafts/                      # Handcrafts (e.g., Nakshi Kantha, Clay Pottery)
    │   ├── education institutions/      # Historical educational structures (e.g., Dhaka College)
    │   ├── festival/                    # Cultural and religious festivals (e.g., Pohela Boishakh)
    │   ├── fishes/                      # Indigenous fish species of Bangladesh (e.g., Ilish)
    │   ├── food/                        # Bengali traditional dishes (e.g., Panta Ilish, Biryani)
    │   ├── historical places/           # Archeological and historic sites (e.g., Lalbagh Fort)
    │   ├── movements/                   # Historic movements of Bangladesh (e.g., Language Movement)
    │   ├── national_achievements/       # Key national milestones and achievements
    │   ├── natural beauty/              # Landscapes, Sundarbans, and scenic reserves
    │   ├── personalities/               # Historic and notable figures of Bangladesh
    │   ├── rivers/                      # Major rivers of Bangladesh (e.g., Padma, Meghna)
    │   ├── sports/                      # Traditional and national sports (e.g., Kabaddi)
    │   ├── sweets/                      # Traditional Bengali sweets (e.g., Roshogolla, Chomchom)
    │   └── wildlife/                    # National wildlife (e.g., Royal Bengal Tiger, Doel)
    │
    ├── languages/                       # Ground-truth annotations in 6 primary languages
    │   ├── pure_bangla/
    │   │   ├── attires/
    │   │   │   └── annotations/
    │   │   │       ├── attires_captions.json
    │   │   │       └── attires_commonsense_reasoning.json
    │   │   └── ... (all 15 categories structured similarly)
    │   ├── english/
    │   ├── chinese/
    │   ├── french/
    │   ├── hindi/
    │   └── urdu/
    │
    └── Dialects/                        # Captions and VQA pairs adapted into 6 regional dialects
        ├── Barisal_dialect/
        │   ├── attires/
        │   │   └── annotations/
        │   │       ├── attires_captions.json
        │   │       └── attires_commonsense_reasoning.json
        │   └── ... (all 15 categories structured similarly)
        ├── Chittagong_dialect/
        ├── Noakhali_dialect/
        ├── Rajshahi_dialect/
        ├── Rangpur_dialect/
        └── Sylheti_dialect/
```

---

## 🛠️ Codebase Organization & Execution

The codebase contains modular code for fine-tuning the vision-language model, running zero-shot/few-shot evaluations, and executing dialect translation pipelines.

### 1. Physics-Fused VLM Training & Evaluation
To run the main physics-fused training loop with LoRA adapters and subsequent evaluation:
```bash
python "models/Main Method.py" \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --vlm_type "qwen" \
    --epochs 3 \
    --batch_size 2 \
    --use_lora True \
    --device "cuda"
```

### 2. Standalone Dialect Pipelines
Each local dialect directory contains an A*-level standalone evaluation script. Run them using:
```bash
# Evaluate on Barishal Dialect
python Dialects/Barishal/dialect_barishal.py --prompt_mode few_shot

# Evaluate on Chittagong Dialect
python Dialects/Chittagong/dialect_chittagong.py --prompt_mode few_shot

# Evaluate on Noakhali Dialect
python Dialects/Noakhali/dialect_noakhali.py --prompt_mode few_shot

# Evaluate on Rajshahi Dialect
python Dialects/Rajshahi/dialect_rajshahi.py --prompt_mode few_shot

# Evaluate on Rangpur Dialect
python Dialects/Rangpur/dialect_rangpur.py --prompt_mode few_shot

# Evaluate on Sylhet Dialect
python Dialects/Sylthet/dialect_sylthet.py --prompt_mode few_shot
```

**Common Pipeline Arguments:**
* `--prompt_mode`: Choose between `zero_shot`, `few_shot`, or `cot` (Chain-of-Thought).
* `--dry_run`: Execute in dry-run mode (simulating API response rates without loading full models/keys).
* `--disable_cache`: Disable local JSON translation caching.

---

## 📊 Benchmark Results

| Model | Categories | Zero-Shot Caption | VQA | Few-Shot Caption | VQA | CoT VQA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| | | **B-F1** \| **LLM** | **Acc(%)** | **B-F1** \| **LLM** | **Acc(%)** | **Acc(%)** |
| **BanglarMukh** | Attires | 0.56 \| 0.40 | 45.0% | 0.58 \| 0.42 | 46.0% | 52.0% |

---

## 📝 Citation

If you use this benchmark or codebase in your research, please cite our paper:

```bibtex
@inproceedings{banglarmukh2026,
  title={BanglarMukh: A Physics-Aware Multilingual Multimodal Vision Language Benchmark for Evaluating Cultural and Traditional Grounding},
  author={BanglarMukh Research Group},
  booktitle={Proceedings of the International Conference on Multimodal Vision-Language Benchmarks},
  year={2026}
}
```
