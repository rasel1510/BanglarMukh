# BanglarMukh: A Physics-Aware Multilingual Multimodal Vision Language Benchmark for Evaluating Cultural and Traditional Grounding


[![Dataset](https://img.shields.io/badge/Dataset-BanglarMukh-blue.svg)](https://github.com/rasel1510/BanglarMukh)
[![OS](https://img.shields.io/badge/OS-Windows/Linux-orange.svg)](#)

---

## Abstract

Bangla culture and tradition is distinctly utilizing local dialects, heritage, tradition, and regular visual practices, yet it is greatly absent from multimodal assessment. We intro- duce **BanglarMukh**, a culturally and traditionally rooted bench- mark for evaluating Large Vision Language Models (LVLMs) on Bangladeshi tradition and culture over linguistic diver- sity. BanglarMukh encompasses 1,448 expert annotated images traversing 15 domains and reinforces both captioning and Visual Question Answering (VQA). Each category is expanded into six standard languages and six native dialects, generating 66,608 evaluation artifacts. Experiments over various physics aware LVLMs exhibit that assessment on base Bangla alone consid- erably inflates real world result, accuracy and reasoning quality reduce clearly over dialect alters, with the superior reduction in free form captioning. Languages such as Hindi, Urdu, Chinese conserve some traditional cues but remain weaker in framed reasoning. We releases the dataset, prompts, and resulting scripts to ensure reproducible, culturally conscious benchmarking.


## 📸 Methodology

The core architecture of BanglarMukh integrates physics-aware features (like texture gradients, edge contours, and structural shapes) with visual tokens, followed by deep cross-conscious fusion before being processed by the language model.

![Main Methodology](./Main%20Methodology.png)

---

## 📁 Dataset Structure

The dataset is organized hierarchically. It separates the raw image repository from the semantic annotations in 6 primary languages and 6 regional dialects across **15 distinct target classes**:

```
BanglarMukh/
└── Data/
    ├── images/
    │   ├── attires/
    │   ├── crafts/
    │   ├── education institutions/
    │   ├── festival/
    │   ├── fishes/
    │   ├── food/
    │   ├── historical places/
    │   ├── movements/
    │   ├── national_achievements/
    │   ├── natural beauty/
    │   ├── personalities/
    │   ├── rivers/
    │   ├── sports/
    │   ├── sweets/
    │   └── wildlife/
    │
    ├── languages/
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
    └── Dialects/
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
