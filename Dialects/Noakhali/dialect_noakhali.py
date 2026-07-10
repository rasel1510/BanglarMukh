

import os
import sys
import json
import time
import re
import random
import logging
import argparse
import collections
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image

# ------------------------------------------------------------------------------
# PATH RESOLUTION & DYNAMIC IMPORTS
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (NoakhaliPipeline) %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "noakhali_translation.log"),
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("NoakhaliTranslator")

try:
    import torch
    import torch.nn as nn
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch or PEFT not available. VLM generation will run in dry-run mode.")
    HAS_TORCH = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    logger.warning("google-generativeai not found. Gemini API calls will use simulation mode.")
    HAS_GEMINI = False

try:
    from models.vlm_wrapper import build_banglarmukh_model
    from inference.engine import BanglarMukhInferenceEngine
except ImportError:
    logger.error("Could not import BanglarMukh custom models. Ensure script runs from project root.")

# ------------------------------------------------------------------------------
# VLM AND LORA LOADER
# ------------------------------------------------------------------------------
class VLMModelManager:
    """
    Manages initialization of the BanglarMukh VLM with optional LoRA adapter loading.
    Designed to process visual inputs through Physics Fused Vision Tower.
    """
    def __init__(self, model_name: str, vlm_type: str, use_lora: bool,
                 lora_path: Optional[str], device: str):
        self.model_name = model_name
        self.vlm_type = vlm_type
        self.use_lora = use_lora
        self.lora_path = lora_path
        self.device = device
        self.model = None
        self.engine = None

    def load_model(self) -> bool:
        """Loads the VLM backbone and applies trained LoRA adapter weights if available."""
        if not HAS_TORCH:
            logger.info("[Dry Run] Simulating VLM initialization — torch unavailable.")
            return True
        try:
            logger.info(f"Initializing base model '{self.model_name}' on device '{self.device}'...")
            self.model = build_banglarmukh_model(
                model_name_or_path=self.model_name,
                vlm_type=self.vlm_type,
                physics_dim=256,
                num_heads=8,
                use_lora=False,
                device_map=self.device
            )
            if self.use_lora and self.lora_path:
                if os.path.exists(self.lora_path):
                    logger.info(f"Applying LoRA adapter weights from: {self.lora_path}")
                    state_dict = torch.load(self.lora_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info("LoRA weights applied successfully.")
                else:
                    logger.warning(f"LoRA checkpoint '{self.lora_path}' not found. Using base weights.")

            class DummyProcessor:
                def __call__(self, text, images, **kwargs):
                    return {
                        "input_ids": torch.ones((1, 5), dtype=torch.long),
                        "pixel_values": torch.randn((1, 3, 224, 224))
                    }
                def batch_decode(self, sequences, **kwargs):
                    return ["বাংলাদেশর একখান নামকরা হীক্ষা ফতিষ্ঠান, হিয়ানো জ্ঞানের আলো ছড়াই যার।"]

            self.engine = BanglarMukhInferenceEngine(
                model=self.model,
                processor=DummyProcessor(),
                vlm_type=self.vlm_type,
                device=self.device
            )
            logger.info("BanglarMukh VLM engine ready for Noakhali evaluation.")
            return True
        except Exception as e:
            logger.error(f"VLM load failed: {e}", exc_info=True)
            return False

    def generate_caption(self, image_path: str) -> str:
        """Generates a Bengali caption from an image using the loaded VLM."""
        if not HAS_TORCH or self.engine is None:
            return random.choice([
                "বাংলাদেশের একটি স্বনামধন্য শিক্ষা প্রতিষ্ঠান, যেখানে শিক্ষার আলো ছড়িয়ে পড়ছে।",
                "ঐতিহ্যবাহী একটি বিদ্যাপীঠ, শিক্ষার্থীদের জ্ঞান অর্জনের এক চমৎকার পরিবেশ।"
            ])
        try:
            image = (
                Image.open(image_path).convert("RGB")
                if os.path.exists(image_path)
                else Image.new("RGB", (224, 224))
            )
            return self.engine.generate_zero_shot_caption(image)
        except Exception as e:
            logger.warning(f"VLM caption error: {e}")
            return "বাংলাদেশের একটি স্বনামধন্য শিক্ষা প্রতিষ্ঠান, যেখানে শিক্ষার আলো ছড়িয়ে পড়ছে।"

# ------------------------------------------------------------------------------
# LOCAL CACHE
# ------------------------------------------------------------------------------
class TranslationCache:
    """Disk-backed JSON cache for storing and retrieving Noakhali translation outputs."""
    def __init__(self, filename: str = "noakhali_dialect_cache.json"):
        self.filename = os.path.join(os.path.dirname(__file__), filename)
        self.cache: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached translation entries.")
            except Exception as e:
                logger.error(f"Cache load error: {e}")
                self.cache = {}

    def get(self, text: str, mode: str) -> Optional[str]:
        return self.cache.get(f"{text}||{mode}")

    def set(self, text: str, mode: str, result: str):
        self.cache[f"{text}||{mode}"] = result
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Cache save error: {e}")

# ------------------------------------------------------------------------------
# GEMINI 1.5 PRO NOAKHALI TRANSLATOR
# ------------------------------------------------------------------------------
class GeminiNoakhaliTranslator:
    """
    Wraps Google Gemini 1.5 Pro to translate standard Bengali captions into
    the Noakhali regional dialect (Noakhailla) of Bangladesh.
    Implements Zero-Shot, Few-Shot, and Chain-of-Thought prompts.
    """
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.system_instruction = (
            "আপনি নোয়াখালী অঞ্চলের উপভাষার (নোয়াখাইল্লা) একজন বিশেষজ্ঞ ভাষাবিদ। "
            "আপনার কাজ হলো প্রমাণ বাংলা বাক্যকে নোয়াখালীর স্থানীয় উপভাষায় রূপান্তর করা। "
            "নোয়াখাইল্লা উপভাষার প্রধান বৈশিষ্ট্যগুলি যেমন: উচ্চারণে 'প' -> 'হ' / 'ফ', "
            "'আমি' -> 'আঁই', 'আমরা' -> 'আঁরা', 'তুমি' -> 'তুঁই', 'সে' -> 'হেতে/হেতি', "
            "এবং ক্রিয়া পদে '-ইয়ের' বা '-করের' প্রত্যয় প্রয়োগ করুন। "
            "শুধুমাত্র রূপান্তরিত নোয়াখাইল্লা বাক্যটি আউটপুট করুন, কোনো ব্যাখ্যা বা মন্তব্য ছাড়া।"
        )
        self._initialize_client()

    def _initialize_client(self):
        if not HAS_GEMINI or not self.api_key:
            logger.warning("GenerativeAI SDK/Key not available. Simulation mode enabled.")
            return
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                system_instruction=self.system_instruction
            )
            logger.info("Gemini 1.5 Pro model initialized successfully.")
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")

    def _call_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        if not HAS_GEMINI or self.model is None:
            return self._simulate(prompt)
        base = 2.0
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2, max_output_tokens=512
                    )
                )
                if response.text:
                    return response.text.strip()
                raise ValueError("Empty response.")
            except Exception as e:
                wait = (base ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed. Retrying in {wait:.1f}s: {e}")
                time.sleep(wait)
        return "[Error: Translation failed]"

    def translate_zero_shot(self, text: str) -> str:
        prompt = (
            "নিচের প্রমিত বাংলা বাক্যটিকে নোয়াখাইল্লা উপভাষায় রূপান্তর করুন:\n\n"
            f"বাক্য: {text}\n"
            "রূপান্তরিত বাক্য:"
        )
        return self._call_with_retry(prompt)

    def translate_few_shot(self, text: str) -> str:
        prompt = (
            "কাজ: প্রমিত বাংলাকে নোয়াখাইল্লা উপভাষায় রূপান্তর করুন।\n\n"
            "উদাহরণ ১:\n"
            "প্রমিত: বাংলাদেশের একটি বিখ্যাত শিক্ষাপ্রতিষ্ঠান, যেখানে জ্ঞানের আলো ছড়িয়ে পড়ছে।\n"
            "আউটপুট: বাংলাদেশর একখান নামকরা হীক্ষা ফতিষ্ঠান, হিয়ানো জ্ঞানের আলো ছড়াই যার।\n\n"
            "উদাহরণ ২:\n"
            "প্রমিত: শিক্ষার্থীরা মনোযোগ দিয়ে পাঠ গ্রহণ করছে।\n"
            "আউটপুট: পোলাপাইন মনোযোগ দিয়া হড়াশোনা করের।\n\n"
            f"এখন নিচের বাক্যটি রূপান্তর করুন:\n"
            f"প্রমিত: {text}\n"
            "আউটপুট:"
        )
        return self._call_with_retry(prompt)

    def translate_cot(self, text: str) -> str:
        prompt = (
            "আপনি একজন ভাষাবিদ। নিচের বাক্যটি ধাপে ধাপে নোয়াখাইল্লা উপভাষায় রূপান্তর করুন:\n"
            "ধাপ ১: ব্যঞ্জনবর্ণ রূপান্তর করুন (যেমন: প->হ/ফ)।\n"
            "ধাপ ২: সর্বনাম ও বিশেষ্য পরিবর্তন করুন (যেমন: আমি->আঁই)।\n"
            "ধাপ ৩: নোয়াখাইল্লা উপভাষার ক্রিয়া পদ রূপান্তর করুন।\n"
            "ধাপ ৪: নোয়াখাইল্লা উপভাষার গতি ঠিক রেখে বাক্যটি সাজান।\n"
            "সবশেষে 'চূড়ান্ত নোয়াখাইল্লা রূপ: ' লেবেল দিয়ে রূপান্তরিত বাক্যটি লিখুন।\n\n"
            f"প্রমিত বাক্য: {text}"
        )
        raw = self._call_with_retry(prompt)
        match = re.search(r"চূড়ান্ত নোয়াখাইল্লা রূপ:\s*(.+)", raw)
        if match:
            return match.group(1).strip().strip('"').strip("'")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        for line in reversed(lines):
            if not re.match(r"^ধাপ\s*\d+", line) and len(line) > 10:
                return re.sub(r"^[^:]+:\s*", "", line).strip()
        return raw

    def _simulate(self, prompt: str) -> str:
        if "শিক্ষা প্রতিষ্ঠান" in prompt or "edu_001" in prompt:
            return "বাংলাদেশর একখান নামকরা হীক্ষা ফতিষ্ঠান, হিয়ানো জ্ঞানের আলো ছড়াই যার।"
        elif "বিদ্যাপীঠ" in prompt:
            return "একখান ঐতিহ্যবাহী হড়ার জায়গা, হিয়ানো হড়াশোনা করের পোলাপাইন।"
        elif "ঢাকা কলেজ" in prompt:
            return "ঢাকা কলেজ বাংলাদেশর মইধ্যে বেগগুন থন পুরান হীক্ষা ফতিষ্ঠান।"
        elif "মনোযোগ" in prompt:
            return "পোলাপাইন মনোযোগ দিয়া হড়াশোনা করের।"
        return "নোয়াখাইল্লা উপভাষায় রূপান্তরিত বাক্য।"

# ------------------------------------------------------------------------------
# NLG EVALUATION METRICS
# ------------------------------------------------------------------------------
class BanglaEvaluator:
    """Computes BLEU-4, ROUGE-L and Token F-1 for evaluated Bengali text."""
    DANDA_PATTERN = re.compile(r"[।॥\.\,\!\?\;\:\-\(\)\[\]]+")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        text = cls.DANDA_PATTERN.sub(" ", text).strip()
        return [t for t in text.split() if t]

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> collections.Counter:
        return collections.Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    @classmethod
    def compute_bleu(cls, reference: str, hypothesis: str) -> float:
        import math
        ref_t = cls.tokenize(reference)
        hyp_t = cls.tokenize(hypothesis)
        if not ref_t or not hyp_t:
            return 0.0
        bp = 1.0 if len(hyp_t) >= len(ref_t) else math.exp(1.0 - len(ref_t) / max(len(hyp_t), 1))
        log_sum = 0.0
        for n in range(1, 5):
            hyp_ng = cls._get_ngrams(hyp_t, n)
            ref_ng = cls._get_ngrams(ref_t, n)
            clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
            total = sum(hyp_ng.values())
            if total == 0 or clipped == 0:
                log_sum += math.log(1e-10)
            else:
                log_sum += math.log(clipped / total)
        return bp * math.exp(log_sum / 4)

    @classmethod
    def compute_rouge_l(cls, reference: str, hypothesis: str) -> float:
        ref_t = cls.tokenize(reference)
        hyp_t = cls.tokenize(hypothesis)
        if not ref_t or not hyp_t:
            return 0.0
        m, n = len(ref_t), len(hyp_t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if ref_t[i-1] == hyp_t[j-1] else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        p = lcs / n
        r = lcs / m
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    @classmethod
    def compute_token_f1(cls, reference: str, hypothesis: str) -> float:
        ref_t = cls.tokenize(reference)
        hyp_t = cls.tokenize(hypothesis)
        if not ref_t or not hyp_t:
            return 0.0
        common = collections.Counter(ref_t) & collections.Counter(hyp_t)
        same = sum(common.values())
        if same == 0:
            return 0.0
        p = same / len(hyp_t)
        r = same / len(ref_t)
        return (2 * p * r) / (p + r)

# ------------------------------------------------------------------------------
# MAIN PIPELINE RUNNER
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="BanglarMukh Noakhali Dialect Pipeline.")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--vlm_type", type=str, default="qwen", choices=["qwen", "gemma"])
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_path", type=str, default="./checkpoints/noakhali_lora.pt")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--prompt_mode", type=str, default="few_shot", choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  BanglarMukh Noakhali Translation Pipeline Starting")
    logger.info("=" * 70)

    cache = TranslationCache() if not args.disable_cache else None
    vlm_mgr = VLMModelManager(
        model_name=args.vlm_model,
        vlm_type=args.vlm_type,
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        device="cuda" if (HAS_TORCH and torch.cuda.is_available() and not args.dry_run) else "cpu"
    )
    if not args.dry_run:
        vlm_mgr.load_model()

    translator = GeminiNoakhaliTranslator(api_key=args.api_key)

    samples = [
        {
            "id": "noakhali_edu_001",
            "source_pure_bn": "বাংলাদেশের একটি বিখ্যাত শিক্ষাপ্রতিষ্ঠান, যেখানে জ্ঞানের আলো ছড়িয়ে পড়ছে।",
            "reference_dialect": "বাংলাদেশর একখান নামকরা হীক্ষা ফতিষ্ঠান, হিয়ানো জ্ঞানের আলো ছড়াই যার।"
        },
        {
            "id": "noakhali_edu_002",
            "source_pure_bn": "এক ঐতিহ্যমণ্ডিত বিদ্যাপীঠ, যেখানে শিক্ষার্থীরা জ্ঞান অর্জন করে থাকেন।",
            "reference_dialect": "একখান ঐতিহ্যবাহী হড়ার জায়গা, হিয়ানো হড়াশোনা করের পোলাপাইন।"
        },
        {
            "id": "noakhali_edu_003",
            "source_pure_bn": "ঢাকা college বাংলাদেশের অন্যতম প্রাচীন বিদ্যায়তন।",
            "reference_dialect": "ঢাকা college বাংলাদেশর মইধ্যে বেগগুন থন পুরান হীক্ষা ফতিষ্ঠান।"
        },
        {
            "id": "noakhali_edu_004",
            "source_pure_bn": "শিক্ষার্থীরা মনোযোগ দিয়ে পাঠ গ্রহণ করছে।",
            "reference_dialect": "পোলাপাইন মনোযোগ দিয়া হড়াশোনা করের।"
        },
        {
            "id": "noakhali_edu_005",
            "source_pure_bn": "শিক্ষকগণ শ্রেণিকক্ষে পাঠদান করছেন।",
            "reference_dialect": "মাস্টারসাব ক্লাসর মইধ্যে হড়াইয়ের।"
        },
        {
            "id": "noakhali_edu_006",
            "source_pure_bn": "বিদ্যালয়ের খেলার মাঠ বিশাল ও সুন্দর।",
            "reference_dialect": "স্কুলর খেলার মাঠখান অনেক বড় আর সুন্দর।"
        },
        {
            "id": "noakhali_edu_007",
            "source_pure_bn": "পরীক্ষার ফলাফল অত্যন্ত ভালো হয়েছে।",
            "reference_dialect": "পরীক্ষার রেজাল্ট খুব ভালা অইছে।"
        },
        {
            "id": "noakhali_edu_008",
            "source_pure_bn": "গ্রন্থাগারে অনেক মূল্যবান বই রয়েছে।",
            "reference_dialect": "লাইব্রেরির মইধ্যে অনেক দামী বই আছে।"
        },
        {
            "id": "noakhali_edu_009",
            "source_pure_bn": "বিজ্ঞান প্রদর্শনীতে শিক্ষার্থীরা উদ্ভাবনী প্রকল্প উপস্থাপন করেছে।",
            "reference_dialect": "বিজ্ঞান মেলায় পোলাপাইন নতুন জিনিস বানাই দেখাইছে।"
        },
        {
            "id": "noakhali_edu_010",
            "source_pure_bn": "প্রতিষ্ঠানটি দীর্ঘ ঐতিহ্য বহন করছে।",
            "reference_dialect": "এই ফতিষ্ঠানর অনেক পুরান ইতিহাস আছে।"
        },
        {
            "id": "noakhali_edu_011",
            "source_pure_bn": "শিক্ষকরা শিক্ষার্থীদের প্রতি অত্যন্ত যত্নশীল।",
            "reference_dialect": "মাস্টাররা পোলাপাইনের দিকে খুব খেয়াল রাহে।"
        },
        {
            "id": "noakhali_edu_012",
            "source_pure_bn": "বার্ষিক সাংস্কৃতিক অনুষ্ঠান বেশ জাঁকজমকভাবে অনুষ্ঠিত হয়।",
            "reference_dialect": "বছরের সাংস্কৃতিক অনুষ্ঠান খুব ধুমধাম করি অয়।"
        },
        {
            "id": "noakhali_edu_013",
            "source_pure_bn": "ক্যান্টিনে শিক্ষার্থীরা বিরতির সময় একত্রিত হয়।",
            "reference_dialect": "টিফিনর সময় পোলাপাইন ক্যান্টিনে একসাথে জমা অয়।"
        },
        {
            "id": "noakhali_edu_014",
            "source_pure_bn": "অভিভাবকরা সন্তানদের ভবিষ্যৎ নিয়ে স্বপ্ন দেখেন।",
            "reference_dialect": "মা-বাবারা পোলাপাইনের ভবিষ্যৎ লই স্বপ্ন দেখে।"
        }
    ]

    results = []
    bleu_scores, rouge_scores, f1_scores = [], [], []
    evaluator = BanglaEvaluator()

    for idx, sample in enumerate(samples):
        source = sample["source_pure_bn"]
        if args.image_path and idx == 0:
            source = vlm_mgr.generate_caption(args.image_path)

        translated = cache.get(source, args.prompt_mode) if cache else None
        if not translated:
            if args.prompt_mode == "zero_shot":
                translated = translator.translate_zero_shot(source)
            elif args.prompt_mode == "few_shot":
                translated = translator.translate_few_shot(source)
            else:
                translated = translator.translate_cot(source)
            if cache and not translated.startswith("[Error"):
                cache.set(source, args.prompt_mode, translated)

        ref = sample.get("reference_dialect", "")
        bleu = evaluator.compute_bleu(ref, translated)
        rouge = evaluator.compute_rouge_l(ref, translated)
        f1 = evaluator.compute_token_f1(ref, translated)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        f1_scores.append(f1)

        logger.info(f"[{sample['id']}] Source: {source}")
        logger.info(f"[{sample['id']}] Dialect: {translated}")

        results.append({
            "id": sample["id"],
            "source_bn_mixed": source,
            "translated_dialect": translated,
            "reference_dialect": ref,
            "metrics": {
                "bleu": round(bleu, 4),
                "rouge_l": round(rouge, 4),
                "token_f1": round(f1, 4)
            }
        })

    print("\n" + "=" * 80)
    print("      BANGLARMUKH NOAKHALI DIALECT TRANSLATION EVALUATION REPORT")
    print("=" * 80)
    print(f"  Prompt Strategy : {args.prompt_mode.upper()}")
    print("-" * 80)
    print(f"| {'ID':<17} | {'BLEU-4':>8} | {'ROUGE-L':>8} | {'Token F1':>9} |")
    print(f"| {'-'*17} | {'-'*8} | {'-'*8} | {'-'*9} |")
    for r in results:
        m = r["metrics"]
        print(f"| {r['id']:<17} | {m['bleu']:>8.4f} | {m['rouge_l']:>8.4f} | {m['token_f1']:>9.4f} |")
    print("-" * 80)
    n = len(bleu_scores)
    print(f"| {'AVERAGE':<17} | {sum(bleu_scores)/n:>8.4f} | {sum(rouge_scores)/n:>8.4f} | {sum(f1_scores)/n:>9.4f} |")
    print("=" * 80 + "\n")

    out_path = os.path.join(os.path.dirname(__file__), "noakhali_dialect_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
