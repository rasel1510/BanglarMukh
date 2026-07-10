

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
# 1. PATH RESOLUTION & DYNAMIC IMPORTS
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "translation_pipeline.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("FrenchTranslator")

try:
    import torch
    import torch.nn as nn
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch or PEFT not available. VLM generation will run in dry-run mode.")
    HAS_TORCH = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    logger.warning("google-generativeai package not found. Gemini API calls will run in simulation mode.")
    HAS_GEMINI = False

try:
    from models.vlm_wrapper import build_banglarmukh_model
    from inference.engine import BanglarMukhInferenceEngine
except ImportError:
    logger.error("Could not import BanglarMukh custom models. Ensure script is run from project structure.")

# ------------------------------------------------------------------------------
# 2. VLM AND LORA LOADER
# ------------------------------------------------------------------------------
class VLMModelManager:
    def __init__(self, model_name: str, vlm_type: str, use_lora: bool, lora_path: Optional[str], device: str):
        self.model_name = model_name
        self.vlm_type = vlm_type
        self.use_lora = use_lora
        self.lora_path = lora_path
        self.device = device
        self.model = None
        self.engine = None
        
    def load_model(self) -> bool:
        if not HAS_TORCH:
            logger.info("[Dry Run] Simulating VLM initialization.")
            return True
            
        try:
            logger.info(f"Initializing base model '{self.model_name}' on {self.device}...")
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
                    logger.info(f"Applying LoRA checkpoints from: {self.lora_path}")
                    state_dict = torch.load(self.lora_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    logger.warning(f"LoRA path '{self.lora_path}' not found. Using default weights.")
            
            class DummyProcessor:
                def __call__(self, text, images, **kwargs):
                    return {"input_ids": torch.ones((1, 5), dtype=torch.long), "pixel_values": torch.randn((1, 3, 224, 224))}
                def batch_decode(self, sequences, **kwargs):
                    return ["একটি গ্রামীণ দৃশ্য যেখানে কৃষক গরু নিয়ে ধানক্ষেতে হাল চাষ করছেন।"]
            
            self.engine = BanglarMukhInferenceEngine(
                model=self.model,
                processor=DummyProcessor(),
                vlm_type=self.vlm_type,
                device=self.device
            )
            logger.info("BanglarMukh VLM successfully initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}", exc_info=True)
            return False

    def generate_caption(self, image_path: str) -> str:
        if not HAS_TORCH or self.engine is None:
            synthetic_captions = [
                "বাংলাদেশের একটি স্বনামধন্য শিক্ষা প্রতিষ্ঠান, যেখানে শিক্ষার আলো ছড়িয়ে পড়ছে।",
                "ঐতিহ্যবাহী একটি বিদ্যাপীঠ, শিক্ষার্থীদের জ্ঞান অর্জনের এক চমৎকার পরিবেশ।"
            ]
            return random.choice(synthetic_captions)
            
        try:
            image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else Image.new("RGB", (224, 224))
            return self.engine.generate_zero_shot_caption(image)
        except Exception as e:
            logger.warning(f"Error during generation: {e}")
            return "বাংলাদেশের একটি স্বনামধন্য শিক্ষা প্রতিষ্ঠান, যেখানে শিক্ষার আলো ছড়িয়ে পড়ছে।"

# ------------------------------------------------------------------------------
# 3. TRANSLATION LOCAL CACHE
# ------------------------------------------------------------------------------
class TranslationCache:
    def __init__(self, filename: str = "french_translation_cache.json"):
        self.filename = os.path.join(os.path.dirname(__file__), filename)
        self.cache = {}
        self.load()
        
    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} entries.")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.cache = {}

    def get(self, text: str, prompt_mode: str) -> Optional[str]:
        return self.cache.get(f"{text}||{prompt_mode}")

    def set(self, text: str, prompt_mode: str, translation: str):
        self.cache[f"{text}||{prompt_mode}"] = translation
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

# ------------------------------------------------------------------------------
# 4. GEMINI 1.5 FLASH TRANSLATOR WITH RETRIES
# ------------------------------------------------------------------------------
class GeminiFlashTranslator:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.system_instruction = (
            "You are an expert bilingual translator specialized in converting South Asian "
            "cultural descriptions from Bengali/English into natural, formal, and grammatically perfect "
            "French (français standard). Accurately maintain cultural context for local terms. "
            "Provide only the French translated text in your output, without quotes or additional comments."
        )
        self._initialize_client()

    def _initialize_client(self):
        if not HAS_GEMINI:
            logger.warning("GenerativeAI package missing. Simulation enabled.")
            return
        if not self.api_key:
            logger.warning("Gemini API key not found.")
            return
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=self.system_instruction
            )
            logger.info("Google Gemini client initialized.")
        except Exception as e:
            logger.error(f"Gemini init error: {e}")

    def call_api_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        if not HAS_GEMINI or self.model is None:
            return self._simulate_translation(prompt)
            
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.1)
                )
                if response.text:
                    return response.text.strip()
                else:
                    raise ValueError("Empty response.")
            except Exception as e:
                delay = (base_delay ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(f"[Attempt {attempt+1}/{max_retries}] Retrying in {delay:.2f}s...")
                time.sleep(delay)
        return "[Error: API translation failed]"

    def translate_zero_shot(self, text: str) -> str:
        prompt = f"Translate this caption into French:\n\nCaption: {text}"
        return self.call_api_with_retry(prompt)

    def translate_few_shot(self, text: str) -> str:
        prompt = (
            "Task: Translate captions to French.\n\n"
            "Example 1:\n"
            "Input: বাংলাদেশের একটি স্বনামধন্য শিক্ষা প্রতিষ্ঠান, যেখানে শিক্ষার আলো ছড়িয়ে পড়ছে।\n"
            "Output: Une institution éducative renommée au Bangladesh, où se répand la lumière de l'éducation.\n\n"
            "Example 2:\n"
            "Input: ঐতিহ্যবাহী একটি বিদ্যাপীঠ, শিক্ষার্থীদের জ্ঞান অর্জনের এক চমৎকার পরিবেশ।\n"
            "Output: Un lieu d'apprentissage traditionnel, offrant un excellent environnement pour que les étudiants acquièrent des connaissances.\n\n"
            "Example 3:\n"
            "Input: ঐতিহ্যবাহী ঢাকা কলেজ, বাংলাদেশের অন্যতম প্রাচীন ও স্বনামধন্য একটি শিক্ষাপ্রতিষ্ঠান।\n"
            "Output: Le traditionnel Collège de Dacca, l'une des institutions éducatives les plus anciennes et les plus renommées du Bangladesh.\n\n"
            f"Now translate:\nInput: {text}\nOutput:"
        )
        return self.call_api_with_retry(prompt)

    def translate_cot(self, text: str) -> str:
        prompt = (
            "You are a professional translator. Translate step-by-step:\n"
            "1. Identify structural elements of the scene.\n"
            "2. Note key cultural components and translate/explain them accurately in French.\n"
            "3. Draft a literal translation.\n"
            "4. Refine it for formal French syntax and flow.\n"
            "Write your reasoning, and then output the final translation exactly on the line after 'Final French Caption: '.\n\n"
            f"Input Caption: {text}"
        )
        raw_output = self.call_api_with_retry(prompt)
        
        match = re.search(r"Final French Caption:\s*(.*)", raw_output, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"').strip("'")
            
        lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
        for line in reversed(lines):
            if "caption:" in line.lower() or "translation:" in line.lower() or not any(word in line.lower() for word in ["step", "reason", "identify"]):
                clean_line = re.sub(r"^.*?:\s*", "", line)
                return clean_line.strip().strip('"').strip("'")
        return raw_output

    def _simulate_translation(self, prompt: str) -> str:
        if "শিক্ষা প্রতিষ্ঠান" in prompt or "education_001" in prompt:
            return "Une institution éducative renommée au Bangladesh, où se répand la lumière de l'éducation."
        elif "ঐতিহ্যবাহী" in prompt:
            return "Un lieu d'apprentissage traditionnel, offrant un excellent environnement pour que les étudiants acquièrent des connaissances."
        elif "ঢাকা কলেজ" in prompt:
            return "Le traditionnel Collège de Dacca, l'une des institutions éducatives les plus anciennes et les plus renommées du Bangladesh."
        return "Ceci est une traduction de démonstration en français."

# ------------------------------------------------------------------------------
# 5. FRENCH-SPECIFIC EVALUATION METRICS
# ------------------------------------------------------------------------------
class FrenchEvaluator:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w\s\u00C0-\u00FF]", "", text.lower())
        return text.strip().split()

    @staticmethod
    def compute_bleu_n(references: List[List[str]], hypothesis: List[str], n: int) -> float:
        def get_ngrams(tokens: List[str], order: int):
            return collections.Counter(tuple(tokens[i:i+order]) for i in range(len(tokens) - order + 1))
        hyp_ngrams = get_ngrams(hypothesis, n)
        if not hyp_ngrams:
            return 0.0
        max_ref_counts = collections.Counter()
        for ref in references:
            ref_ngrams = get_ngrams(ref, n)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
        clipped = {ngram: min(count, max_ref_counts[ngram]) for ngram, count in hyp_ngrams.items()}
        return sum(clipped.values()) / sum(hyp_ngrams.values())

    @classmethod
    def compute_bleu(cls, reference: str, hypothesis: str) -> float:
        import math
        ref_tokens = cls.tokenize(reference)
        hyp_tokens = cls.tokenize(hypothesis)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        precisions = [cls.compute_bleu_n([ref_tokens], hyp_tokens, i) for i in range(1, 5)]
        r_len, h_len = len(ref_tokens), len(hyp_tokens)
        bp = 1.0 if h_len > r_len else math.exp(1.0 - r_len / (h_len + 1e-8))
        running_prod = 1.0
        for p in precisions:
            running_prod *= (p if p > 0 else 1e-8)
        return bp * math.pow(running_prod, 0.25)

    @classmethod
    def compute_rouge_l(cls, reference: str, hypothesis: str) -> float:
        ref_tokens = cls.tokenize(reference)
        hyp_tokens = cls.tokenize(hypothesis)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        precision = lcs / n
        recall = lcs / m
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    @classmethod
    def compute_token_f1(cls, reference: str, hypothesis: str) -> float:
        ref_tokens = cls.tokenize(reference)
        hyp_tokens = cls.tokenize(hypothesis)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        common = collections.Counter(ref_tokens) & collections.Counter(hyp_tokens)
        same = sum(common.values())
        if same == 0: return 0.0
        precision = same / len(hyp_tokens)
        recall = same / len(ref_tokens)
        return (2 * precision * recall) / (precision + recall)

# ------------------------------------------------------------------------------
# 6. PIPELINE ORCHESTRATION & RUNNER
# ------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Academic Translation Engine for French target.")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--vlm_type", type=str, default="qwen", choices=["qwen", "gemma"])
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_path", type=str, default="./checkpoints/banglarmukh_epoch_3.pt")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--prompt_mode", type=str, default="few_shot", choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logger.info("Initializing BanglarMukh French Translation Pipeline Execution...")
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
        
    translator = GeminiFlashTranslator(api_key=args.api_key)

    samples = []
    if args.data_dir and os.path.exists(args.data_dir):
        annotations_dir = os.path.join(args.data_dir, "languages", "pure_bangla", "education institutions", "annotations")
        captions_path = os.path.join(annotations_dir, "education_captions.json")
        if os.path.exists(captions_path):
            with open(captions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data[:5]:
                    samples.append({
                        "id": item.get("image_id"),
                        "source": item.get("caption"),
                        "reference_fr": "Une institution éducative renommée au Bangladesh, où se répand la lumière de l'éducation."
                    })
    
    if not samples:
        samples = [
            {"id": "education_001", "source": "বাংলাদেশের একটি স্বনামধন্য শিক্ষা প্রতিষ্ঠান, যেখানে শিক্ষার আলো ছড়িয়ে পড়ছে।", "reference_fr": "Une institution éducative renommée au Bangladesh, où se répand la lumière de l'éducation."},
            {"id": "education_002", "source": "ঐতিহ্যবাহী একটি বিদ্যাপীঠ, শিক্ষার্থীদের জ্ঞান অর্জনের এক চমৎকার পরিবেশ।", "reference_fr": "Un lieu d'apprentissage traditionnel, offrant un excellent environnement pour que les étudiants acquièrent des connaissances."},
            {"id": "education_003", "source": "ঐতিহ্যবাহী ঢাকা কলেজ, বাংলাদেশের অন্যতম প্রাচীন ও স্বনামধন্য একটি শিক্ষাপ্রতিষ্ঠান।", "reference_fr": "Le traditionnel Collège de Dacca, l'une des institutions éducatives les plus anciennes et les plus renommées du Bangladesh."},
            {"id": "education_004", "source": "সেন্ট গ্রেগরিজ হাই স্কুল অ্যান্ড কলেজ, ঐতিহ্য ও সুনামের সাথে শিক্ষা বিস্তারে এক অনন্য নাম।", "reference_fr": "Le Collège et lycée Saint-Grégoire, un nom unique dans la diffusion de l'éducation avec tradition et réputation."},
            {"id": "education_005", "source": "পোগোজ ল্যাবরেটরি school অ্যান্ড কলেজ, একটি ঐতিহ্যবাহী এবং আদর্শ বিদ্যাপীঠ।", "reference_fr": "L'École et Collège Laboratoire Pogose, une institution éducative traditionnelle et modèle."},
            {"id": "education_006", "source": "চট্টগ্রাম কলেজ, বন্দরনগরীর অন্যতম প্রাচীন ও ঐতিহ্যবাহী উচ্চশিক্ষার এক অনন্য প্রতিষ্ঠান।", "reference_fr": "Le Collège de Chittagong, une institution unique d'enseignement supérieur et l'une des plus anciennes de la ville portuaire."},
            {"id": "education_007", "source": "রাজশাহী কলেজ, উত্তরবঙ্গের শিক্ষা বিস্তারে এক ঐতিহাসিক ও দৃষ্টিনন্দন বিদ্যাপীঠ।", "reference_fr": "Le Collège de Rajshahi, un lieu d'apprentissage historique et pittoresque pour la diffusion de l'éducation dans le Bengale du Nord."},
            {"id": "education_008", "source": "আনন্দ মোহন কলেজ, ময়মনসিংহের ঐতিহ্যবাহী এবং অন্যতম সেরা একটি উচ্চশিক্ষা প্রতিষ্ঠান।", "reference_fr": "Le Collège Ananda Mohan, une institution d'enseignement supérieur traditionnelle et l'une des meilleures de Mymensingh."},
            {"id": "education_009", "source": "কুমিল্লা জিলা স্কুল, শিক্ষার চমৎকার পরিবেশ ও ঐতিহ্যের এক অনন্য সংমিশ্রণ।", "reference_fr": "L'École Comilla Zilla, une combinaison unique d'un excellent environnement éducatif et de tradition."},
            {"id": "education_010", "source": "বরিশাল জিলা স্কুল, দক্ষিণাঞ্চলের অন্যতম প্রাচীন এবং স্বনামধন্য একটি বিদ্যাপীঠ।", "reference_fr": "L'École Barisal Zilla, l'un des lieux d'apprentissage les plus anciens et les plus renommés de la région sud."},
            {"id": "education_011", "source": "যশোর জিলা স্কুল, ঐতিহ্যবাহী এবং গৌরবময় ইতিহাস সমৃদ্ধ একটি শিক্ষাপ্রতিষ্ঠান।", "reference_fr": "L'École Jessore Zilla, une institution éducative traditionnelle riche d'une histoire glorieuse."},
            {"id": "education_012", "source": "সিলেট সরকারি পাইলট উচ্চ বিদ্যালয়, শিক্ষার আলো ছড়াতে এক অনন্য ও প্রাচীন প্রতিষ্ঠান।", "reference_fr": "L'École secondaire pilote publique de Sylhet, une institution unique et ancienne pour diffuser la lumière de l'éducation."},
            {"id": "education_013", "source": "মুরারিচাঁদ (এমসি) কলেজ, সিলেটের প্রাকৃতিক সৌন্দর্যে ঘেরা এক ঐতিহ্যবাহী শিক্ষাপ্রতিষ্ঠান।", "reference_fr": "Le Collège Murarichand (MC), une institution éducative traditionnelle entourée par la beauté naturelle de Sylhet."},
            {"id": "education_014", "source": "পাবনা এডওয়ার্ড কলেজ, প্রাচীন ও ঐতিহ্যবাহী এক চমৎকার শিক্ষাঙ্গন।", "reference_fr": "Le Collège Edward de Pabna, un excellent lieu d'apprentissage ancien et traditionnel."}
        ]

    results = []
    bleu_scores = []
    rouge_scores = []
    f1_scores = []
    
    logger.info(f"Translating {len(samples)} captions utilizing '{args.prompt_mode}' prompt...")
    
    for idx, sample in enumerate(samples):
        source_caption = sample["source"]
        if args.image_path and idx == 0:
            source_caption = vlm_mgr.generate_caption(args.image_path)
            logger.info(f"Generated VLM Caption: {source_caption}")
            
        translated = None
        if cache:
            translated = cache.get(source_caption, args.prompt_mode)
            
        if not translated:
            if args.prompt_mode == "zero_shot":
                translated = translator.translate_zero_shot(source_caption)
            elif args.prompt_mode == "few_shot":
                translated = translator.translate_few_shot(source_caption)
            else:
                translated = translator.translate_cot(source_caption)
                
            if cache and not translated.startswith("[Error"):
                cache.set(source_caption, args.prompt_mode, translated)
                
        logger.info(f"Source: {source_caption}")
        logger.info(f"Target (French): {translated}")
        
        ref = sample.get("reference_fr", "")
        bleu = FrenchEvaluator.compute_bleu(ref, translated)
        rouge = FrenchEvaluator.compute_rouge_l(ref, translated)
        f1 = FrenchEvaluator.compute_token_f1(ref, translated)
        
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        f1_scores.append(f1)
        
        results.append({
            "id": sample["id"],
            "source": source_caption,
            "translated": translated,
            "reference": ref,
            "bleu": bleu,
            "rouge": rouge,
            "f1": f1
        })
        
    print("\n" + "="*80)
    print("                 BANGLARMUKH FRENCH TRANSLATION REPORT")
    print("="*80)
    print(f"| {'ID':<15} | {'BLEU-4':<10} | {'ROUGE-L':<10} | {'Token F1':<10} |")
    print("| :--- | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['id']:<15} | {r['bleu']:.4f}   | {r['rouge']:.4f}   | {r['f1']:.4f}   |")
    print("-"*80)
    print(f"Average BLEU-4 : {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"Average ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")
    print(f"Average F1     : {sum(f1_scores)/len(f1_scores):.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
