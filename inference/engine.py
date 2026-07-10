import torch
import re
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image

from inference.prompts import (
    ZERO_SHOT_CAP_PROMPT = ZERO_SHOT_CAPTION_PROMPT,
    FEW_SHOT_CAP_PROMPT = FEW_SHOT_CAPTION_PROMPT,
    ZERO_SHOT_VQA_PROMPT,
    FEW_SHOT_VQA_PROMPT,
    COT_VQA_PROMPT
)

# Bring prompt templates into local scope for clean usage
ZERO_SHOT_CAP_PROMPT = globals().get('ZERO_SHOT_CAPTION_PROMPT', None)
FEW_SHOT_CAP_PROMPT = globals().get('FEW_SHOT_CAPTION_PROMPT', None)

from inference.prompts import (
    ZERO_SHOT_CAPTION_PROMPT,
    FEW_SHOT_CAPTION_PROMPT,
    ZERO_SHOT_VQA_PROMPT,
    FEW_SHOT_VQA_PROMPT,
    COT_VQA_PROMPT
)

class BanglarMukhInferenceEngine:
    """
    Inference engine that manages zero-shot, few-shot, and Chain-of-Thought (CoT) prompting
    for captioning and visual question answering (VQA) using Qwen2-VL or PaliGemma.
    """
    def __init__(self, model: torch.nn.Module, processor: Any, vlm_type: str = "qwen", device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.vlm_type = vlm_type.lower()
        self.device = device
        
        self.model.eval()

    def _prepare_inputs(self, prompt: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Preprocesses text and image inputs based on the model's architecture.
        """
        if self.vlm_type == "qwen":
            # Formulate Qwen model format utilizing the appropriate chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template if processor supports it
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = f"<|im_start|>user\n<|image_pad|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Use qwen-vl-utils style processing or standard processor calls
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
        elif self.vlm_type == "gemma":
            
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
        else:
            raise ValueError(f"Unknown VLM type: {self.vlm_type}")
            
        # Move all tensors to the target device
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def _generate_text(self, inputs: Dict[str, Any], max_new_tokens: int = 512) -> str:
        """
        Generates text given preprocessed inputs, handling sequence padding and decoder start tokens.
        """
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, 
                temperature=0.0
            )
            
            # Retrieve generated segment only by stripping the original prompt tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = [
                out_ids[input_len:] for out_ids in generated_ids
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
        return output_text.strip()

    def generate_zero_shot_caption(
        self, 
        image: Image.Image, 
        source_language: str = "Bengali", 
        other_language: str = "English"
    ) -> str:
        """
        Generates a zero-shot caption for the given image.
        """
        prompt = ZERO_SHOT_CAPTION_PROMPT.format(
            source_language=source_language,
            other_language=other_language
        )
        inputs = self._prepare_inputs(prompt, image)
        return self._generate_text(inputs, max_new_tokens=128)

    def generate_few_shot_caption(
        self, 
        image: Image.Image, 
        examples: List[Dict[str, Any]], 
        image_path_label: str = "current_image.jpg",
        source_language: str = "Bengali", 
        other_language: str = "English"
    ) -> str:
        """
        Generates a few-shot caption using a list of three examples.
        Each example in the list should contain: {"image_path": str, "caption": str}
        """
        if len(examples) < 3:
            raise ValueError("Few-shot captioning requires at least 3 examples.")
            
        prompt = FEW_SHOT_CAPTION_PROMPT.format(
            source_language=source_language,
            other_language=other_language,
            example_image_1=examples[0]["image_path"],
            example_caption_1=examples[0]["caption"],
            example_image_2=examples[1]["image_path"],
            example_caption_2=examples[1]["caption"],
            example_image_3=examples[2]["image_path"],
            example_caption_3=examples[2]["caption"],
            image_path=image_path_label
        )
        inputs = self._prepare_inputs(prompt, image)
        return self._generate_text(inputs, max_new_tokens=128)

    def generate_zero_shot_vqa(
        self, 
        image: Image.Image, 
        question: str, 
        options: List[str], 
        image_path_label: str = "current_image.jpg",
        source_language: str = "Bengali"
    ) -> Tuple[Optional[int], str, str]:
        """
        Answers VQA in zero-shot style.
        Returns:
            parsed_index: Selected option index (starting from 0)
            parsed_answer: Answer text
            raw_output: Raw generated text
        """
        formatted_options = "\n".join([f"({i}) {opt}" for i, opt in enumerate(options)])
        prompt = ZERO_SHOT_VQA_PROMPT.format(
            source_language=source_language,
            image_path=image_path_label,
            question=question,
            options=formatted_options
        )
        inputs = self._prepare_inputs(prompt, image)
        raw_output = self._generate_text(inputs, max_new_tokens=64)
        
        index, answer = self._parse_vqa_response(raw_output)
        return index, answer, raw_output

    def generate_few_shot_vqa(
        self, 
        image: Image.Image, 
        question: str, 
        options: List[str], 
        examples: List[Dict[str, Any]], 
        image_path_label: str = "current_image.jpg",
        source_language: str = "Bengali"
    ) -> Tuple[Optional[int], str, str]:
        """
        Answers VQA in few-shot style.
        Each example in the list should contain:
            {"image_path": str, "question": str, "options": List[str], "index": int, "answer": str}
        """
        if len(examples) < 3:
            raise ValueError("Few-shot VQA requires at least 3 examples.")
            
        formatted_options = "\n".join([f"({i}) {opt}" for i, opt in enumerate(options)])
        
        formatted_examples = []
        for i, ex in enumerate(examples):
            ex_opts = "\n".join([f"({j}) {o}" for j, o in enumerate(ex["options"])])
            formatted_examples.append({
                f"example_image_{i+1}": ex["image_path"],
                f"example_question_{i+1}": ex["question"],
                f"example_options_{i+1}": ex_opts,
                f"example_index_{i+1}": ex["index"],
                f"example_answer_{i+1}": ex["answer"]
            })
            
        prompt = FEW_SHOT_VQA_PROMPT.format(
            source_language=source_language,
            image_path=image_path_label,
            question=question,
            options=formatted_options,
            example_image_1=formatted_examples[0]["example_image_1"],
            example_question_1=formatted_examples[0]["example_question_1"],
            example_options_1=formatted_examples[0]["example_options_1"],
            example_index_1=formatted_examples[0]["example_index_1"],
            example_answer_1=formatted_examples[0]["example_answer_1"],
            example_image_2=formatted_examples[1]["example_image_2"],
            example_question_2=formatted_examples[1]["example_question_2"],
            example_options_2=formatted_examples[1]["example_options_2"],
            example_index_2=formatted_examples[1]["example_index_2"],
            example_answer_2=formatted_examples[1]["example_answer_2"],
            example_image_3=formatted_examples[2]["example_image_3"],
            example_question_3=formatted_examples[2]["example_question_3"],
            example_options_3=formatted_examples[2]["example_options_3"],
            example_index_3=formatted_examples[2]["example_index_3"],
            example_answer_3=formatted_examples[2]["example_answer_3"]
        )
        
        inputs = self._prepare_inputs(prompt, image)
        raw_output = self._generate_text(inputs, max_new_tokens=64)
        
        index, answer = self._parse_vqa_response(raw_output)
        return index, answer, raw_output

    def generate_cot_vqa(
        self, 
        image: Image.Image, 
        question: str, 
        options: List[str], 
        image_path_label: str = "current_image.jpg",
        source_language: str = "Bengali"
    ) -> Tuple[Dict[str, str], Optional[int], str, str]:
        """
        Answers VQA using Chain-of-Thought reasoning.
        Returns:
            reasoning_steps: Dictionary of reasoning steps 1-4
            parsed_index: Index of final answer option
            parsed_answer: Content of final answer option
            raw_output: Full generated string
        """
        formatted_options = "\n".join([f"({i}) {opt}" for i, opt in enumerate(options)])
        prompt = COT_VQA_PROMPT.format(
            source_language=source_language,
            image_path=image_path_label,
            question=question,
            options=formatted_options
        )
        inputs = self._prepare_inputs(prompt, image)
        raw_output = self._generate_text(inputs, max_new_tokens=512)
        
        steps, index, answer = self._parse_cot_response(raw_output)
        return steps, index, answer, raw_output

    def _parse_vqa_response(self, text: str) -> Tuple[Optional[int], str]:
        """
        Parses answers matching the format: Index: <index>, Answer: "<answer>"
        """
        # Regular expression to extract the index and the answer
        match = re.search(r"Index:\s*(\d+),\s*Answer:\s*\"([^\"]+)\"", text)
        if match:
            idx = int(match.group(1))
            ans = match.group(2)
            return idx, ans
        
    
        match_alt = re.search(r"Index:\s*(\d+).*?Answer:\s*(.+)", text, re.IGNORECASE)
        if match_alt:
            idx = int(match_alt.group(1))
            ans = match_alt.group(2).strip().strip('"').strip("'")
            return idx, ans
            
        return None, text

    def _parse_cot_response(self, text: str) -> Tuple[Dict[str, str], Optional[int], str]:
        """
        Parses CoT responses.
        Extracts step-by-step reasoning steps 1 to 4 and the final answers.
        """
        steps = {"step_1": "", "step_2": "", "step_3": "", "step_4": ""}
        
        # Extract steps using regex
        step1_match = re.search(r"Step 1:\s*(.*?)(?=Step 2:|Final Answer:|$)", text, re.DOTALL)
        step2_match = re.search(r"Step 2:\s*(.*?)(?=Step 3:|Final Answer:|$)", text, re.DOTALL)
        step3_match = re.search(r"Step 3:\s*(.*?)(?=Step 4:|Final Answer:|$)", text, re.DOTALL)
        step4_match = re.search(r"Step 4:\s*(.*?)(?=Final Answer:|$)", text, re.DOTALL)
        
        if step1_match: steps["step_1"] = step1_match.group(1).strip()
        if step2_match: steps["step_2"] = step2_match.group(1).strip()
        if step3_match: steps["step_3"] = step3_match.group(1).strip()
        if step4_match: steps["step_4"] = step4_match.group(1).strip()
        
        # Parse final answer block
        final_match = re.search(r"Final Answer:\s*(.*)", text)
        if final_match:
            final_text = final_match.group(1).strip()
            idx, ans = self._parse_vqa_response(final_text)
            return steps, idx, ans
            
            
        idx, ans = self._parse_vqa_response(text)
        return steps, idx, ans
