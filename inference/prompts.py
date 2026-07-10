


# 1. Zero-shot Captioning Prompt
ZERO_SHOT_CAPTION_PROMPT = """You are an assistant who generates short, fluent captions in {source_language} only.
Look carefully at the given image and write exactly one meaningful sentence describing it.
Do not use any {other_language} words, do not add extra explanations, labels, or quotes.
Your entire output must be only the {source_language} caption as plain text."""

# 2. Few-shot Captioning Prompt
FEW_SHOT_CAPTION_PROMPT = """You are an assistant who generates short, fluent captions in {source_language} only.
Here are three example captions:
Example 1: Image: {example_image_1} Caption: {example_caption_1}
Example 2: Image: {example_image_2} Caption: {example_caption_2}
Example 3: Image: {example_image_3} Caption: {example_caption_3}
Now, generate a caption for the following image: Image: {image_path}
Write exactly one meaningful {source_language} sentence. Do not use any {other_language} words, do not add extra explanations, labels, or quotes. Your entire output must be only the {source_language} caption as plain text."""

# 3. Zero-shot VQA Prompt
ZERO_SHOT_VQA_PROMPT = """You are an AI assistant that answers visual multiple-choice questions in {source_language}.
Task:
1. Look carefully at the given image: {image_path}
2. Read the question: {question}
3. Review the provided answer choices: {options}
4. Select the single most accurate answer.
Response Rules:
- The index must be the programming list index (starting from 0).
- Respond ONLY with the exact format below.
- Use {source_language} text for the answer option.
- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.
- Follow this exact structure:
Index: <option index>, Answer: "<option text in {source_language}>\""""

# 4. Few-shot VQA Prompt
FEW_SHOT_VQA_PROMPT = """You are an AI assistant that answers visual multiple-choice questions in {source_language}.
Task:
1. Look carefully at the given image: {image_path}
2. Read the question: {question}
3. Review the provided answer choices: {options}
4. Select the single most accurate answer.
Response Rules:
- The index must be the programming list index (starting from 0).
- Respond ONLY with the exact format below.
- Use {source_language} text for the answer option.
- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.
- Follow this exact structure: Index: <option index>, Answer: "<option text in {source_language}>"
Here are three example QA pairs:
Example 1: Image: {example_image_1} Question: {example_question_1} Options: {example_options_1} Answer: Index: {example_index_1}, Answer: "{example_answer_1}"
Example 2: Image: {example_image_2} Question: {example_question_2} Options: {example_options_2} Answer: Index: {example_index_2}, Answer: "{example_answer_2}"
Example 3: Image: {example_image_3} Question: {example_question_3} Options: {example_options_3} Answer: Index: {example_index_3}, Answer: "{example_answer_3}"
Now, answer for the given image. Image: {image_path} Question: {question} Options: {options}"""

# 5. Chain-of-Thought (CoT) VQA Prompt
COT_VQA_PROMPT = """You are an AI assistant that answers visual multiple-choice questions in {source_language}.
Task:
1. Look carefully at the given image: {image_path}
2. Read the question: {question}
3. Review the provided answer choices: {options}
4. Select the single most accurate answer.
Response Rules:
- The index must be the programming list index (starting from 0).
- Use {source_language} text for the answer option.
- In Reasoning En, write step-by-step reasoning in English - break down the solution logically:
  Step 1: Describe key visual observations.
  Step 2: Match observations to relevant answer choices.
  Step 3: Eliminate incorrect choices with brief justification.
  Step 4: Conclude why the final choice is correct.
- Be clear, concise, and factual (avoid overly long explanations).
- Follow this exact response format:
Reasoning En:
Step 1: <your observations>
Step 2: <your matching logic>
Step 3: <your elimination of wrong options>
Step 4: <your final choice reasoning>
Final Answer: Index: <option index>, Answer: "<option text in {source_language}>\""""

# 6. LLM-as-a-Judge Prompt
LLM_AS_A_JUDGE_PROMPT = """You are a highly skilled and impartial caption evaluator. Your task is to carefully compare a generated caption with a reference caption and score it according to the following dimensions:
1. Relevance (0-1): How well does the generated caption describe the main objects, actions, and cultural context of the reference caption? Reward high semantic overlap and penalize missing or hallucinated details.
2. Clarity (0-1): Is the caption grammatically correct, well-structured, and easy to read?
3. Conciseness (0-1): Is the caption free of redundancy, filler words, or unnecessary complexity while still conveying the full meaning?
4. Creativity (0-1): Does the caption show originality or an engaging phrasing, rather than being overly generic?
After scoring each dimension, compute an Overall (0-1) score that reflects the holistic quality of the generated caption, giving slightly higher weight to Relevance and Clarity.
Your response must strictly follow this JSON-like structure:
Relevance: [float between 0 and 1]
Clarity: [float between 0 and 1]
Conciseness: [float between 0 and 1]
Creativity: [float between 0 and 1]
Overall: [float between 0 and 1]
Explanation: [Concise explanation: mention key strengths, weaknesses, and reasoning for the scores.]

Reference Caption: {reference_caption}
Generated Caption: {generated_caption}"""
