#!/usr/bin/env python3
"""Generate MTBench prompt files for benchmarking"""

import os

# MTBench questions from various categories
MTBENCH_SAMPLES = [
    ("Please teach me how to make pancake.", "cooking"),
    ("Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.", "writing"),
    ("Write a persuasive email to convince your introverted friend to attend a party.", "writing"),
    ("Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer than two paragraphs.", "writing"),
    ("Write a function to find the nth Fibonacci number using dynamic programming.", "coding"),
    ("Implement a regular expression in Python to validate an email address.", "coding"),
    ("Write a program to find the nth Fibonacci number using dynamic programming.", "math"),
    ("Given that f(x) = 5x^3 - 2x + 3, find the value of f(2).", "math"),
    ("Solve for x in the equation 3x + 10 = 5(x - 2).", "math"),
    ("If the endpoints of a line segment are (2, -2) and (10, 4), what is the length of the segment?", "math"),
    ("How can I improve my time management skills?", "general"),
    ("What are the most effective ways to deal with stress?", "general"),
    ("How do I develop my critical thinking skills?", "general"),
    ("What are some effective ways to improve communication in a relationship?", "relationship"),
    ("How can governments and individuals balance economic growth with environmental sustainability?", "reasoning"),
    ("How do language and cultural barriers affect the way people communicate and form relationships in multicultural societies?", "reasoning"),
    ("Describe a scenario where artificial intelligence could be used to improve the quality and efficiency of healthcare delivery.", "reasoning"),
    ("How many times does the average human blink in a lifetime?", "fermi"),
    ("How many atoms are in a grain of salt?", "fermi"),
    ("How many lightning strikes occur on Earth each day?", "fermi"),
]

def format_llama_prompt(user_message):
    """Format prompt using Llama chat template"""
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )
    
    return prompt

def main():
    output_dir = "mtbench_prompts"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (question, category) in enumerate(MTBENCH_SAMPLES, 1):
        prompt = format_llama_prompt(question)
        
        prompt_file = os.path.join(output_dir, f"prompt_{i:02d}.txt")
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        # Store category as metadata
        meta_file = os.path.join(output_dir, f"meta_{i:02d}.txt")
        with open(meta_file, 'w') as f:
            f.write(f"Category: {category}\nQuestion: {question}\n")
    
    print(f"Generated {len(MTBENCH_SAMPLES)} MTBench prompt files in {output_dir}/")
    print(f"Prompt files: prompt_01.txt to prompt_{len(MTBENCH_SAMPLES):02d}.txt")
    print(f"Metadata files: meta_01.txt to meta_{len(MTBENCH_SAMPLES):02d}.txt")

if __name__ == "__main__":
    main()
