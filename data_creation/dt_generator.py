import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class EntropyDynamicTemperatureGenerator:
    def __init__(self, model_name, base_temperature=0.8, N=0.8, theta=1.0, device='cuda'):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_temperature = base_temperature
        self.N = N
        self.theta = theta

    def calculate_entropy_from_logits(self, logits):
        max_logits = torch.max(logits, dim=-1, keepdim=True).values
        stable_logits = logits - max_logits
        exp_logits = torch.exp(stable_logits)
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        softmax_probs = exp_logits / sum_exp_logits
        entropy = -(softmax_probs * stable_logits).sum(dim=-1)
        return entropy

    def adjust_temperature(self, entropy):
        entropy = entropy.item()
        if entropy == 0:
            return 0  # Directly use 0 to indicate greedy decoding
        return self.base_temperature * (self.N ** (self.theta / entropy))

    def generate(self, prompt, max_tokens=50):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_text = prompt

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]

            entropy = self.calculate_entropy_from_logits(logits)
            temperature = self.adjust_temperature(entropy)
            if temperature < 0.001:
                # Greedy decoding: Select the token with the highest probability
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Use temperature-scaled logits to sample the next token
                scaled_logits = logits / temperature
                next_token_id = torch.multinomial(torch.softmax(scaled_logits, dim=-1), num_samples=1)

            next_token_str = self.tokenizer.decode(next_token_id.squeeze().item())
            generated_text += next_token_str
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_str in [self.tokenizer.eos_token]:
                break

        return generated_text

    def delete(self):
        del self.model

if __name__ == "__main__":

    model_name = "meta-llama/Llama-3.1-8B-Instruct" 
    generator = EntropyDynamicTemperatureGenerator(model_name, base_temperature=0.8, N=0.8, theta=1.0)


    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

For the following question, provide a step-by-step explanation of your thought process.
Use the format demonstrated below for your response.
```Example Format:
Explanation: <Your detailed explanation here, outlining how you arrived at your answer.>
Answer: <Insert your concise answer here, which should include a number (e.g., 1)>
Ensure that your response strictly adheres to this format. Explicitly include the words 'Explanation:', 'Answer:'.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


    result = generator.generate(prompt, max_tokens=1024)
    print("Generated Text:")
    print(result)
