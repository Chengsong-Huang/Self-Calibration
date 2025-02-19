import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, HfFolder

def main(base_model_name, lora_path, merged_model_path, upload_to_hf, hf_repo_name):
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.resize_token_embeddings(len(tokenizer))
    # Load LoRA model and merge
    peft_model = PeftModel.from_pretrained(model, lora_path)
    peft_model = peft_model.merge_and_unload()

    # Save the merged model and tokenizer
    peft_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model and tokenizer saved to {merged_model_path}")

    # Optionally upload to Hugging Face Hub
    if upload_to_hf:
        if not hf_repo_name:
            print("Error: --hf_repo_name must be provided when --upload_to_hf is set.")
            return

        api = HfApi()
        username = api.whoami()['name']
        repo_id = f"{username}/{hf_repo_name}"

        print(f"Uploading to Hugging Face Hub repository: {repo_id}")
        peft_model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"Model and tokenizer successfully uploaded to Hugging Face Hub at {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into a base model and optionally upload to Hugging Face.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Path or name of the base model.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint.")
    parser.add_argument("--merged_model_path", type=str, required=True, help="Path to save the merged model.")
    parser.add_argument("--upload_to_hf", action="store_true", help="Whether to upload the merged model to Hugging Face Hub.")
    parser.add_argument("--hf_repo_name", type=str, help="Name of the repository on Hugging Face Hub.")

    args = parser.parse_args()
    main(args.base_model_name, args.lora_path, args.merged_model_path, args.upload_to_hf, args.hf_repo_name)