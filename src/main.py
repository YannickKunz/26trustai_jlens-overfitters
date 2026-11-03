import jlens
import transformers
import torch
from datasets import load_dataset
from jlens.examples import load_wikitext_prompts

MODEL_NAME = "Qwen/Qwen3.5-4B"
# MODEL_NAME = "Qwen/Qwen3.6-27B"

def main():
    jlens.configure_logging()

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        dtype=torch.bfloat16
    ).cuda()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)

    # Wikitext
    # prompts = load_wikitext_prompts(n_prompts=11)

    # Math
    prompts = load_dataset("openai/gsm8k", "main")
    prompts = [prompt["question"] for prompt in prompts["train"]][:1000]

    lens = jlens.fit(
        model, prompts, dim_batch=8, max_seq_len=128, checkpoint_path="gsm8k_questions_ckpt.pt"
    )
    lens.save("jacobian_lens.pt")

if __name__ == "__main__":
    main()