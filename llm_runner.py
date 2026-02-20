#-------------------------------UPDATED (Fixed output + Stable + Batch-Safe) -------------------------------------#
import torch
import json
import os
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from model_registry import LLM_MODELS
from tqdm import tqdm
from prompts import build_prompt_appearance, build_prompt_gbv


# -------------------------
# Utility
# -------------------------

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def repair_json(text: str) -> str:
    """
    Repair common LLM JSON truncation issues.
    """

    if not text:
        return ""

    text = text.strip()

    # Replace single quotes
    text = text.replace("'", '"')

    # Fix quoted booleans
    text = text.replace('"true"', 'true')
    text = text.replace('"false"', 'false')

    # Count brackets
    open_curly = text.count("{")
    close_curly = text.count("}")
    open_square = text.count("[")
    close_square = text.count("]")

    # Close square brackets first
    if open_square > close_square:
        text += "]" * (open_square - close_square)

    # Then close curly brackets
    if open_curly > close_curly:
        text += "}" * (open_curly - close_curly)

    return text


# Stop generation once a JSON object closes
# class StopOnJSONEnd(StoppingCriteria):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, input_ids, scores, **kwargs):
#         decoded = self.tokenizer.decode(
#             input_ids[0],
#             skip_special_tokens=True
#         )
#         return decoded.strip().endswith("}")
# This works but can be inefficient for long generations, so we can optimize by only decoding the last few tokens:   
# class StopOnJSONEnd(StoppingCriteria):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, input_ids, scores, **kwargs):
#         # Only decode last 50 tokens for efficiency
#         decoded = self.tokenizer.decode(
#             input_ids[0][-50:],
#             skip_special_tokens=True
#         )
#         return decoded.strip().endswith("}")


# -------------------------
# Runner
# -------------------------

class UnifiedLLMRunner:

    def __init__(self, task="appearance", batch_size=8, max_new_tokens=180):
        self.device = "cuda"
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        if task == "appearance":
            self.build_prompt = build_prompt_appearance
            self.task_name = "appearance"
        elif task == "gbv":
            self.build_prompt = build_prompt_gbv
            self.task_name = "gbv"
        else:
            raise ValueError("Task must be 'appearance' or 'gbv'")

        self.datasetName = "unKNOWN"

        # ⚠ Use absolute path in production
        self.output_base = "/datasets/cl0059/outputs/llm_results"
        os.makedirs(self.output_base, exist_ok=True)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # -------------------------
    # Load Model
    # -------------------------
    def load_model(self, model_id):

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        model.eval()
        return model, tokenizer


    # -------------------------
    # Build Inputs (Chat-aware)
    # -------------------------
    ### Updated to force JSON prefix anchor for better output consistency across models, especially those that may not follow instructions as strictly. This should help ensure that the model's response starts with a JSON object, improving parsing reliability.
    def build_inputs(self, tokenizer, comments, model_name):

        prompts = []

        for comment in comments:

            base_prompt = self.build_prompt(comment)

            # Force JSON prefix anchor
            base_prompt = base_prompt + "\n\nReturn ONLY valid JSON.\nThe first character of your response MUST be '{'.\n"

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:

                if "llama" in model_name.lower():
                    messages = [
                        {"role": "system", "content":
                            "You are a strict information extraction system. "
                            "You must output valid JSON only. "
                            "Do not explain. Do not continue text. "
                            f"If no {self.task_name}, return contains_{self.task_name}=false JSON."
                        },
                        {"role": "user", "content": base_prompt}
                    ]

                elif "gemma" in model_name.lower():
                    messages = [
                        {"role": "user", "content": base_prompt}
                    ]

                else:
                    messages = [
                        {"role": "system", "content":
                            "You are a strict JSON-only classifier."
                        },
                        {"role": "user", "content": base_prompt}
                    ]

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            else:
                prompt = base_prompt

            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)

        return inputs


    # -------------------------
    # Batch Processing
    # -------------------------
    def process_dataset(self, comments, model_name, model_id):

        output_file = os.path.join(
            self.output_base,
            f"{model_name}_{self.task_name}_results_{self.datasetName}.jsonl"
        )

        # Check if file exists and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            # 'w' mode opens for writing and truncates the file to zero length
            with open(output_file, 'w') as f:
                pass

        print(f"\n🚀 Running {model_name}")
        clear_gpu_memory()

        model, tokenizer = self.load_model(model_id)

        # stopping_criteria = StoppingCriteriaList([
        #     StopOnJSONEnd(tokenizer)
        # ])

        for i in tqdm(range(0, len(comments), self.batch_size)):

            batch = comments[i:i+self.batch_size]
            batch_cids = [cid for cid, _ in batch]
            batch_comments = [text for _, text in batch]

            inputs = self.build_inputs(tokenizer, batch_comments, model_name)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.1,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    # stopping_criteria=stopping_criteria
                )

            # Only decode generated part of the sequence for efficiency and to avoid decoding the prompt
            generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

            decoded = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            cleaned_outputs = []

            for output in decoded:
                output = output.strip()

                # If model did not start JSON, discard garbage
                # Trim leading garbage
                if "{" in output:
                    output = output[output.find("{"):]
                else:
                    output = ""
                # Do NOT cut at first }
                # Let repair_json handle bracket balancing

                output = repair_json(output)
                cleaned_outputs.append(output)

            with open(output_file, "a") as f:
                # for cid, output in zip(batch_cids, decoded):
                for cid, output in zip(batch_cids, cleaned_outputs):
                    record = {
                        "model": model_name,
                        "cid": cid,
                        "raw_output": output.strip()
                    }
                    f.write(json.dumps(record) + "\n")

        del model
        del tokenizer
        clear_gpu_memory()

        print(f"✅ Completed {model_name}")


    # -------------------------
    # Run All
    # -------------------------
    def run_all(self, comments, datasetName):
        self.datasetName = datasetName
        for name, model_id in LLM_MODELS.items():
            try:
                self.process_dataset(comments, name, model_id)
            except Exception as e:
                print(f"❌ {name} failed: {e}")
