import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # define GPU id, remove if you want to use all GPUs available
import torch
from tqdm import tqdm
import time
from contextlib import contextmanager
import numpy as np
import sys

sys.path.append('/home/jewon/code/FASTLLM/Medusa')

from medusa.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
import transformers
from huggingface_hub import hf_hub_download

@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

def medusa_forward(input_ids, model, tokenizer, medusa_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
    wall_times = {'medusa': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}
    
    with timed(wall_times, 'init'):
        if hasattr(model, "medusa_choices") and model.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = model.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=model.base_model.device
            )
        model.medusa_buffers = medusa_buffers
        model.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_medusa_mode(model)
        medusa_logits, logits = initialize_medusa(
                input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
        )
    new_token = 0

    for idx in range(max_steps): 
        with timed(wall_times, 'medusa'):
            candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                )

        with timed(wall_times, 'tree'):
            medusa_logits, logits, outputs = tree_decoding(
                    model,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )

        with timed(wall_times, 'posterior'):
            best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature, posterior_threshold, posterior_alpha
                )
        
        with timed(wall_times, 'update'):
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break

    return input_ids, new_token, idx, wall_times

model_name = 'FasterDecoding/medusa-vicuna-7b-v1.3'
model = MedusaModel.from_pretrained(
    model_name,
    medusa_num_heads = 4,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
tokenizer = model.get_tokenizer()

medusa_choices = mc_sim_7b_63

temperature = 0.7
posterior_threshold = 0.09
posterior_alpha = 0.3

prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"

with torch.inference_mode():
    input_ids = tokenizer([prompt]).input_ids
    output_ids, new_token, idx, wall_time = medusa_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    medusa_choices,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                )
    output_ids = output_ids[0][len(input_ids[0]) :]
    print("Output length:", output_ids.size(-1))
    print("Compression ratio:", new_token / idx)
    
output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
print(output)