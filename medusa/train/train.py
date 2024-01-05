# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother


from typing import Any, Dict, List, Optional, Union

import sys
sys.path.append("../..")

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
from medusa.model.medusa_model import MedusaModel, MedusaConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/ssd0/data/fast-llm/Llama-2-70B-Chat-fp16")
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="/ssd0/data/fast-llm/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True



import argparse
parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/ssd0/data/fast-llm/Llama-2-70B-Chat-fp16')
parser.add_argument('--configpath', type=str, default="/home/jewon/code/FASTLLM/EAGLE/train/llama_2_chat_70B_config.json")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--gradient-accumulation-steps', type=int, default=2)
parser.add_argument('--tmpdir', type=str, default='/ssd0/data/fast-llm/eagle_train_data')
parser.add_argument('--outdir', type=str, default='/ssd0/checkpoints/fast-llm/medusa_test_1')
parser.add_argument('--cpdir', type=str, default='/ssd0/checkpoints/fast-llm/medusa_test_1')
args = parser.parse_args()

train_config={
    "lr":args.lr,
    "bs":args.bs,
    "gradient_accumulation_steps":args.gradient_accumulation_steps,
    "datapath":f"{args.tmpdir}",
    "num_epochs":1,
    "num_warmup_steps":400,
    "total_steps":4070,
    "num_workers":16,
    "max_len":2048,
    "config_path":args.configpath,
    "b1":0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
}

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data=datapath
        self.transform = transform


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        # try:
        data=torch.load(self.data[index])
        new_data={}
        hidden_state=data['hidden_state'][:train_config["max_len"]][None,:]
        input_ids = data['input_ids'][:train_config["max_len"]][None,:]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None,:]

        # except:
        #     with open("error_path.txt", "w") as file:
        #         file.write(self.data[index])
        #     print('error path',self.data[index])


        length=hidden_state.shape[1]
        #length_q = data['query_ids'].shape[1]
        attention_mask=[1]*length
        loss_mask=loss_mask[0].tolist()
        loss_mask[-1]=0

        input_ids_target=input_ids[:,1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target=hidden_state[:,1:,:]
        zeropadding=torch.zeros(1, 1, target.shape[2])
        target=torch.cat((target,zeropadding), dim=1)
        loss_mask[-1]=0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"]=target
        new_data["hidden_state_big"]=hidden_state
        new_data["input_ids"] = input_ids_target
        #sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        #label = data['y']

        if self.transform:
            new_data = self.transform(new_data)

        return new_data

class DataCollatorWithPadding:


    def paddingtensor(self,intensors,N):
        B,n,S=intensors.shape
        #padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self,intensors,N):
        B,n=intensors.shape
        padding_tensor = torch.zeros(B, N - n,dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states=torch.cat([self.paddingtensor(item['hidden_state_big'],max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor([item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids":batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target":batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def train():
    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.basepath,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Add Medusa heads
    medusa_lm_head = MedusaModel(
        base_model=model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path=args.basepath,
        only_medusa=False,
    )
    
    medusa_heads = medusa_lm_head.medusa_head

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.basepath,
        # cache_dir=training_args.cache_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token


    # Generate Medusa config for pushing to HF hub
    medusa_config = MedusaConfig(
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path=args.basepath,
    )

    # Save Medusa config
    medusa_config.save_pretrained(args.outdir)
    
    datapath = list_files(train_config["datapath"])
    traindatapath=datapath[:int(len(datapath)*0.95)]
    testdatapath=datapath[int(len(datapath)*0.95):]

    train_dataset = CustomDataset(traindatapath)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config["bs"], shuffle=True, num_workers=train_config["num_workers"],collate_fn=DataCollatorWithPadding())
    test_dataset = CustomDataset(testdatapath)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=train_config["bs"], shuffle=False, num_workers=train_config["num_workers"],collate_fn=DataCollatorWithPadding())
    
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    set_seed(0)
    accelerator = Accelerator(mixed_precision='bf16',gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    optimizer = torch.optim.AdamW(medusa_heads.parameters(), lr=train_config["lr"])
    
    # huggingface cosine scheduler with warmup
    from transformers import get_cosine_schedule_with_warmup

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_config["num_warmup_steps"], num_training_steps=train_config["total_steps"])
    
    medusa_heads, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        medusa_heads, optimizer, train_dataloader, lr_scheduler
    )
    
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"{args.outdir}/logs")
    
    global_step=0
    for epoch in range(1):
        medusa_heads.train()
        # medusa_lm_head.train()
        log_dict = {}
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = 0
            loss_fct = CrossEntropyLoss()
            for i in range(medusa_config.medusa_num_heads):
                medusa_logits = medusa_heads.module[i](batch["hidden_states"].to(torch.bfloat16))[:,:-2-i,:].contiguous()
                medusa_labels = batch["input_ids"][..., 2 + i :].contiguous()
                medusa_logits = medusa_logits.view(-1, medusa_logits.shape[-1])
                medusa_labels = medusa_labels.view(-1)
                medusa_labels = medusa_labels.to(medusa_logits.device)
                loss_i = loss_fct(medusa_logits, medusa_labels)

                # loss_i = loss_fct(medusa_logits.view(-1, medusa_logits.shape[-1]), medusa_labels.view(-1))
                loss += loss_i
                not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
                medusa_labels = medusa_labels[not_ignore]

                for k in range(1, 6):
                    _, topk = medusa_logits.topk(k, dim=-1)
                    topk = topk[not_ignore]
                    correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                    log_dict[f"medusa{i}_top{k}"] = correct.float().mean().item()
                
                log_dict[f"medusa{i}_loss"] = loss_i.item()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(medusa_lm_head.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            if accelerator.is_main_process:
                writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("loss", loss.item(), global_step)
                for k, v in log_dict.items():
                    writer.add_scalar(k, v, global_step)

    # Save MedusaHead seperately
    if hasattr(medusa_lm_head, "module"):
        lm_head = medusa_lm_head.module.medusa_head
    else:
        lm_head = medusa_lm_head.medusa_head

    # Save Medusa heads
    accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")

    epoch_loss = 0
    num_batches = 0
    medusa_heads.eval()
    
    for step, batch in enumerate(test_dataloader):
        test_log_dict = {}
        with torch.no_grad():
            for i in range(medusa_config.medusa_num_heads):
                medusa_logits = medusa_heads.module[i](batch["hidden_states"].to(torch.bfloat16))[:,:-2,:].contiguous()
                medusa_labels = batch["input_ids"][..., 2 :].contiguous()
                medusa_logits = medusa_logits.view(-1, medusa_logits.shape[-1])
                medusa_labels = medusa_labels.view(-1)
                medusa_labels = medusa_labels.to(medusa_logits.device)
                loss_i = loss_fct(medusa_logits, medusa_labels)
                
                not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
                medusa_labels = medusa_labels[not_ignore]

                loss += loss_i
                not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
                medusa_labels = medusa_labels[not_ignore]
                
            for k in range(1, 6):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                test_log_dict[f"test_medusa{i}_top{k}"] = correct.float().mean().item()
            
            test_log_dict[f"test_medusa{i}_loss"] = loss_i.item()
    
    if accelerator.is_main_process:
        for k, v in test_log_dict.items():
            writer.add_scalar(k, v, epoch+1)
        epoch_loss += loss.item()
        num_batches += 1
            
if __name__ == "__main__":
    train()
