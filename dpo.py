# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# Full training
python dpo.py \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --model_name_or_path "sft_qwen2_1.5b_ultrafeedback_3epoch" \
    --learning_rate 5.0e-7 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2_1.5B_DPO_ultrafeedback \
    --no_remove_unused_columns

# LoRA:
python examples/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""
from typing import Dict, Optional, Tuple, Union, List, Literal
from trl.commands.cli_utils import DPOScriptArguments, TrlParser
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    maybe_extract_prompt,
    maybe_apply_chat_template,
)

class ECEDP0Trainer(DPOTrainer):
    def __init__(self, *args, beta_update_interval=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_update_interval = beta_update_interval
        self.eval_step_counter = 1
        self.temperature = nn.Parameter((torch.ones(1)*1).to(device))
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Check if it's time to update beta
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chosen = []
        reject = []
        self.model.eval() 
        with torch.no_grad():  
            if self.eval_step_counter % self.beta_update_interval == 0:
                eval_dataloader = self.get_eval_dataloader(eval_dataset)
                for batch in eval_dataloader:
                    compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
                    with compute_loss_context_manager():
                        chosen_rewards, rejected_rewards = self.get_batch_loss_metrics(self.model, batch, train_eval="ece")
                        chosen.extend(chosen_rewards.tolist())
                        reject.extend(rejected_rewards.tolist())
                ece = set_temperature(chosen, reject, self.temperature, script_args.output_dir)
                log_value = self.temperature.detach().cpu().item()
                wandb.log({'temperature_trajectory': self.beta})
                wandb.log({'ece': ece})
                self.temperature = nn.Parameter((torch.ones(1)*1).to(device))
            # Increment the counter
            self.eval_step_counter += 1
        print("now calling eval")
        # Now call the original evaluate function
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval", "ece"] = "train",
    ):
        """Compute the DPO losfs and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            _
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if train_eval == "ece":
            return chosen_rewards, rejected_rewards

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics
    


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    def format_messages(example, tokenizer):
        # Ensure we keep the "messages" key with a list of dictionaries
        formatted_messages = messages_dict = {"messages": [{"role": msg["role"], "content": msg["content"]} for msg in example["messages"]]}
        print(formatted_messages)
        
        # Apply the chat template if necessary and return the correct structure
        return {"messages": maybe_apply_chat_template(formatted_messages, tokenizer=tokenizer)}


    dataset = load_dataset(args.dataset_name)

    with PartialState().local_main_process_first():
        dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)

        # Apply the custom function to the dataset
        dataset['train'] = dataset['train_prefs'].map(
            lambda x: format_messages(x, tokenizer),
            num_proc=training_args.dataset_num_proc
        )

        dataset['test'] = dataset['test_prefs'].map(
            lambda x: format_messages(x, tokenizer),
            num_proc=training_args.dataset_num_proc
        )

        

    ########## 
    # Training
    ################
    training_args.save_strategy = "epoch"
    training_args.save_total_limit=4

    trainer = ECEDP0Trainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model(training_args.output_dir)