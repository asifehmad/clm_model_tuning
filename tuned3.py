#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import logging
import math
import os
import random
from itertools import chain

import datasets
import hydra
import torch
import transformers
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

import bittensor

deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=4)

def check_cfg_and_load_defaults(cfg: DictConfig) -> DictConfig:

    subtensor = bittensor.subtensor(network=cfg.bittensor.network)
    if cfg.dataset.block_size is None:
        cfg.dataset.block_size = subtensor.validator_sequence_length
    if cfg.training.train_batch_size is None:
        cfg.training.train_batch_size = subtensor.validator_batch_size
    if cfg.training.eval_batch_size is None:
        cfg.training.eval_batch_size = subtensor.validator_batch_size

    return cfg


# def create_accelerator(cfg: DictConfig) -> Accelerator:

#     accelerator = (
#         Accelerator(log_with=cfg.tracking.report_to, logging_dir=cfg.output_dir)
#         if cfg.tracking.enabled
#         else Accelerator(fp16=True, deepspeed_plugin=deepspeed_plugin)
#     )
#     if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
#         transformers.utils.logging.set_verbosity_info()
#     else:
#         datasets.utils.logging.set_verbosity_error()
#         transformers.utils.logging.set_verbosity_error()

#     return accelerator


def load_raw_datasets(cfg: DictConfig) -> DatasetDict:

    if cfg.dataset.name == "bittensor":

        dataset = bittensor.dataset(
            no_tokenizer=True,
            batch_size=cfg.training.train_batch_size,
            block_size=cfg.dataset.block_size,
        )
        dataloader = dataset.dataloader(cfg.dataset.num_batches)
        bittensor_dataset = {"text": []}
        for batch in tqdm(dataloader, desc="Loading data from bittensor IPFS"):
            bittensor_dataset["text"].extend(batch)
        raw_datasets = Dataset.from_dict(bittensor_dataset)

        dataset.close()  # Avoid leaving threadqueue running.
        return raw_datasets

    if os.path.exists(cfg.dataset.name):
        data_files = {"text": cfg.dataset.name}
        dataset_args = {}

        extension = os.path.splitext(cfg.dataset.name)[-1].lstrip(".")

        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = cfg.dataset.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        raw_datasets = raw_datasets["text"]
    else:
        raw_datasets = load_dataset(cfg.dataset.name, cfg.dataset.config_name)

    return raw_datasets


def load_model_and_tokenizer(cfg: DictConfig):

    if cfg.model.config_name is not None:
        config = AutoConfig.from_pretrained(cfg.model.config_name)
    else:
        config = AutoConfig.from_pretrained(cfg.model.name)

    if cfg.tokenizer.name is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.name, use_fast=cfg.tokenizer.use_fast
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name, use_fast=cfg.tokenizer.use_fast
        )
    #tokenizer.pad_token = cfg.tokenizer.pad_token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        from_tf=bool(".ckpt" in cfg.model.name),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


#def create_optimizer(cfg, model):

 #   no_decay = ["bias", "LayerNorm.weight"]
  #  optimizer_grouped_parameters = [
     #   {
     #       "params": [
       #         p
         #       for n, p in model.named_parameters()
         #       if not any(nd in n for nd in no_decay)
          #  ],
          #  "weight_decay": cfg.training.weight_decay,
       # },
       # {
           # "params": [
             #   p
              #  for n, p in model.named_parameters()
               # if any(nd in n for nd in no_decay)
            #],
           # "weight_decay": 0.0,
       # },
    #]
    #return (torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.training.learning_rate)
         #  if accelerator.state.deepspeed_plugin is None
         #  or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
          # else DummyOptim
   # )


def preprocess(cfg, accelerator, tokenizer, raw_datasets):

    # First we tokenize all the texts.
    column_names = raw_datasets.column_names
    text_column_name = "text" if "text" in column_names else column_names["train"][0]
    if cfg.dataset.concatenate_raw is True:
        pad = False
    else:
        pad = "max_length"

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= cfg.dataset.block_size:
            total_length = (
                total_length // cfg.dataset.block_size
            ) * cfg.dataset.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + cfg.dataset.block_size]
                for i in range(0, total_length, cfg.dataset.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_fn(examples):
        result = tokenizer(
            examples[text_column_name],
            padding=pad,
            truncation=True,
            max_length=cfg.dataset.block_size,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():

        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            num_proc=cfg.tokenizer.preprocessing_num_workers,
            load_from_cache_file=not cfg.dataset.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if cfg.dataset.concatenate_raw is True:
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=cfg.tokenizer.preprocessing_num_workers,
                load_from_cache_file=not cfg.dataset.overwrite_cache,
                desc=f"Grouping texts in chunks of {cfg.dataset.block_size}",
            )

    return tokenized_datasets
  

# New Code #
def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


# New Code #
def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict
    return (epoch, last_global_step)


# New Code #
def evaluate(cfg, model, eval_dataloader, accelerator, eval_dataset):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(cfg.training.eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss
  

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    cfg = check_cfg_and_load_defaults(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    accelerator = (
        Accelerator(log_with=cfg.tracking.report_to, logging_dir=cfg.output_dir)
        if cfg.tracking.enabled
        else Accelerator(fp16=True, deepspeed_plugin=deepspeed_plugin)
        #else Accelerator()
    )
    #accelerator = (
       # Accelerator(log_with=cfg.tracking.report_to, logging_dir=cfg.output_dir) if cfg.tracking.enabled else Accelerator()
    #)
    #accelerator = create_accelerator(cfg)
    accelerator.wait_for_everyone()





   
    if cfg.training.seed is not None:
        logger.info(f"Setting random seed to {cfg.training.seed}")
        set_seed(cfg.training.seed)

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))
    
    tokenizer, model = load_model_and_tokenizer(cfg)
    #optimizer = create_optimizer(cfg, model)

# Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=cfg.training.learning_rate)


    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
        )




   # lr_scheduler = get_scheduler(
     #   name=cfg.training.lr_scheduler,
     #   optimizer=optimizer,
      #  num_warmup_steps=cfg.training.lr_warmup_steps,
       # num_training_steps=cfg.training.max_train_steps,
   # )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Load and preprocess data
    raw_datasets = load_raw_datasets(cfg)
    tokenized_datasets = preprocess(cfg, accelerator, tokenizer, raw_datasets)
    if "train" not in tokenized_datasets.column_names:
        tokenized_datasets = tokenized_datasets.train_test_split(
            test_size=cfg.training.val_split_percent / 100
        )
        tokenized_datasets_test_valid = tokenized_datasets["test"].train_test_split(
            test_size=0.5
        )
        tokenized_datasets["test"] = tokenized_datasets_test_valid["train"]
        tokenized_datasets["validation"] = tokenized_datasets_test_valid["test"]

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        ex = train_dataset[index]
        logger.info(f"Sample {index} of the training set: {ex}: \n")
        logger.info(tokenizer.decode(ex["input_ids"]))

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=cfg.training.train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=cfg.training.eval_batch_size,
    )

    
    # New Code
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        cfg.training.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = cfg.training.num_epochs * num_update_steps_per_epoch
    else:
        cfg.training.num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)
    
    
    # Prepare everything using our accelerator
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
    
    
    
 # Figure out how many steps we should save the Accelerator states
    if hasattr(cfg.checkpointing_steps, "isdigit"):
        checkpointing_steps = cfg.checkpointing_steps
        if cfg.training.checkpoint.checkpointing_steps.isdigit():
            checkpointing_steps = int(cfg.training.checkpoint.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if cfg.with_tracking:
        experiment_config = vars(cfg)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        _, last_global_step = load_training_checkpoint(
            model,
            cfg.training.checkpoint.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {cfg.training.checkpoint.resume_from_checkpoint}")
        resume_step = last_global_step
        starting_epoch = resume_step // len(train_dataloader)
        resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, cfg.training.num_epochs):
        model.train()
        if cfg.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if cfg.training.checkpoint.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if cfg.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / cfg.training.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if cfg.output_dir is not None:
                        output_dir = os.path.join(cfg.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= cfg.training.max_train_steps:
                break

        perplexity, eval_loss = evaluate(cfg, model, eval_dataloader, accelerator, eval_dataset)
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if cfg.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        # New Code #
        # Save the DeepSpeed checkpoint to the specified path
        checkpoint_model(cfg.output_dir, epoch, model, epoch, completed_steps)

        # New Code #
        # Tracks the best checkpoint and best metric
        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(cfg.output_dir, str(epoch))
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

    # New Code #
    # Loads the best checkpoint after the training is finished
    if cfg.load_best_model:
        _, last_global_step = load_training_checkpoint(
            model,
            "/".join(best_metric_checkpoint.split("/")[:-1]),
            tag=best_metric_checkpoint.split("/")[-1],
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )

    # New Code #
    # Evaluates using the best checkpoint
    perplexity, eval_loss = evaluate(cfg, model, eval_dataloader, accelerator, eval_dataset)
    logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")
    if perplexity != best_metric:
        raise AssertionError(
            f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
        )

    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        # New Code #
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.save_pretrained(
            cfg.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(cfg.output_dir)
            if cfg.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        with open(os.path.join(cfg.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)


if __name__ == "__main__":
    main()
    
    
#     # Scheduler and math around the number of training steps.
#     overrode_max_train_steps = False
#     num_update_steps_per_epoch = math.ceil(
#         len(train_dataloader) / cfg.training.gradient_accumulation_steps
#     )
#     if cfg.training.max_train_steps is None:
#         cfg.training.max_train_steps = (
#             cfg.training.num_epochs * num_update_steps_per_epoch
#         )
#         overrode_max_train_steps = True

#     # We need to recalculate our total training steps as the size of the training dataloader
#     # may have changed.
#     num_update_steps_per_epoch = math.ceil(
#         len(train_dataloader) / cfg.training.gradient_accumulation_steps
#     )
#     if overrode_max_train_steps:
#         cfg.training.max_train_steps = (
#             cfg.training.num_epochs * num_update_steps_per_epoch
#         )
#     # Afterwards we recalculate our number of training epochs
#     cfg.training.num_epochs = math.ceil(
#         cfg.training.max_train_steps / num_update_steps_per_epoch
#     )

#     # We need to initialize the trackers we use, and also store our configuration.
#     # We initialize the trackers only on main process because `accelerator.log`
#     # only logs on main process and we don't want empty logs/runs on other processes.
#     if cfg.tracking.enabled is True and accelerator.is_main_process:
#         experiment_config = vars(cfg)
#         # TensorBoard cannot log Enums, need the raw value
#         experiment_config["lr_scheduler_type"] = experiment_config[
#             "lr_scheduler_type"
#         ].value
#         accelerator.init_trackers("finetune_using_clm", experiment_config)

#     logger.info("***** Running training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num Epochs = {cfg.training.num_epochs}")
#     logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
#     logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")

#     # Only show the progress bar once on each machine.
#     progress_bar = tqdm(
#         range(cfg.training.max_train_steps),
#         disable=not accelerator.is_local_main_process,
#     )

#     completed_steps = 0
#     starting_epoch = 0

#     # Potentially load in the weights and states from a previous save
#     if cfg.training.checkpoint.resume_from_checkpoint > 0:
#         accelerator.print(
#             f"Resumed from checkpoint: {cfg.training.checkpoint.resume_from_checkpoint}"
#         )
#         accelerator.load_state(cfg.training.checkpoint.resume_from_checkpoint)
#         path = os.path.basename(cfg.training.checkpoint.resume_from_checkpoint)
#         training_difference = os.path.splitext(path)[0]

#         if "epoch" in training_difference:
#             starting_epoch = int(training_difference.replace("epoch_", "")) + 1
#             resume_step = None
#         else:
#             resume_step = int(training_difference.replace("step_", ""))
#             starting_epoch = resume_step // len(train_dataloader)
#             resume_step -= starting_epoch * len(train_dataloader)

#     for epoch in range(starting_epoch, cfg.training.num_epochs):
#         model.train()
#         if cfg.tracking.enabled is True:
#             total_loss = 0
#         train_losses = []
#         for step, batch in enumerate(train_dataloader):
#             # We need to skip steps until we reach the resumed step
#             if (
#                 cfg.training.checkpoint.resume_from_checkpoint
#                 and epoch == starting_epoch
#             ):
#                 if resume_step is not None and step < resume_step:
#                     completed_steps += 1
#                     continue

#             outputs = model(**batch)
#             loss = outputs.loss
#             train_losses.append(
#                 accelerator.gather(loss.repeat(cfg.training.train_batch_size))
#             )
#             # We keep track of the loss at each epoch
#             if cfg.tracking.enabled is True:
#                 total_loss += loss.detach().float()
#             loss = loss / cfg.training.gradient_accumulation_steps
#             accelerator.backward(loss)

#             if (
#                 step % cfg.training.gradient_accumulation_steps == 0
#                 or step == len(train_dataloader) - 1
#             ):
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()
#                 progress_bar.update(1)
#                 completed_steps += 1

#             if step % cfg.training.eval_every == 0:
#                 train_losses_tensor = torch.cat(train_losses)
#                 train_loss = torch.mean(train_losses_tensor)
#                 model.eval()
#                 eval_losses = []
#                 for _eval_step, eval_batch in enumerate(eval_dataloader):
#                     with torch.no_grad():
#                         outputs = model(**eval_batch)

#                     loss = outputs.loss
#                     eval_losses.append(
#                         accelerator.gather(loss.repeat(cfg.training.eval_batch_size))
#                     )

#                 losses = torch.cat(eval_losses)
#                 losses = losses[: len(eval_dataset)]
#                 try:
#                     eval_loss = torch.mean(losses)
#                     perplexity = math.exp(eval_loss)
#                 except OverflowError:
#                     perplexity = float("inf")

#                 logger.info(
#                     f"epoch {epoch}: perplexity: {perplexity} train_loss: {train_loss} eval_loss: {eval_loss}"
#                 )

#                 epoch_dir = f"epoch_{epoch}_most_recent"
#                 if cfg.output_dir is not None:
#                     output_dir = os.path.join(cfg.output_dir, epoch_dir)
#                 unwrapped_model = accelerator.unwrap_model(model)
#                 unwrapped_model.save_pretrained(
#                     output_dir,
#                     is_main_process=accelerator.is_main_process,
#                     save_function=accelerator.save,
#                     state_dict=accelerator.get_state_dict(model),
#                 )
#                 if accelerator.is_main_process:
#                     tokenizer.save_pretrained(output_dir)

#         if cfg.tracking.enabled is True:
#             accelerator.log(
#                 {
#                     "perplexity": perplexity,
#                     "eval_loss": eval_loss,
#                     "train_loss": total_loss.item() / len(train_dataloader),
#                     "epoch": epoch,
#                     "step": completed_steps,
#                 },
#                 step=completed_steps,
#             )

#         logger.info(f"done epoch {epoch}")

#     if cfg.output_dir is not None:
#         accelerator.wait_for_everyone()
#         unwrapped_model = accelerator.unwrap_model(model)
#         unwrapped_model.save_pretrained(
#             cfg.output_dir,
#             is_main_process=accelerator.is_main_process,
#             save_function=accelerator.save,
#             state_dict=accelerator.get_state_dict(model),
#         )
#         if accelerator.is_main_process:
#             tokenizer.save_pretrained(cfg.output_dir)
    
#     #print('Pushing Model weights and other related files to Hugging Face Hub')
#     #model.push_to_hub(cfg.output_dir) 
#     #print('Pushing the Tokenizer and related files to Hugging Face Hub')
#     #tokenizer.push_to_hub(cfg.output_dir)

# if __name__ == "__main__":
#     main()
