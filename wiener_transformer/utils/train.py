import os
import csv
import sys
import yaml
from os.path import exists, join

from accelerate import Accelerator
import GPUtil
import nltk
import torch
from torch.optim.lr_scheduler import LambdaLR

from wiener_transformer.utils.loss import (
    LabelSmoothing,
    SimpleLossCompute,
    AutomaticWeightedLoss
)
from wiener_transformer.utils.helpers import (
    Batch,
    TrainState,
    rate,
    run_epoch,
    DummyOptimizer,
    DummyScheduler,
    calculate_bleu
)
from wiener_transformer.utils.data_loader import create_dataloaders
from wiener_transformer.transformer import make_model
from wiener_transformer.utils.vocab import load_wmt, load_tokenizers
from wiener_transformer.utils.wienerloss import WienerLoss


def train_model(
    train,
    valid,
    test,
    src_tokenizer,
    tgt_tokenizer,
    config,
    accelerator=None,
    model=None,
    optimizer=None,
    lr_scheduler=None
):
    """
    Train the model using the specified configuration and dataset.

    Args:
        train: Training dataset.
        valid: Validation dataset.
        test: Test dataset.
        src_tokenizer: Source tokenizer.
        tgt_tokenizer: Target tokenizer.
        config: Configuration dictionary.
        accelerator: Accelerator object for distributed training (optional).
        model: Preloaded model (optional).
        optimizer: Preloaded optimizer (optional).
        lr_scheduler: Preloaded learning rate scheduler (optional).
    """

    if config.get("distributed", False):
        device = accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.get("automatic_weighted_loss", False):
        awl = AutomaticWeightedLoss(2)
    else:
        awl = None

    d_model = config.get("d_model", 512)
    N = config.get("N", 6)
    embedding_type = config.get("embedding_type", "learned")
    pad_idx = 2
    wiener_condition = config.get("wiener_loss", False)
    gamma = config.get("gamma", 0.1)
    eps = config.get("eps", 1e-5)
    loss_weight = config.get("loss_weight", 1)

    src_vocab, tgt_vocab = src_tokenizer.get_vocab(), tgt_tokenizer.get_vocab()

    if model is None:
        model = make_model(len(src_vocab), len(tgt_vocab), N=N, d_model=d_model,
                           embedding_type=embedding_type).to(device)

    kldiv_fn = LabelSmoothing(
        size=len(tgt_vocab), padding_idx=pad_idx, smoothing=0.1
    )

    if wiener_condition:
        wiener_fn = WienerLoss(filter_dim=1, epsilon=eps, rel_epsilon=True)
    else:
        wiener_fn = None

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        train,
        valid,
        test,
        src_tokenizer,
        tgt_tokenizer,
        device,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"]
        )

    if optimizer is None:
        parameters = list(model.parameters())
        if awl is not None:
            parameters += list(awl.parameters())
        optimizer = torch.optim.Adam(
            parameters, lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
        )

    if lr_scheduler is None:
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, d_model, factor=1, warmup=config["warmup"]
            ),
        )

    if config["distributed"]:
        model, optimizer, train_dataloader, lr_scheduler, kldiv_fn, wiener_fn, awl = (
            accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler, kldiv_fn, wiener_fn, awl
            )
        )
        module = model.module
    else:
        module = model
    
    train_state = TrainState()

    if not config["distributed"] or accelerator.is_main_process:
        with open(
            os.path.join(
                os.environ["experiments_dir"],
                config["file_prefix"],
                "bleu.csv"
            ),
            "w",
            newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", 
                "corpus_bleu", 
                "sentence_bleu", 
                "sacrebleu_score", 
                "recall", 
                "precision", 
                "f1"
            ])

        with open(
            os.path.join(
                os.environ["experiments_dir"],
                config["file_prefix"],
                "loss.csv"
            ),
            "w",
            newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", 
                "train_total_loss", 
                "train_wiener_loss", 
                "train_kldiv_loss", 
                "val_total_loss", 
                "val_wiener_loss", 
                "val_kldiv_loss"
            ])

    for epoch in range(config["num_epochs"]):
        model.train()
        if not config["distributed"] or accelerator.is_main_process:
            print(f"Epoch {epoch} Training ====", flush=True)
            
        train_loss, train_wiener_loss, train_kldiv_loss, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            wiener_fn,
            SimpleLossCompute(module.generator, kldiv_fn),
            optimizer,
            lr_scheduler,
            loss_weight=loss_weight,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
            accelerator=accelerator if config["distributed"] else None,
            max_batches=float("inf"),
            gamma=gamma,
            awl=awl
        )

        if config["distributed"]:
            accelerator.wait_for_everyone()
        
        if not config["distributed"] or accelerator.is_main_process:
            GPUtil.showUtilization()
            module = model.module if config["distributed"] else model
            checkpoints_dir = os.path.join(
                os.environ["experiments_dir"],
                config["file_prefix"],
                "checkpoints"
            )
            os.makedirs(checkpoints_dir, exist_ok=True)
            file_path = os.path.join(checkpoints_dir, f"epoch_{epoch}.pt")
            state_dict = (
                accelerator.get_state_dict(model)
                if config["distributed"]
                else model.state_dict()
            )
            torch.save(state_dict, file_path) 
            print(f"Epoch {epoch} Validation ====", flush=True)
            model.eval()

        val_loss, val_wiener_loss, val_kldiv_loss, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            wiener_fn,
            SimpleLossCompute(module.generator, kldiv_fn),
            DummyOptimizer(),
            DummyScheduler(),
            loss_weight=loss_weight,
            mode="eval",
            accelerator=accelerator if config["distributed"] else None,
            max_batches=float("inf"),
            awl=awl
        )

        if config["distributed"]:
            accelerator.wait_for_everyone()

        if not config["distributed"] or accelerator.is_main_process:
            print(f"Validation Loss: {val_loss}")

            sentence_bleu_score, corpus_bleu_score, sacrebleu_score, recall, precision, f1 = calculate_bleu(
                module, test_dataloader, pad_idx, tgt_tokenizer, max_len=50
            )
    
            with open(
                os.path.join(
                    os.environ["experiments_dir"],
                    config["file_prefix"],
                    "bleu.csv"
                ),
                "a",
                newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    corpus_bleu_score,
                    sentence_bleu_score,
                    sacrebleu_score,
                    recall,
                    precision,
                    f1
                ])
    
            with open(
                os.path.join(
                    os.environ["experiments_dir"],
                    config["file_prefix"],
                    "loss.csv"
                ),
                "a",
                newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    train_loss, 
                    train_wiener_loss if config["wiener_loss"] else "nan", 
                    train_kldiv_loss, 
                    val_loss, 
                    val_wiener_loss if config["wiener_loss"] else "nan", 
                    val_kldiv_loss
                ])

    if not config["distributed"] or accelerator.is_main_process:
        file_path = os.path.join(os.environ["experiments_dir"],
                                 config["file_prefix"], "final_model.pt")
        state_dict = (
            accelerator.get_state_dict(model)
            if config["distributed"]
            else model.state_dict()
        )
        torch.save(state_dict, file_path) 
        print("Final model saved")

    if config["distributed"]:
        accelerator.wait_for_everyone()


def load_trained_model(config):
    """
    Load a trained model from disk. If no trained model exists, train it first.

    Args:
        config: Configuration dictionary.

    Returns:
        Trained model.
    """

    accelerator = Accelerator() if config["distributed"] else None

    if config["distributed"]:
        with accelerator.main_process_first():        
            train, valid, test = load_wmt(("en", "de"))
            src_tokenizer, tgt_tokenizer = load_tokenizers(train)
            src_vocab, tgt_vocab = src_tokenizer.get_vocab(), tgt_tokenizer.get_vocab()
    else:
        train, valid, test = load_wmt(("en", "de"))
        src_tokenizer, tgt_tokenizer = load_tokenizers(train)
        src_vocab, tgt_vocab = src_tokenizer.get_vocab(), tgt_tokenizer.get_vocab()

    if config.get("from_checkpoint", False) and config.get("checkpoint_dir"):
        checkpoint_dir = config["checkpoint_dir"]
        model_path = os.path.join(checkpoint_dir)
        
        if exists(model_path):
            print(f"Loading model from {model_path}")
            N = config.get("N", 6)
            d_model = config.get("d_model", 512)
            embedding_type = config.get("embedding_type", "learned")
            model = make_model(
                len(src_vocab),
                len(tgt_vocab),
                N=N,
                d_model=d_model,
                embedding_type=embedding_type
            )
            unwrapped_model = (
                accelerator.unwrap_model(model)
                if config["distributed"]
                else model
            )
            unwrapped_model.load_state_dict(torch.load(model_path))
            print("Model loaded from checkpoint")

            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
            )

            lr_scheduler = LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda step: rate(
                    step, d_model, factor=1, warmup=config["warmup"]
                ),
            )
        else:
            print("Checkpoint not found or inaccessible, training from scratch")
            model, optimizer, lr_scheduler = None, None, None
    else:
        print("Not loading from checkpoint, training from scratch")
        model, optimizer, lr_scheduler = None, None, None

    train_model(
        train,
        valid,
        test,
        src_tokenizer,
        tgt_tokenizer,
        config,
        accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    return model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_config_yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    os.environ["base_dir"] = config.get(
        "base_dir", "/scratch_brain/acd23/code/irp-acd23"
    )
    os.environ["experiments_dir"] = os.path.join(
        os.environ["base_dir"],
        "experiments/wiener_loss"
    )

    nltk.data.path.append(os.environ["base_dir"])

    os.environ["TORCH_CUDA_CACHE_DIR"] = os.path.join(
        os.environ["base_dir"], "/.cache/torch/kernels"
    )
    os.environ["PYTORCH_KERNEL_CACHE_PATH"] = os.path.join(
        os.environ["base_dir"], "/.cache/torch/kernels"
    )

    load_trained_model(config)
