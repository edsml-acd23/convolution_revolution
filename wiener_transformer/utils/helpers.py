import os
import logging
from contextlib import nullcontext
import time

from tqdm import tqdm
import numpy as np
import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU
from evaluate import load


def calculate_bleu(model, test_dataloader, pad_idx, tgt_tokenizer, max_len=50):
    """
    Calculate BLEU scores and BERTScore for model predictions on the test dataset.

    Args:
        model: The trained model to be evaluated.
        test_dataloader: DataLoader providing batches of test data.
        pad_idx: The index used for padding in the sequences.
        tgt_tokenizer: Tokenizer for the target language.
        max_len: Maximum length for generated sequences (default: 50).

    Returns:
        Tuple containing:
            - corpus_bleu_score: BLEU score for the entire corpus.
            - average_sentence_bleu_score: Average BLEU score for individual sentences.
            - sacrebleu_score: SacreBLEU score for the corpus.
            - recall: Average BERTScore recall.
            - precision: Average BERTScore precision.
            - f1: Average BERTScore F1 score.
    """
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)
    model.eval()
    all_references = []
    all_hypotheses = []
    all_references_untokenized = []
    all_hypotheses_untokenized = []
    total_sentence_bleu = 0.0
    num_sentences = 0
    smoothing_function = SmoothingFunction().method1

    nltk.download("punkt", download_dir=os.environ.get("base_dir", "/scratch_brain/acd23/code/irp-acd23"))
    nltk.data.path.append(os.environ.get("base_dir", "/scratch_brain/acd23/code/irp-acd23"))

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            src_sentences, tgt_sentences = batch
            
            src_mask = (src_sentences != pad_idx).unsqueeze(-2)
            src_sentences = src_sentences.to("cuda")
            src_mask = src_mask.to("cuda")
            
            memory = model.encode(src_sentences, src_mask)
            
            start_token = tgt_tokenizer.token_to_id("<s>")
            end_token = tgt_tokenizer.token_to_id("</s>")
            ys = torch.ones(src_sentences.size(0), 1).fill_(start_token).type_as(src_sentences)
            
            for i in range(max_len - 1):
                tgt_mask = (torch.triu(torch.ones((ys.size(1), ys.size(1)), device=ys.device)) == 1).transpose(0, 1)
                tgt_mask = tgt_mask.unsqueeze(0)
                
                out = model.decode(memory, src_mask, ys, tgt_mask)
                
                prob = model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.unsqueeze(1)
                
                ys = torch.cat([ys, next_word], dim=1)

            trimmed_ys = []
            for seq in ys:
                trimmed_seq = []
                for token in seq:
                    if token.item() == end_token:
                        break
                    trimmed_seq.append(token.item())
                trimmed_ys.append(trimmed_seq)

            hypotheses = [tgt_tokenizer.decode(y, skip_special_tokens=True) for y in trimmed_ys]
            tgt_sentences_str = [tgt_tokenizer.decode(tgt_sentence.cpu().numpy(), skip_special_tokens=True) for tgt_sentence in tgt_sentences]

            for hypothesis, tgt_sentence_str in zip(hypotheses, tgt_sentences_str):
                hypothesis = hypothesis.strip()
                tgt_sentence_str = tgt_sentence_str.strip()
                all_hypotheses_untokenized.append(hypothesis)
                all_references_untokenized.append(tgt_sentence_str)
                reference_tokens = tgt_sentence_str.split()
                hypothesis_tokens = hypothesis.split()

                if hypothesis_tokens and reference_tokens:
                    all_hypotheses.append(hypothesis_tokens)
                    all_references.append([reference_tokens])

                    sentence_bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)
                    total_sentence_bleu += sentence_bleu_score
                    num_sentences += 1

    corpus_bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothing_function) * 100
    average_sentence_bleu_score = (total_sentence_bleu / num_sentences) * 100 if num_sentences > 0 else 0

    sacre_bleu = BLEU()
    sacrebleu_score = sacre_bleu.corpus_score(all_hypotheses_untokenized, [all_references_untokenized]).score

    bertscore = load("bertscore", cache_dir=os.environ.get("base_dir", "/scratch_brain/acd23"))
    results = bertscore.compute(predictions=all_hypotheses_untokenized, references=all_references_untokenized, lang="de")
    recall = np.mean(results["recall"]).item()
    precision = np.mean(results["precision"]).item()
    f1 = np.mean(results["f1"]).item()

    return corpus_bleu_score, average_sentence_bleu_score, sacrebleu_score, recall, precision, f1


class TrainState:
    """Track the number of steps, examples, and tokens processed during training.

    Attributes:
        step (int): Number of steps in the current epoch.
        accum_step (int): Number of gradient accumulation steps.
        samples (int): Total number of examples used.
        tokens (int): Total number of tokens processed.
    """
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class Batch:
    """Object for holding a batch of data with masks during training.

    Args:
        src: Source sequences for the batch.
        tgt: Target sequences for the batch (default: None).
        pad: Index used for padding (default: 2).

    Attributes:
        src: Source sequences for the batch.
        src_mask: Mask indicating non-padding elements in the source sequences.
        tgt: Target sequences without the last token (teacher forcing).
        tgt_y: Target sequences without the first token (teacher forcing).
        tgt_mask: Mask indicating non-padding elements and future positions in the target sequences.
        ntokens: Total number of non-padding tokens in the target sequences.
    """
    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words in the target sequences.

        Args:
            tgt: Target sequences.
            pad: Index used for padding.

        Returns:
            Tensor representing the mask for the target sequences.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    

class DummyOptimizer(torch.optim.Optimizer):
    """A dummy optimizer that does nothing. Used for testing or evaluation without updates.

    Methods:
        step(): Placeholder method to simulate an optimizer step.
        zero_grad(set_to_none=False): Placeholder method to simulate zeroing gradients.
    """
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    """A dummy learning rate scheduler that does nothing. Used for testing or evaluation without updates.

    Methods:
        step(): Placeholder method to simulate a scheduler step.
    """
    def step(self):
        None
    

def subsequent_mask(size):
    """
    Create a mask to hide subsequent positions in a sequence.

    Args:
        size: The length of the sequence.

    Returns:
        Tensor of shape (1, size, size) with True values in the upper triangular part.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def run_epoch(
    data_iter,
    model,
    wiener_fn,
    classic_loss,
    optimizer,
    scheduler,
    accelerator,
    mode="train",
    loss_weight=1,
    accum_iter=1,
    train_state=TrainState(),
    max_batches=float("inf"),
    gamma=0.1,
    awl=None
):
    """
    Train or evaluate the model for a single epoch.

    Args:
        data_iter: DataLoader providing batches of data.
        model: The model being trained or evaluated.
        wiener_fn: Function to compute the Wiener loss (if applicable).
        classic_loss: Function to compute the standard loss (e.g., cross-entropy).
        optimizer: Optimizer used for training.
        scheduler: Learning rate scheduler.
        accelerator: Distributed training utility (e.g., from Hugging Face's Accelerate).
        mode: Mode of operation, either "train" or "eval" (default: "train").
        loss_weight: Weighting factor for the loss function (default: 1).
        accum_iter: Number of steps for gradient accumulation (default: 1).
        train_state: Object tracking training state (default: TrainState()).
        max_batches: Maximum number of batches to process (default: inf).
        gamma: Regularization parameter for the Wiener loss (default: 0.1).
        awl: Automatic weighting of losses (if applicable).

    Returns:
        Tuple containing:
            - avg_loss: Average combined loss over the epoch.
            - avg_wiener: Average Wiener loss over the epoch.
            - avg_kldiv: Average Kullback-Leibler divergence loss over the epoch.
            - train_state: Updated training state after the epoch.
    """
    total_tokens = 0
    total_loss = 0
    total_wiener = 0
    total_kldiv = 0
    tokens = 0
    n_accum = 0
    batch_count = 0

    context_manager = torch.no_grad() if mode == "eval" else nullcontext()
    start = time.time()
    with context_manager:
        for i, batch in enumerate(data_iter):
            if i >= max_batches:
                break
            
            out = model.forward(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            if wiener_fn is not None:
                if accelerator is not None:
                    true = model.module.tgt_embed(batch.tgt_y)
                else:
                    true = model.tgt_embed(batch.tgt_y)
                wiener = wiener_fn.forward(out.view(-1, 512), true.view(-1, 512), gamma=gamma)
            else:
                wiener = 0
            _, kldiv_node = classic_loss(out, batch.tgt_y, batch.ntokens)
            if awl is None:
                kldiv_node = loss_weight*kldiv_node
                loss = wiener + kldiv_node
            else:
                loss, losses = awl(wiener, kldiv_node)
                wiener, kldiv_node = losses

            if mode == "train" or mode == "train+log":
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens
                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                scheduler.step()

            total_loss += loss.item()
            if wiener_fn is not None:
                total_wiener += wiener.item()
            total_kldiv += kldiv_node.item()
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            batch_count += 1

            if i % 300 == 1 and (mode == "train" or mode == "train+log") and (accelerator is None or accelerator.is_main_process):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Combined Loss: %6.5f | Wiener Loss: %6.5f | KLDiv Loss: %6.5f"
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum,  loss.item(), wiener.item() if wiener_fn is not None else 0, kldiv_node.item(), tokens / elapsed, lr)
                )
                start = time.time()
                tokens = 0

            del loss

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_wiener = total_wiener / batch_count if batch_count > 0 else 0
    avg_kldiv = total_kldiv / batch_count if batch_count > 0 else 0

    return avg_loss, avg_wiener, avg_kldiv, train_state


def rate(step, model_size, factor, warmup):
    """
    Compute the learning rate according to the warmup strategy.

    Args:
        step: Current step number.
        model_size: Dimensionality of the model.
        factor: Scaling factor for the learning rate.
        warmup: Number of warmup steps.

    Returns:
        Learning rate value for the current step.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def pad(tensor, pad, value):
    """
    Pad a tensor with a specified value.

    Args:
        tensor: The tensor to be padded.
        pad: A tuple specifying the padding to be applied to each dimension.
        value: The value to use for padding.

    Returns:
        Padded tensor.
    """
    return torch.nn.functional.pad(tensor, pad, value=value)


def tokenize(text, tokenizer):
    """
    Tokenize a given text using a specified tokenizer.

    Args:
        text: The text to be tokenized.
        tokenizer: The tokenizer to be used for tokenization.

    Returns:
        List of tokens.
    """
    return [tok.text for tok in tokenizer.tokenizer(text)]
