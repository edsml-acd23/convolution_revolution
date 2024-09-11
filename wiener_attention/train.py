import os
import torch
import sys
import yaml

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from wiener_attention.data import load_imdb, create_dataloaders
from wiener_attention.model import make_bert_model, make_bert_tokenizer
from wiener_attention.attention_mechanism import WienerSelfAttention
from wiener_attention.wiener_metric import WienerSimilarityMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prefix = "experiments/wiener_attention"


def train_model(model, train_loader, val_loader, dir, learning_rate=2e-5, epochs=100):
    """
    Train the model using the provided data loaders and save results.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        dir (str): Directory to save results and checkpoints.
        learning_rate (float, optional): Learning rate for optimization. Defaults to 2e-5.
        epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        torch.nn.Module: The trained model.
    """
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    results = {
        'Epoch': [],
        'Training Loss': [],
        'Training Accuracy': [],
        'Training Precision': [],
        'Training Recall': [],
        'Validation Loss': [],
        'Validation Accuracy': [],
        'Validation Precision': [],
        'Validation Recall': []
    }

    checkpoints_dir = os.path.join(prefix, dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        train_precision = precision_score(all_labels, all_predictions, average='weighted')
        train_recall = recall_score(all_labels, all_predictions, average='weighted')

        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training Precision: {train_precision:.4f}")
        print(f"Training Recall: {train_recall:.4f}")

        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids, labels, attention_mask = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples += labels.size(0)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())

        val_accuracy = val_correct / val_samples
        avg_val_loss = val_loss / len(val_loader)
        val_precision = precision_score(all_val_labels, all_val_predictions, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_predictions, average='weighted')

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")

        results['Epoch'].append(epoch + 1)
        results['Training Loss'].append(avg_train_loss)
        results['Training Accuracy'].append(train_accuracy)
        results['Training Precision'].append(train_precision)
        results['Training Recall'].append(train_recall)
        results['Validation Loss'].append(avg_val_loss)
        results['Validation Accuracy'].append(val_accuracy)
        results['Validation Precision'].append(val_precision)
        results['Validation Recall'].append(val_recall)

        save_results(results, dir)

        checkpoint_path = os.path.join(checkpoints_dir, f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

    save_model(model, dir)

    return model


def save_results(results, dir):
    """
    Save training and validation results to a CSV file.

    Args:
        results (dict): Dictionary containing training and validation metrics.
        dir (str): Directory to save the results file.
    """
    results_df = pd.DataFrame(results)
    results_path = os.path.join(prefix, config["dir"], "results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


def save_model(model, dir):
    """
    Save the trained model to a file.

    Args:
        model (torch.nn.Module): The trained model to be saved.
        dir (str): Directory to save the model file.
    """
    model_path = os.path.join(prefix, dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    """
    Main execution block for training the model.
    """
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_config_yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    wiener = config.get("wiener_attention", False)
    dir = config.get("dir")

    if wiener:
        eps = config.get("eps", 1e-5)
        gamma = config.get("gamma", 0.1)
        wiener_similarity = WienerSimilarityMetric(filter_dim=1, epsilon=eps, rel_epsilon=True)
        wiener_attention = WienerSelfAttention
    else:
        wiener_similarity, wiener_attention, gamma = None, None, None

    epochs = config.get("epochs", 100)
    learning_rate = config.get("learning_rate", 2e-5)
    batch_size = config.get("batch_size", 32)
    max_padding = config.get("max_len", 32)

    train, val, test = load_imdb()
    tokenizer = make_bert_tokenizer()
    train_loader, val_loader, test_loader = create_dataloaders(
        train,
        val,
        test,
        tokenizer,
        device,
        batch_size=batch_size,
        max_padding=max_padding
    )

    model = make_bert_model(wiener_attention=wiener_attention, wiener_similarity=wiener_similarity, gamma=gamma)
    model = train_model(model, train_loader, val_loader, dir, learning_rate=learning_rate, epochs=epochs)
