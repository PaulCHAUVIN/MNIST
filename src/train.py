import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
from models import SimpleConvNet
from config_loader import load_config
import logging
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformer_package.models import ViT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(config_path: str):
    config = load_config(config_path)

    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=config['batch_size'], shuffle=False)

    # Model, Loss, and Optimizer setup
    if config['model'] == 'SimpleConvNet':
        model = SimpleConvNet()

    elif config['model'] == 'ViT':
        model = ViT(image_size = 28, channel_size = 1, patch_size = 7, embed_size = 512, num_heads = 8, classes = 10, num_layers = 3, hidden_size = 256, dropout = 0.2)
    
    else:
        raise ValueError(f"Model {config['model']} not supported")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Lists to track metrics
    train_losses = []
    val_losses = []

    with mlflow.start_run():
        # Training loop
        model.train()
        for epoch in range(config['max_epochs']):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    logger.info(f"Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # Average training loss for the epoch
            train_losses.append(running_loss / len(train_loader.dataset)) 
            mlflow.log_metric('train_loss', train_losses[-1], step=epoch)

            # Validation loop
            model.eval()
            val_loss = 0
            correct = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    all_preds.extend(pred.squeeze().tolist())
                    all_targets.extend(target.tolist())
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            accuracy = accuracy_score(all_targets, all_preds) 
            f1 = f1_score(all_targets, all_preds, average='macro')
            precision = precision_score(all_targets, all_preds, average='macro')
            recall = recall_score(all_targets, all_preds, average='macro')

            logger.info(f"Epoch {epoch}): Validation loss: {val_loss:.4f}, Training loss: {train_losses[-1]:.4f}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

            # Log metrics using MLflow
            mlflow.log_metric('validation_loss', val_loss, step=epoch)
            mlflow.log_metric('accuracy', accuracy, step=epoch)
            mlflow.log_metric('f1_score', f1, step=epoch)
            mlflow.log_metric('precision', precision, step=epoch)
            mlflow.log_metric('recall', recall, step=epoch)

        # Log and save the model using MLFlow
        mlflow.log_params(config)

        # Save only the model's state dictionary to a local file
        model_path = "model_state_dict.pth"
        torch.save(model.state_dict(), model_path)

        # Log the saved model state dictionary with MLflow
        mlflow.log_artifact(model_path, artifact_path="models")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network on the MNIST dataset.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (YAML format).')
    args = parser.parse_args()
    train_model(args.config)