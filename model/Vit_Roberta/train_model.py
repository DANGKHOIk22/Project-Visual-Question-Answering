import torch
import torch.nn as nn
from Vit_Roberta_model import VQAModel,VisualEncoder,TextEncoder,Classifier
from data_loader import get_dataloaders  
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, criterion, device):

    model.eval()
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for images, questions, labels in dataloader:
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total

    return avg_loss, accuracy

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []
        model.train()

        for images, questions, labels in train_loader:
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        # Compute average training loss
        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        # Evaluate on validation data
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"[EPOCH {epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Update learning rate scheduler
        scheduler.step()

    return train_losses, val_losses

def model_pipeline():
    """
    Manages the full model training pipeline: loading data, initializing model, training, and evaluating.
    """
    lr = 1e-3
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders()
    classes=2
    # Initialize Model
    n_classes = len(classes)
    hidden_size = 256
    dropout_prob = 0.2

    text_encoder = TextEncoder().to(device)
    visual_encoder = VisualEncoder().to(device)
    classifier = Classifier(
        hidden_size=hidden_size,
        dropout_prob=dropout_prob,
        n_classes=n_classes
    ).to(device)

    model = VQAModel(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        classifier=classifier
    ).to(device)
    model.freeze()

    # Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler_step_size = int(epochs * 0.8)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    # Train Model
    train_losses, val_losses = fit(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, device, epochs
    )

    print("Training complete!")

    return train_losses, val_losses

if __name__ == "__main__":
    seed = 59
    set_seed(seed)
    model_pipeline()
