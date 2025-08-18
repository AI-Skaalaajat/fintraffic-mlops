import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import random
from PIL import Image
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import yaml
import argparse
from model import create_road_classifier

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# --- Dataset Prep ---
class RoadDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"[Corrupt] {filepath} - {e}")
        return False

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0

def configure_mlflow_training():
    """Configure MLflow for training environment."""
    # Set MLflow tracking URI for training
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service.mlflow.svc.cluster.local:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Configure S3 endpoint for artifact storage
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://minio-service.kubeflow.svc.cluster.local:9000")
    
    print(f"MLflow training configured with tracking URI: {tracking_uri}")

def train(config, device, model_output_path=None):
    """The main training and validation loop."""
    # Configure MLflow
    configure_mlflow_training()
    # --- Load and encode all data ---
    image_paths, labels = [], []
    class_names = sorted(os.listdir(config['data']['data_path']))

    for class_name in class_names:
        folder = os.path.join(config['data']['data_path'], class_name)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if is_valid_image(path):
                image_paths.append(path)
                labels.append(class_name)
            else:
                print(f"Skipped corrupt file: {file}")

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    NUM_CLASSES = len(le.classes_)

    print(f"Classes: {', '.join(le.classes_)}")
    print(f"Class distribution: {Counter(labels)}")

    # --- Train/Test split ---
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels_encoded, 
        test_size=config['data']['test_size'], 
        stratify=labels_encoded, 
        random_state=config['data']['random_state']
    )
    
    # --- Transforms ---
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['training']['img_size'], config['training']['img_size'])),
        transforms.RandomHorizontalFlip(config['augmentation']['horizontal_flip_prob']),
        transforms.RandomRotation(degrees=config['augmentation']['rotation_degrees']),
        transforms.ColorJitter(
            brightness=config['augmentation']['brightness_range'], 
            contrast=config['augmentation']['contrast_range'], 
            saturation=config['augmentation']['saturation_range']
        ),
        transforms.RandomAffine(
            degrees=0, 
            translate=config['augmentation']['translate'], 
            scale=config['augmentation']['scale']
        ),
        transforms.ToTensor(),
        transforms.Normalize(config['normalization']['mean'], config['normalization']['std']),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['training']['img_size'], config['training']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(config['normalization']['mean'], config['normalization']['std']),
    ])

    # --- Dataloaders ---
    train_loader = DataLoader(
        RoadDataset(train_paths, train_labels, transform_train), 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        RoadDataset(val_paths, val_labels, transform_val), 
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    # --- Model ---
    model = create_road_classifier(
        NUM_CLASSES, 
        pretrained_path=config['model'].get('pretrained_path')
    )
    model = model.to(device)

    # --- Class weights ---
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = [total / label_counts[i] for i in range(NUM_CLASSES)]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = FocalLoss(alpha=class_weights_tensor, gamma=config['loss']['focal_loss_gamma'])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=config['scheduler']['mode'], 
        factor=config['scheduler']['factor'], 
        patience=config['scheduler']['patience']
    )

    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta']
    )

    # --- Training Loop ---
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log hyperparameters
        mlflow.log_params({
            "img_size": config['training']['img_size'],
            "batch_size": config['training']['batch_size'],
            "epochs": config['training']['epochs'],
            "learning_rate": config['training']['learning_rate'],
            "weight_decay": config['training']['weight_decay'],
            "optimizer": "AdamW",
            "loss_function": "FocalLoss",
            "scheduler": "ReduceLROnPlateau",
            "train_samples": len(train_paths),
            "val_samples": len(val_paths)
        })
        
        mlflow.set_tag("classes", ", ".join(le.classes_))

        best_val_acc = 0.0
        
        for epoch in range(config['training']['epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{config['training']['epochs']}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # --- Training Phase ---
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            train_preds_all = []
            train_labels_all = []

            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                train_preds_all.extend(preds.cpu().numpy())
                train_labels_all.extend(labels.cpu().numpy())

            train_acc = train_correct / train_total
            avg_train_loss = running_loss / len(train_loader)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # --- Validation Phase ---
            model.eval()
            val_correct = 0
            val_total = 0
            val_preds_all = []
            val_labels_all = []
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    
                    val_preds_all.extend(preds.cpu().numpy())
                    val_labels_all.extend(labels.cpu().numpy())

            val_acc = val_correct / val_total
            print(f"Validation Acc: {val_acc:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            }, step=epoch)

            scheduler.step(val_acc)
            
            early_stopping(val_acc)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Log the best model to MLflow
                mlflow.pytorch.log_model(model, "model")
                print(f"New best model logged to MLflow with validation accuracy: {best_val_acc:.4f}")
                
                # Save the model to the specified output path for KFP
                if model_output_path:
                    os.makedirs(model_output_path, exist_ok=True)
                    model_save_path = os.path.join(model_output_path, "model.pth")
                    torch.save(model, model_save_path)
                    print(f"New best model saved for KFP artifact at: {model_save_path}")

        # --- Post-Training Analysis ---
        print("\nLogging artifacts...")
        report = classification_report(val_labels_all, val_preds_all, target_names=le.classes_)
        cm = confusion_matrix(val_labels_all, val_preds_all)
        
        # Save artifacts to KFP output path if provided, otherwise save locally for MLflow
        if model_output_path:
            report_path = os.path.join(model_output_path, "classification_report.txt")
            cm_path = os.path.join(model_output_path, "confusion_matrix.txt")
            
            with open(report_path, "w") as f:
                f.write(report)
            with open(cm_path, "w") as f:
                f.write(np.array2string(cm))
                
            mlflow.log_artifact(report_path)
            mlflow.log_artifact(cm_path)
            print(f"Artifacts for KFP saved to {model_output_path}")

        else:
            # Original behavior: save locally, log to MLflow, then delete
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")
            os.remove("classification_report.txt")
            
            with open("confusion_matrix.txt", "w") as f:
                f.write(np.array2string(cm))
            mlflow.log_artifact("confusion_matrix.txt")
            os.remove("confusion_matrix.txt")

        mlflow.log_metric("best_val_acc", best_val_acc)
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        
        # --- Auto-register model to MLflow Model Registry ---
        print("\nRegistering model to MLflow Model Registry...")
        try:
            # Register the logged model to the Model Registry
            model_name = "dry_wet_model"
            model_uri = f"runs:/{run.info.run_id}/model"
            
            # Register the model
            registered_model = mlflow.register_model(model_uri, model_name)
            print(f"Model registered: {model_name}, Version: {registered_model.version}")
            
            # Transition the model to Production stage
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Production",
                archive_existing_versions=True  # Archive previous Production versions
            )
            print(f"Model version {registered_model.version} transitioned to Production stage")
            
            # Add model description
            client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=f"Dry/Wet road classifier trained with validation accuracy: {best_val_acc:.4f}"
            )
            
            print("✅ Model successfully registered and set to Production!")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to register model to MLflow Model Registry: {e}")
            print("The training completed successfully, but model registration failed.")
            print("You may need to manually register the model or check MLflow configuration.")


def main():
    """Main function to run the training pipeline."""
    # --- Configuration and Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a road condition classifier for dry/wet surfaces.")
    parser.add_argument(
        '--config-path', 
        type=str, 
        default="config.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=None,
        help="Path to the training data directory. Overrides the path in config.yaml."
    )
    parser.add_argument(
        '--model-output-path',
        type=str,
        default=None,
        help="Path to save the trained model and artifacts for Kubeflow."
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    # Override data_path if provided via command-line
    if args.data_path:
        config['data']['data_path'] = args.data_path

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config['random_seed'])
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # --- Run Training ---
    train(config, device, args.model_output_path)

    
if __name__ == '__main__':
    main() 