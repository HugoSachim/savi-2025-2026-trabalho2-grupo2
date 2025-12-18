import glob
import os
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import requests
import seaborn
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm
import wandb
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import scrolledtext
import os
import platform
import subprocess

class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        # Storing arguments in class properties
        self.args = args
        self.model = model

        # Create the dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
        # For testing we typically set shuffle to false

        # Setup loss function
        self.loss = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        wandb.init(
            project="mnist_savi",
            name=self.args['experiment_full_name'],
            config=self.args
        )

        # Start from scratch or resume training
        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    def train(self):

        print('Training started. Max epochs = ' + str(self.args['num_epochs']))

        # -----------------------------------------
        # Iterate all epochs
        # -----------------------------------------
        for i in range(self.epoch_idx, self.args['num_epochs'] + 1):

            self.epoch_idx = i
            print('\nEpoch index = ' + str(self.epoch_idx))

            # -----------------------------------------
            # Train
            # -----------------------------------------
            self.model.train()
            train_batch_losses = []
            num_batches = len(self.train_dataloader)

            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.train_dataloader), total=num_batches):

                # Forward
                label_pred_tensor = self.model(image_tensor)

                # Cross-Entropy Loss (NO softmax)
                batch_loss = self.loss(label_pred_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())

                # Backprop
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            # -----------------------------------------
            # Test
            # -----------------------------------------
            self.model.eval()
            test_batch_losses = []
            num_batches = len(self.test_dataloader)

            with torch.no_grad():
                for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                        enumerate(self.test_dataloader), total=num_batches):

                    label_pred_tensor = self.model(image_tensor)
                    batch_loss = self.loss(label_pred_tensor, label_gt_tensor)
                    test_batch_losses.append(batch_loss.item())

            # ---------------------------------
            # End of epoch
            # ---------------------------------
            print('Finished epoch ' + str(i) + ' out of ' + str(self.args['num_epochs']))

            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            wandb.log({
                "train_loss": train_epoch_loss,
                "test_loss": test_epoch_loss,
                "epoch": self.epoch_idx
            })

            self.log_epoch_metrics(self.epoch_idx)
            self.draw()
            self.saveTrain()

        print('Training completed.')
        print('Training losses: ' + str(self.train_epoch_losses))
        print('Test losses: ' + str(self.test_epoch_losses))

    def loadTrain(self):
        print('Resuming training from last checkpoint.')

        # find the checkpoint file
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        print('checkpoint_file: ' + str(checkpoint_file))

        # Verify if file exists. If not abort. Cannot resume without the checkpoint.pkl
        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        print(checkpoint.keys())

        self.epoch_idx = checkpoint['epoch_idx']+1
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])  # contains the model's weights
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])  # contains the optimizer's

    def saveTrain(self):

        # Create the dictionary to save the checkpoint.pkl
        checkpoint = {}
        checkpoint['epoch_idx'] = self.epoch_idx
        checkpoint['train_epoch_losses'] = self.train_epoch_losses
        checkpoint['test_epoch_losses'] = self.test_epoch_losses

        checkpoint['model_state_dict'] = self.model.state_dict()  # contains the model's weights
        # contains the optimizer's state
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

        # Save the best.pkl
        if self.test_epoch_losses[-1] == min(self.test_epoch_losses):
            best_file = os.path.join(self.args['experiment_full_name'], 'best.pkl')
            torch.save(checkpoint, best_file)

    def draw(self):
        plt.figure(1)
        plt.clf()

        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        offset_graph = 0.1
        axis = plt.gca()
        axis.set_xlim([-offset_graph, len(self.train_epoch_losses) - 1 + offset_graph])
        # Calcular o máximo valor entre treino e teste
        all_losses = self.train_epoch_losses + self.test_epoch_losses
        if all_losses:
            max_loss = max(all_losses)
        else:
            max_loss = offset_graph  # valor default caso esteja vazio

        axis.set_ylim([0, max_loss * (1+offset_graph)])

        # plot training
        if len(self.train_epoch_losses) > 0:
            xs = range(len(self.train_epoch_losses))
            plt.plot(xs, self.train_epoch_losses, 'r-', linewidth=2)

        # plot testing
        if len(self.test_epoch_losses) > 0:
            xs = range(len(self.test_epoch_losses))
            plt.plot(xs, self.test_epoch_losses, 'b-', linewidth=2)

            # draw best checkpoint
            best_epoch_idx = int(np.argmin(self.test_epoch_losses))
            print('best_epoch_idx:', best_epoch_idx)
            plt.plot([best_epoch_idx, best_epoch_idx], [0, 0.5], 'g--', linewidth=1)

        plt.legend(['Train', 'Test', 'Best'], loc='upper right')

        # caminho do ficheiro
        img_path = os.path.join(self.args['experiment_full_name'], 'training.png')
        plt.savefig(img_path)

        # abrir o ficheiro da imagem automaticamente
        self.open_file(img_path)


    def evaluate(self):

        # -----------------------------------------
        # Iterate over test batches and compute the ground truth and predicted values for all examples
        # -----------------------------------------
        self.model.eval()  # set model to evaluation mode
        num_batches = len(self.test_dataloader)

        gt_classes = []
        predicted_classes = []

        for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                enumerate(self.test_dataloader), total=num_batches):

            # Ground truth
            batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()

            # Prediction
            logits = self.model(image_tensor)

            # Compute the probabilities using softmax
            probs = torch.softmax(logits, dim=1)
            batch_predicted_classes = probs.argmax(dim=1).tolist()

            gt_classes.extend(batch_gt_classes)
            predicted_classes.extend(batch_predicted_classes)

        # -----------------------------------------
        # Create confusion matrix
        # -----------------------------------------
        confusion_matrix = np.zeros((10, 10), dtype=int)
        for gt, pred in zip(gt_classes, predicted_classes):
            confusion_matrix[gt][pred] += 1

        # -----------------------------------------
        # Plot confusion matrix
        # -----------------------------------------
        plt.figure(2)
        class_names = [str(i) for i in range(10)]
        seaborn.heatmap(confusion_matrix,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=True,
                        xticklabels=class_names,
                        yticklabels=class_names)

        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted classes', fontsize=14)
        plt.ylabel('True classes', fontsize=14)
        plt.xticks(rotation=0, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()

        img_path_matrix = os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png')
        plt.savefig(img_path_matrix)

        # abrir o ficheiro da imagem automaticamente
        self.open_file(img_path_matrix)


        # -----------------------------------------
        # Compute statistics per class
        # -----------------------------------------
        statistics = {}
        per_class_precisions = []
        per_class_recalls = []
        per_class_f1 = []

        total_TP = 0
        total_FP = 0
        total_FN = 0

        for c in range(10):
            TP = int(confusion_matrix[c][c])
            FP = int(confusion_matrix[:, c].sum() - TP)
            FN = int(confusion_matrix[c, :].sum() - TP)

            precision, recall = self.getPrecisionRecall(TP, FP, FN)
            f1 = self.getF1(precision, recall)

            statistics[c] = {
                "digit": c,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            # For macro metrics
            per_class_precisions.append(precision if precision is not None else 0)
            per_class_recalls.append(recall if recall is not None else 0)
            per_class_f1.append(f1 if f1 is not None else 0)

            # For global (micro) metrics
            total_TP += TP
            total_FP += FP
            total_FN += FN

        # -----------------------------------------
        # Global (micro) metrics
        # -----------------------------------------
        global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else None
        global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else None
        global_f1 = self.getF1(global_precision, global_recall)

        statistics["global"] = {
            "precision": global_precision,
            "recall": global_recall,
            "f1_score": global_f1
        }

        # -----------------------------------------
        # Macro metrics (média simples das classes)
        # -----------------------------------------
        macro_precision = sum(per_class_precisions) / len(per_class_precisions)
        macro_recall = sum(per_class_recalls) / len(per_class_recalls)
        macro_f1 = sum(per_class_f1) / len(per_class_f1)

        statistics["macro"] = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1
        }

        print("Global metrics (micro):", statistics["global"])
        print("Macro metrics:", statistics["macro"])

        # -----------------------------------------
        # Save JSON
        # -----------------------------------------
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(statistics, f, indent=4)

        # --- depois de salvar o JSON ---
        json_imag_path = self.save_metrics_text(json_filename)
        self.open_file(json_imag_path)

        wandb.log({
            "final_confusion_matrix": wandb.Image(img_path_matrix)
        })


    def getPrecisionRecall(self, TP, FP, FN):

        precision = TP / (TP + FP) if (TP + FP) > 0 else None
        recall = TP / (TP + FN) if (TP + FN) > 0 else None

        return precision, recall


    def getF1(self, precision, recall):
        if precision is None or recall is None or (precision + recall == 0):
            return None
        return 2 * precision * recall / (precision + recall)


    def log_epoch_metrics(self, epoch_idx):

        # Avaliação rápida para obter preds/GT
        self.model.eval()
        gt_classes = []
        predicted_classes = []

        with torch.no_grad():
            for images, labels in self.test_dataloader:
                gt = labels.argmax(dim=1)
                pred = torch.softmax(self.model(images), dim=1).argmax(dim=1)
                gt_classes.extend(gt.tolist())
                predicted_classes.extend(pred.tolist())

        # Construir matriz de confusão
        confusion_matrix = np.zeros((10, 10), dtype=int)
        for gt, pred in zip(gt_classes, predicted_classes):
            confusion_matrix[gt][pred] += 1

        # Calcular métricas globais
        total_TP = sum(confusion_matrix[c][c] for c in range(10))
        total_FP = sum(confusion_matrix[:, c].sum() - confusion_matrix[c][c] for c in range(10))
        total_FN = sum(confusion_matrix[c, :].sum() - confusion_matrix[c][c] for c in range(10))

        precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
        recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Plot matriz de confusão
        plt.figure(figsize=(6, 6))
        seaborn.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Epoch {epoch_idx}")

        # Log para wandb
        wandb.log({
            "precision_global": precision,
            "recall_global": recall,
            "f1_global": f1,
            "confusion_matrix": wandb.Image(plt),
            "epoch": epoch_idx
        })

        plt.close()

    def save_metrics_text(self, json_file, output_path=None):
        # -----------------------------------------
        # Ler métricas do JSON
        # -----------------------------------------
        with open(json_file, 'r') as f:
            saved_stats = json.load(f)

        # -----------------------------------------
        # Construir o texto
        # -----------------------------------------
        display_text = ""
        for key in saved_stats:
            if key in ["global", "macro"]:
                display_text += f"{key.upper()} metrics:\n"
                display_text += f"  Precision: {saved_stats[key]['precision']:.4f}\n"
                display_text += f"  Recall   : {saved_stats[key]['recall']:.4f}\n"
                display_text += f"  F1-score : {saved_stats[key]['f1_score']:.4f}\n\n"
            else:
                display_text += f"Classe {saved_stats[key]['digit']}:\n"
                display_text += f"  TP: {saved_stats[key]['TP']}, FP: {saved_stats[key]['FP']}, FN: {saved_stats[key]['FN']}\n"
                display_text += f"  Precision: {saved_stats[key]['precision']:.4f}\n"
                display_text += f"  Recall   : {saved_stats[key]['recall']:.4f}\n"
                display_text += f"  F1-score : {saved_stats[key]['f1_score']:.4f}\n\n"

        # -----------------------------------------
        # Caminho para salvar o ficheiro de texto
        # -----------------------------------------
        if output_path is None:
            output_path = os.path.join(os.path.dirname(json_file), "metrics_summary.txt")

        with open(output_path, 'w') as f:
            f.write(display_text)

        print(f"Métricas salvas em ficheiro de texto: {output_path}")

        return output_path


    def open_file(self, path):
        try:
            system_name = platform.system()
            if system_name == "Windows":
                os.startfile(path)  # abre com programa padrão
            elif system_name == "Darwin":  # macOS
                subprocess.call(["open", path])
            else:  # Linux e outros
                subprocess.call(["xdg-open", path])
        except Exception as e:
            print("Não foi possível abrir a imagem:", e)
