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
        self.loss = nn.MSELoss()  # Mean Squared Error Loss

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
        for i in range(self.epoch_idx, self.args['num_epochs'] + 1):  # number of epochs

            self.epoch_idx = i
            print('\nEpoch index = ' + str(self.epoch_idx))
            # -----------------------------------------
            # Train - Iterate over batches
            # -----------------------------------------
            self.model.train()  # set model to training mode
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.train_dataloader), total=num_batches):  # type: ignore

                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # Update model
                self.optimizer.zero_grad()  # resets the gradients from previous batches
                batch_loss.backward()  # the actual backpropagation
                self.optimizer.step()

            # -----------------------------------------
            # Test - Iterate over batches
            # -----------------------------------------
            self.model.eval()  # set model to evaluation mode

            test_batch_losses = []
            num_batches = len(self.test_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.test_dataloader), total=num_batches):  # type: ignore
                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # During test there is no model update

            # ---------------------------------
            # End of the epoch training
            # ---------------------------------
            print('Finished epoch ' + str(i) + ' out of ' + str(self.args['num_epochs']))
            # print('batch_losses: ' + str(batch_losses))

            # update the training epoch losses
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            # update the testing epoch losses
            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            wandb.log({
                "train_loss": train_epoch_loss,
                "test_loss": test_epoch_loss,
                "epoch": self.epoch_idx
            })

            self.log_epoch_metrics(self.epoch_idx)

            # Draw the updated training figure
            self.draw()

            # Save the training state
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
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))
 

    def evaluate(self):

        # -----------------------------------------
        # Iterate over test batches and compute the ground trutch and predicted  values for all examples
        # -----------------------------------------
        self.model.eval()  # set model to evaluation mode
        num_batches = len(self.test_dataloader)

        self.gts = []  # list of ground truth labels
        self.preds = []  # list of predicted labels

        gt_classes = []
        predicted_classes = []

        for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                enumerate(self.test_dataloader), total=num_batches):

            # Ground truth
            batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()

            # Prediction
            logits = self.model.forward(image_tensor)

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
        title_conf_matrix = 'Confusion Matrix'
        seaborn.heatmap(confusion_matrix,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=True,
                        xticklabels=class_names,
                        yticklabels=class_names)

        plt.title(title_conf_matrix, fontsize=16)
        plt.xlabel('Predicted classes', fontsize=14)
        plt.ylabel('True classes', fontsize=14)
        plt.xticks(rotation=0, ha='right', fontsize=12)  # Rodar rótulos do X para melhor leitura
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png'))

        # -----------------------------------------
        # Compute statistics per class
        # -----------------------------------------
        statistics = {}
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

            # For global metrics
            if precision is not None and recall is not None:
                per_class_f1.append(f1)

            total_TP += TP
            total_FP += FP
            total_FN += FN

        # -----------------------------------------
        # Global metrics
        # -----------------------------------------

        # Macro = média simples das classes
        global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else None
        global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else None
        global_f1 = self.getF1(global_precision, global_recall)

        statistics["global"] = {
            "precision": global_precision,
            "recall": global_recall,
            "f1_score": global_f1
        }

        print("Global metrics:", statistics["global"])

        # -----------------------------------------
        # Save JSON
        # -----------------------------------------
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(statistics, f, indent=4)

        wandb.log({
            "final_confusion_matrix": wandb.Image(os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png'))
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
