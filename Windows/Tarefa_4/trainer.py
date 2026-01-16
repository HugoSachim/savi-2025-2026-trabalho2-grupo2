import os
import platform
import subprocess
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):
        # Guardar argumentos
        self.args = args
        self.model = model
        
        # Detetar número de classes (default 10 se não existir)
        self.num_classes = self.args.get('num_classes', 10)

        # Dataloaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

        # Configurar Loss e Optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)

        # --- NOVO: Ativar modo interativo para gráficos em tempo real ---
        plt.ion() 
        self.fig = plt.figure(1) # Cria a janela uma vez

        # Inicializar WandB (se configurado)
        try:
            wandb.init(
                project="mnist_savi",
                name=self.args.get('experiment_full_name', 'experiment'),
                config=self.args,
                mode="offline"  # Podes mudar para "online" se tiveres conta
            )
        except:
            print("Aviso: WandB não iniciado.")

        # Resumo ou Início Limpo
        if self.args.get('resume_training', False):
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    def train(self):
        print('Training started. Max epochs = ' + str(self.args['num_epochs']))

        for i in range(self.epoch_idx, self.args['num_epochs'] + 1):
            self.epoch_idx = i
            print('\nEpoch index = ' + str(self.epoch_idx))

            # =========================================
            # TRAIN
            # =========================================
            self.model.train()
            train_batch_losses = []
            
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                # Mover para GPU
                device = next(self.model.parameters()).device
                image_tensor = image_tensor.to(device)
                label_gt_tensor = label_gt_tensor.to(device)

                # Forward
                label_pred_tensor = self.model(image_tensor)

                # FIX ROBUSTO DE DIMENSÕES
                # Transforma (Batch, 11, 1, 1) em (Batch, 11)
                label_pred_tensor = label_pred_tensor.flatten(start_dim=1)

                # Calcular Loss
                batch_loss = self.loss(label_pred_tensor, label_gt_tensor.argmax(dim=1))
                train_batch_losses.append(batch_loss.item())

                # Backprop
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            # =========================================
            # TEST
            # =========================================
            self.model.eval()
            test_batch_losses = []

            with torch.no_grad():
                for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
                    device = next(self.model.parameters()).device
                    image_tensor = image_tensor.to(device)
                    label_gt_tensor = label_gt_tensor.to(device)

                    label_pred_tensor = self.model(image_tensor)
                    
                    # FIX ROBUSTO NO TESTE
                    label_pred_tensor = label_pred_tensor.flatten(start_dim=1)

                    batch_loss = self.loss(label_pred_tensor, label_gt_tensor.argmax(dim=1))
                    test_batch_losses.append(batch_loss.item())

            # =========================================
            # LOGS E FIM DE ÉPOCA
            # =========================================
            print('Finished epoch ' + str(i) + ' out of ' + str(self.args['num_epochs']))

            train_epoch_loss = np.mean(train_batch_losses) if train_batch_losses else 0
            test_epoch_loss = np.mean(test_batch_losses) if test_batch_losses else 0

            self.train_epoch_losses.append(train_epoch_loss)
            self.test_epoch_losses.append(test_epoch_loss)

            # Logs WandB
            try:
                wandb.log({"train_loss": train_epoch_loss, "test_loss": test_epoch_loss, "epoch": self.epoch_idx})
            except: pass

            # Save & Draw
            try:
                self.log_epoch_metrics(self.epoch_idx)
                self.draw()
                self.saveTrain()
            except Exception as e:
                print(f"Erro não crítico ao salvar/desenhar: {e}")

        print('Training completed.')

    def log_epoch_metrics(self, epoch_idx):
        self.model.eval()
        gt_classes = []
        predicted_classes = []

        # Usar apenas uma amostra do teste para ser rápido, ou tudo se for pequeno
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images = images.to(device)
                
                logits = self.model(images).flatten(start_dim=1) # Fix dimensões aqui também
                pred = torch.softmax(logits, dim=1).argmax(dim=1)
                gt = labels.argmax(dim=1)
                
                gt_classes.extend(gt.cpu().tolist())
                predicted_classes.extend(pred.cpu().tolist())

        # Matriz de Confusão dinâmica (suporta 11 classes)
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for gt, pred in zip(gt_classes, predicted_classes):
            if gt < self.num_classes and pred < self.num_classes:
                confusion_matrix[gt][pred] += 1

        # Plot
        plt.figure(figsize=(10, 8))
        seaborn.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Epoch {epoch_idx}")
        
        try:
            wandb.log({"confusion_matrix": wandb.Image(plt), "epoch": epoch_idx})
        except: pass
        
        plt.close()

    def draw(self):
        plt.figure(1)
        plt.clf() # Limpar figura atual
        
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Define limites para o gráfico não ficar a saltar
        offset_graph = 0.1
        all_losses = self.train_epoch_losses + self.test_epoch_losses
        max_loss = max(all_losses) if all_losses else 2.5 # Default 2.5 se vazio
        
        plt.ylim([0, max_loss * (1 + offset_graph)])
        
        # Plot Train
        if len(self.train_epoch_losses) > 0:
            plt.plot(range(len(self.train_epoch_losses)), self.train_epoch_losses, 'r-', label='Train', linewidth=2)
        
        # Plot Test
        if len(self.test_epoch_losses) > 0:
            plt.plot(range(len(self.test_epoch_losses)), self.test_epoch_losses, 'b-', label='Test', linewidth=2)
            
            # Melhor ponto
            best_idx = np.argmin(self.test_epoch_losses)
            plt.axvline(x=best_idx, color='g', linestyle='--', label='Best')

        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3) # Grelha ajuda a ver valores

        # --- ATUALIZAÇÃO EM TEMPO REAL ---
        plt.draw()
        plt.pause(0.01) # Pausa mínima para o sistema desenhar
        
        # Salvar imagem na mesma
        try:
            save_dir = self.args.get('experiment_full_name', '.')
            img_path = os.path.join(save_dir, 'training.png')
            plt.savefig(img_path)
        except: pass

    def saveTrain(self):
        checkpoint = {
            'epoch_idx': self.epoch_idx,
            'train_epoch_losses': self.train_epoch_losses,
            'test_epoch_losses': self.test_epoch_losses,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        save_dir = self.args.get('experiment_full_name', '.')
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pkl'))

        if self.test_epoch_losses[-1] == min(self.test_epoch_losses):
            torch.save(checkpoint, os.path.join(save_dir, 'best.pkl'))

    def loadTrain(self):
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)

        checkpoint = torch.load(checkpoint_file)
        self.epoch_idx = checkpoint['epoch_idx'] + 1
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Resuming training from epoch', self.epoch_idx)

    def open_file(self, path):
        # Método auxiliar mantido para compatibilidade, caso uses noutro lado
        try:
            if platform.system() == "Windows": os.startfile(path)
            elif platform.system() == "Darwin": subprocess.call(["open", path])
            else: subprocess.call(["xdg-open", path])
        except: pass
