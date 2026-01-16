import argparse
import torch
import torch.nn as nn
from dataset import Dataset
from model import ModelFCN
from trainer import Trainer
import os

def load_backbone_weights(fcn_model, task1_path):
    print(f"[INFO] A tentar carregar pesos da Tarefa 1 de: {task1_path}")
    
    if not os.path.exists(task1_path):
        print(f"[AVISO] Ficheiro {task1_path} não encontrado! A treinar do zero...")
        return fcn_model

    # Carregar checkpoint da Tarefa 1
    checkpoint = torch.load(task1_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Filtrar apenas as camadas convolucionais (conv1, bn1, conv2, etc.)
    # Ignoramos as camadas 'fc1', 'fc2' pois a FCN usa nomes diferentes ('fcn_conv')
    fcn_dict = fcn_model.state_dict()
    
    # Copiar pesos que tenham o mesmo nome e tamanho
    pretrained_dict = {k: v for k, v in state_dict.items() if k in fcn_dict and v.size() == fcn_dict[k].size()}
    
    # Atualizar modelo FCN
    fcn_dict.update(pretrained_dict)
    fcn_model.load_state_dict(fcn_dict)
    
    print(f"[SUCESSO] Transferidos {len(pretrained_dict)} parâmetros da Tarefa 1!")
    return fcn_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Windows/Tarefa_4/mixed_dataset') 
    parser.add_argument('--task1_model', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/datasets/savi_experiments/Tarefa_1_ModelBetterCNN/best.pkl', help="Caminho para o modelo da Tarefa 1")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5) # Menos épocas necessárias com transfer learning
    parser.add_argument('--experiment_name', type=str, default='FCN_Transfer_v2')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    args_dict = vars(args)
    args_dict['percentage_examples'] = 1
    args_dict['experiment_full_name'] = f"experiments/{args.experiment_name}"
    args_dict['num_classes'] = 11 

    # Datasets (Usa o dataset.py robusto!)
    train_set = Dataset(args_dict, is_train=True)
    test_set  = Dataset(args_dict, is_train=False)

    # 1. Inicializar Modelo
    model = ModelFCN()
    
    # 2. Transfer Learning (A MAGIA ACONTECE AQUI)
    model = load_backbone_weights(model, args.task1_model)
    
    model = model.to(device)

    # 3. Treinar
    trainer = Trainer(args_dict, train_set, test_set, model)
    trainer.train()

if __name__ == '__main__':
    main()