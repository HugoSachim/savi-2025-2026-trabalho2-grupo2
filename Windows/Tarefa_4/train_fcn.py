import argparse
import torch
from dataset import Dataset
from model import ModelFCN
from trainer import Trainer

def main():
    # Definições
    parser = argparse.ArgumentParser()
    # Aponta para a pasta onde o script anterior gerou os dados
    parser.add_argument('--dataset_folder', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Windows/Tarefa_4/mixed_dataset') 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='FCN_Experiment')
    args = parser.parse_args()

    # Configuração
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Dicionário de argumentos para o Trainer/Dataset
    args_dict = vars(args)
    args_dict['percentage_examples'] = 0.2
    args_dict['experiment_full_name'] = f"experiments/{args.experiment_name}"
    
    # --- IMPORTANTE: Definir 11 classes (0-9 + Fundo) ---
    args_dict['num_classes'] = 11 

    # Carregar Dados
    # Nota: O dataset.py precisa de ser ajustado se tiver hardcoded [0]*10. 
    # Se não quiseres mexer no dataset.py, terás erro de dimensão aqui.
    # Vê a nota abaixo sobre o dataset.py.
    train_set = Dataset(args_dict, is_train=True)
    test_set  = Dataset(args_dict, is_train=False) # Nota: precisas de gerar test set também ou usar split

    # Inicializar Modelo FCN
    model = ModelFCN().to(device)

    # Iniciar Treino
    trainer = Trainer(args_dict, train_set, test_set, model)
    trainer.train()

if __name__ == '__main__':
    main()