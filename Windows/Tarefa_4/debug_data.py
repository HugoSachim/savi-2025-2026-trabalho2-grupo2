import torch
from dataset import Dataset
import matplotlib.pyplot as plt

def main():
    # Simula os argumentos do treino
    args = {
        'dataset_folder': 'C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Windows/Tarefa_4/mixed_dataset', # Ajusta o caminho se necessário
        'percentage_examples': 0.1,
        'num_classes': 11
    }
    
    print("A carregar dataset...")
    ds = Dataset(args, is_train=True)
    
    # Buscar 5 exemplos
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img_t, label_t = ds[i]
        
        # Converter tensor one-hot para número
        label_idx = torch.argmax(label_t).item()
        
        # Mostrar imagem
        axs[i].imshow(img_t.squeeze(), cmap='gray')
        axs[i].set_title(f"Label: {label_idx}")
        axs[i].axis('off')
        
    plt.show()
    print("Verifica se as imagens acima têm números visíveis e se a label faz sentido.")

if __name__ == "__main__":
    main()