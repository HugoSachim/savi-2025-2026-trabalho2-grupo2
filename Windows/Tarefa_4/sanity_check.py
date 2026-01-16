import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob
import random
from model import ModelFCN

def main():
    # --- CONFIGURAÇÃO ---
    model_path = 'experiments/FCN_Experiment/best.pkl'
    dataset_dir = r"C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Windows/Tarefa_4/mixed_dataset/train/images"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Carregar Modelo
    print(f"[INFO] A carregar modelo de: {model_path}")
    if not os.path.exists(model_path):
        print("ERRO: Ficheiro do modelo não encontrado!")
        return

    model = ModelFCN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Escolher uma imagem de treino que NÃO seja fundo (classes 0-9)
    # Vamos ler o labels.txt para garantir que pegamos num número
    labels_file = os.path.join(os.path.dirname(dataset_dir), "labels.txt")
    print(f"[INFO] A procurar um dígito real em: {labels_file}")
    
    digit_images = []
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                label = int(parts[1])
                if label < 10: # É um dígito (0-9)
                    digit_images.append((filename, label))
    
    if not digit_images:
        print("ERRO: Não encontrei dígitos no labels.txt!")
        return

    # Escolher um aleatório
    chosen_file, true_label = random.choice(digit_images)
    image_path = os.path.join(dataset_dir, chosen_file)

    print(f"\n[TESTE] Imagem escolhida: {chosen_file}")
    print(f"[TESTE] Label Verdadeira: {true_label}")

    # 3. Inferência
    img = Image.open(image_path).convert('L')
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        # O output pode ser (1, 11, 1, 1) ou (1, 11) dependendo da versão do forward
        if output.dim() == 4:
            output = output.view(1, -1)
            
        probs = F.softmax(output, dim=1)
        score, pred_class = torch.max(probs, dim=1)

    print(f"\n--- RESULTADOS ---")
    print(f"Classe Prevista: {pred_class.item()}")
    print(f"Confiança:       {score.item():.4f}")
    
    if pred_class.item() == true_label:
        print("\n✅ SUCESSO! O modelo reconhece este dígito.")
    else:
        print(f"\n❌ FALHA! O modelo errou (Previu {pred_class.item()}, era {true_label}).")
        if pred_class.item() == 10:
            print("   -> O modelo acha que o dígito é FUNDO (Overfitting ao background).")

    img.show()

if __name__ == "__main__":
    main()