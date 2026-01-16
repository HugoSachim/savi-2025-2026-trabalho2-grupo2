import glob
import os
import torch
from PIL import Image
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):
        self.args = args
        self.train = is_train

        # 1. Construir caminho das imagens
        split_name = 'train' if is_train else 'test'
        # Usar os.path.join para garantir caminhos corretos no Windows
        self.dataset_path = os.path.join(args['dataset_folder'], split_name)
        image_folder = os.path.join(self.dataset_path, 'images')

        print(f"[{split_name.upper()}] A procurar imagens em: {image_folder}")

        # 2. Carregar lista de imagens (suporta jpg e png)
        # glob recursivo ou direto para garantir que apanha os ficheiros
        self.image_filenames = glob.glob(os.path.join(image_folder, "*.jpg"))
        
        # Se não encontrar jpg, tenta png
        if len(self.image_filenames) == 0:
             self.image_filenames = glob.glob(os.path.join(image_folder, "*.png"))

        self.image_filenames.sort()
        
        count = len(self.image_filenames)
        print(f"[{split_name.upper()}] Imagens encontradas: {count}")

        if count == 0:
            raise ValueError(f"CRÍTICO: Nenhuma imagem encontrada em {image_folder}. Verifica se geraste o dataset corretamente!")

        # 3. Carregar Labels
        self.labels_filename = os.path.join(self.dataset_path, 'labels.txt')
        self.labels = []

        if not os.path.exists(self.labels_filename):
             raise FileNotFoundError(f"Labels não encontradas: {self.labels_filename}")

        print(f"[{split_name.upper()}] A ler labels de: {self.labels_filename}")
        
        with open(self.labels_filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # O último elemento é a classe, o resto ignora-se
                    label = float(parts[-1]) 
                    self.labels.append(label)

        # 4. Ajustar percentagem de dados
        percentage = args.get('percentage_examples', 1.0)
        num_examples = int(round(len(self.image_filenames) * percentage))
        
        # Proteção para não ficar com 0
        if num_examples < 1 and len(self.image_filenames) > 0:
            num_examples = 1

        self.image_filenames = self.image_filenames[0:num_examples]
        self.labels = self.labels[0:num_examples]

        # Verificar consistência
        if len(self.labels) != len(self.image_filenames):
            print(f"[AVISO] Diferença entre Imagens ({len(self.image_filenames)}) e Labels ({len(self.labels)}).")
            # Ajusta pelo menor
            min_len = min(len(self.labels), len(self.image_filenames))
            self.image_filenames = self.image_filenames[:min_len]
            self.labels = self.labels[:min_len]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Carregar Label (One-Hot Encoding)
        label_index = int(self.labels[idx])
        num_classes = self.args.get('num_classes', 11) # Default 11 para FCN
        
        label = [0.0] * num_classes
        if label_index < num_classes:
            label[label_index] = 1.0
        
        label_tensor = torch.tensor(label, dtype=torch.float)

        # Carregar Imagem
        image_filename = self.image_filenames[idx]
        image = Image.open(image_filename).convert('L')
        image_tensor = self.to_tensor(image)

        return image_tensor, label_tensor
    
