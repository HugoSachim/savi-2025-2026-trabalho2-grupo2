import glob
import os
import torch
from PIL import Image
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):
        self.args = args
        self.train = is_train

        split_name = 'train' if is_train else 'test'
        self.dataset_path = os.path.join(args['dataset_folder'], split_name)
        image_folder = os.path.join(self.dataset_path, 'images')
        labels_file = os.path.join(self.dataset_path, 'labels.txt')

        print(f"[{split_name.upper()}] A carregar dataset de: {self.dataset_path}")

        # 1. Carregar o Mapa de Labels (Filename -> Class)
        # Isto garante que não há desfasamento, mesmo que os ficheiros estejam desordenados
        self.label_map = {}
        if os.path.exists(labels_file):
            with open(labels_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # parts[0] é o nome do ficheiro (ex: patch_123.jpg)
                        # parts[1] é a classe
                        filename = parts[0]
                        label_class = int(float(parts[1]))
                        self.label_map[filename] = label_class
        else:
            raise FileNotFoundError(f"Ficheiro de labels não encontrado: {labels_file}")

        # 2. Listar Imagens
        # Vamos carregar apenas as imagens que estão listadas no labels.txt
        # Isto evita carregar imagens "fantasmas" ou sem label
        self.image_filenames = []
        self.labels = []
        
        # Procurar ficheiros físicos para validar
        # Criamos um set de ficheiros existentes na pasta para ser rápido
        existing_files = set(os.listdir(image_folder))
        
        for filename, label in self.label_map.items():
            if filename in existing_files:
                full_path = os.path.join(image_folder, filename)
                self.image_filenames.append(full_path)
                self.labels.append(label)
        
        # Opcional: Ordenar para consistência entre runs (mas agora labels e imagens estão sincronizados)
        # Vamos zipar, ordenar e unzipar
        if len(self.image_filenames) > 0:
            zipped = sorted(zip(self.image_filenames, self.labels))
            self.image_filenames, self.labels = zip(*zipped)

        count = len(self.image_filenames)
        print(f"[{split_name.upper()}] Total de pares imagem-label validados: {count}")

        if count == 0:
            raise ValueError(f"Nenhuma imagem válida encontrada em {image_folder} com correspondência em {labels_file}")

        # 3. Reduzir dataset se pedido (percentagem)
        percentage = args.get('percentage_examples', 1.0)
        if percentage < 1.0:
            num_examples = int(round(count * percentage))
            self.image_filenames = self.image_filenames[:num_examples]
            self.labels = self.labels[:num_examples]
            print(f"[{split_name.upper()}] Reduzido para {len(self.image_filenames)} exemplos ({percentage*100}%)")

        if is_train:
            # Aumentação de dados APENAS no treino
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),      # Roda aleatoriamente +/- 15 graus
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Pequenos desvios e zoom
                transforms.ToTensor(),
                # Opcional: Adicionar ruído se necessário, mas para MNIST isto chega
            ])
        else:
            # No teste, queremos a imagem original limpa
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Carregar Label
        label_index = int(self.labels[idx])
        
        # Lidar com número de classes (default 11 para FCN)
        num_classes = self.args.get('num_classes', 11)
        
        # Proteção: Se a label for inválida (ex: erro no txt), força Fundo (10)
        if label_index >= num_classes:
            label_index = 10 
            
        # One-Hot Encoding
        label = [0.0] * num_classes
        label[label_index] = 1.0
        label_tensor = torch.tensor(label, dtype=torch.float)

        # Carregar Imagem
        image_filename = self.image_filenames[idx]
        image = Image.open(image_filename).convert('L')
        
        # APLICAR A TRANSFORMAÇÃO DEFINIDA NO INIT
        image_tensor = self.transform(image)

        return image_tensor, label_tensor
    
