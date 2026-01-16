import os
import glob
import random
import shutil
from PIL import Image
from tqdm import tqdm

# ================= CONFIGURAÇÕES =================
# Onde estão as imagens "Cenas" geradas na Tarefa 2 (INPUT)
SOURCE_IMAGES_DIR = r"C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_D/mnist_detection/train/images"
SOURCE_LABELS_DIR = r"C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_D/mnist_detection/train/labels"

# Onde será criado o novo dataset pronto a usar (OUTPUT)
OUTPUT_ROOT = r"C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Tarefa_4/mixed_dataset"

SPLIT_RATIO = 0.8       # 80% Treino, 20% Teste
BACKGROUND_RATIO = 0.2  # Quantidade de fundos vs dígitos
CROP_SIZE = 28          # Tamanho final das imagens (28x28)
# =================================================

def calculate_iou(box1, box2):
    """Calcula Intersection over Union para validar fundos."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return interArea / float(box1Area + box2Area - interArea)

def get_ground_truth(image_name, label_dir):
    """Lê labels ignorando cabeçalhos e erros."""
    base_name = os.path.splitext(image_name)[0]
    label_path = os.path.join(label_dir, base_name + ".txt")
    
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Ignora cabeçalhos ou linhas vazias
                if not line or line.lower().startswith('label') or line.lower().startswith('class'):
                    continue
                
                parts = line.split(',')
                if len(parts) < 5: parts = line.split()

                if len(parts) >= 5:
                    try:
                        cls = int(parts[0])
                        x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        gt_boxes.append({'class': cls, 'box': [x1, y1, x2, y2]})
                    except ValueError:
                        continue
    return gt_boxes

def main():
    # 1. Limpeza Inicial
    if os.path.exists(OUTPUT_ROOT):
        print(f"[INFO] A remover dataset antigo em {OUTPUT_ROOT}...")
        shutil.rmtree(OUTPUT_ROOT)
    
    # Criar pastas finais
    train_dir = os.path.join(OUTPUT_ROOT, "train")
    test_dir = os.path.join(OUTPUT_ROOT, "test")
    
    os.makedirs(os.path.join(train_dir, "images"))
    os.makedirs(os.path.join(test_dir, "images"))

    # 2. Listar Imagens Fonte
    image_paths = glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*.png")) + \
                  glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*.jpg"))
    image_paths.sort()
    
    print(f"[INFO] Imagens de origem encontradas: {len(image_paths)}")

    # Lista temporária para guardar TODOS os crops antes de dividir
    # Formato: {'image': PIL_Image_Object, 'filename': str, 'class': int}
    # Nota: Se tiveres pouca RAM (<8GB) e o dataset for gigante, avisa-me para ajustar isto.
    # Mas para MNIST e 28x28, guardar em memória é rápido e viável.
    all_crops_metadata = [] 

    # Como não podemos guardar 60.000 imagens PIL em RAM facilmente sem estourar,
    # vamos guardar numa pasta temporária "buffer" e depois mover.
    buffer_dir = os.path.join(OUTPUT_ROOT, "buffer")
    os.makedirs(buffer_dir, exist_ok=True)

    print("[INFO] A gerar crops (Dígitos + Fundo)...")
    crop_counter = 0

    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert('L')
            w_img, h_img = img.size
            img_filename = os.path.basename(img_path)
            gts = get_ground_truth(img_filename, SOURCE_LABELS_DIR)

            if not gts: continue

            # --- A. Dígitos (Positivos) ---
            for gt in gts:
                x1, y1, x2, y2 = gt['box']
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if x2 <= x1 or y2 <= y1: continue

                crop = img.crop((x1, y1, x2, y2)).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
                
                filename = f"patch_{crop_counter:06d}.jpg"
                save_path = os.path.join(buffer_dir, filename)
                crop.save(save_path)
                
                all_crops_metadata.append({'filename': filename, 'class': gt['class']})
                crop_counter += 1

            # --- B. Fundo (Negativos - Classe 10) ---
            num_bg = max(1, int(len(gts) * BACKGROUND_RATIO))
            attempts = 0
            generated = 0
            
            while generated < num_bg and attempts < 100:
                attempts += 1
                rnd_size = random.randint(22, 40)
                if w_img - rnd_size <= 0 or h_img - rnd_size <= 0: continue

                rnd_x = random.randint(0, w_img - rnd_size)
                rnd_y = random.randint(0, h_img - rnd_size)
                box = [rnd_x, rnd_y, rnd_x + rnd_size, rnd_y + rnd_size]

                # Verificar overlap
                overlap = False
                for gt in gts:
                    if calculate_iou(box, gt['box']) > 0.05:
                        overlap = True; break
                
                if not overlap:
                    crop = img.crop((box[0], box[1], box[2], box[3])).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
                    
                    filename = f"patch_{crop_counter:06d}.jpg"
                    save_path = os.path.join(buffer_dir, filename)
                    crop.save(save_path)

                    all_crops_metadata.append({'filename': filename, 'class': 10})
                    crop_counter += 1
                    generated += 1

        except Exception as e:
            print(f"Erro na imagem {img_path}: {e}")

    # 3. Baralhar e Dividir
    print(f"\n[INFO] Total de crops gerados: {len(all_crops_metadata)}")
    print("[INFO] A baralhar e criar Train/Test splits...")
    
    random.seed(42)
    random.shuffle(all_crops_metadata)

    split_idx = int(len(all_crops_metadata) * SPLIT_RATIO)
    train_data = all_crops_metadata[:split_idx]
    test_data = all_crops_metadata[split_idx:]

    # 4. Mover ficheiros e criar labels.txt finais
    def process_set(dataset_list, folder_name):
        dest_img_path = os.path.join(OUTPUT_ROOT, folder_name, "images")
        dest_txt_path = os.path.join(OUTPUT_ROOT, folder_name, "labels.txt")
        
        print(f" -> A preencher {folder_name.upper()} ({len(dataset_list)} imagens)...")
        
        with open(dest_txt_path, 'w') as f:
            for item in tqdm(dataset_list):
                src = os.path.join(buffer_dir, item['filename'])
                dst = os.path.join(dest_img_path, item['filename'])
                
                # Mover ficheiro
                shutil.move(src, dst)
                
                # Escrever no txt (Apenas nome e classe!)
                f.write(f"{item['filename']} {item['class']}\n")

    process_set(train_data, "train")
    process_set(test_data, "test")

    # 5. Limpar buffer
    try:
        os.rmdir(buffer_dir)
    except:
        pass

    print("\n[SUCESSO] Dataset completo gerado!")
    print(f"Local: {OUTPUT_ROOT}")
    print("Agora podes correr o 'train_fcn.py' diretamente.")

if __name__ == "__main__":
    main()