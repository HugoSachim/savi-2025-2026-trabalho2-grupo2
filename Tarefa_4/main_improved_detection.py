#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import numpy as np
import time

# Importa a tua classe FCN. Certifica-te que ela está no model.py!
from model import ModelFCN 

def get_ground_truth_list(image_path):
    img_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(img_dir)
    img_filename = os.path.basename(image_path)
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(parent_dir, "labels", label_filename)
    
    gt_list = []
    if not os.path.exists(label_path): 
        return gt_list

    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue # Ignora linhas vazias

                # Separa por vírgula (se for CSV) ou espaço
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()

                # Se a linha tiver menos de 5 elementos, ignora
                if len(parts) < 5: continue

                try:
                    # Tenta converter para números
                    # Se for o cabeçalho ("label", "xmin"...), isto falha e vai para o except
                    cls = int(parts[0])
                    x1 = int(float(parts[1]))
                    y1 = int(float(parts[2]))
                    x2 = int(float(parts[3]))
                    y2 = int(float(parts[4]))
                    
                    gt_list.append({'class': cls, 'box': [x1, y1, x2, y2]})
                except ValueError:
                    # Se der erro de conversão (ex: leu "label" em vez de um número),
                    # simplesmente ignora esta linha e passa à próxima.
                    continue
                    
    except Exception as e:
        print(f"[AVISO] Erro ao ler GT: {e}")
        
    return gt_list

def nms_global(boxes, scores, classes, iou_threshold=0.05):
    """Remove caixas sobrepostas (Non-Maximum Suppression)"""
    if not boxes: return [], [], []
    b = torch.tensor(boxes, dtype=torch.float)
    s = torch.tensor(scores)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.sort(0, descending=True)[1]
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1: break
        xx1 = torch.clamp(x1[order[1:]], min=x1[i])
        yy1 = torch.clamp(y1[order[1:]], min=y1[i])
        xx2 = torch.clamp(x2[order[1:]], max=x2[i])
        yy2 = torch.clamp(y2[order[1:]], max=y2[i])
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (iou <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0: break
        order = order[ids + 1] if ids.dim() > 0 else order[ids + 1].unsqueeze(0)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [classes[i] for i in keep]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='/home/hogu/Desktop/savi-2025-2026-trabalho2-grupoX/experiments/FCN_Transfer_v2/best.pkl')
    parser.add_argument('-i', '--image_path', type=str, default='/home/hogu/Desktop/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_D/mnist_detection/test/images/85.png')
    parser.add_argument('-t', '--threshold', type=float, default=0.95)
    parser.add_argument('-s', '--scale', type=float, default=1.0) # Ajuste de tamanho
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Carregar Modelo
    print(f"[INFO] A carregar modelo: {args.model_path}")
    if not os.path.exists(args.model_path):
        print("ERRO: Modelo não encontrado!")
        return

    # Inicializa a FCN com 11 classes (0-9 + Fundo)
    model = ModelFCN(num_classes=11).to(device)
    
    # Carrega os pesos (com fix para o erro de pickle)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Carregar e Preparar Imagem
    if not os.path.exists(args.image_path):
        print("ERRO: Imagem não encontrada!")
        return

    img_orig = Image.open(args.image_path).convert('RGB')
    img_gray = img_orig.convert('L')
    
    # Redimensionar se necessário (Scale)
    w, h = img_gray.size
    new_w, new_h = int(w * args.scale), int(h * args.scale)
    if args.scale != 1.0:
        print(f"[INFO] Redimensionar: {w}x{h} -> {new_w}x{new_h} (Scale {args.scale})")
        img_input = img_gray.resize((new_w, new_h), Image.BILINEAR)
    else:
        img_input = img_gray

    # Converter para Tensor
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_input).unsqueeze(0).to(device)

    # 3. Inferência (A Magia da FCN)
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        # O output é um mapa de calor: (1, 11, H_out, W_out)
        probabilities = F.softmax(output, dim=1)
        best_scores, best_classes = torch.max(probabilities, dim=1)

    # 4. Processar Resultados
    scores_np = best_scores.squeeze().cpu().numpy()
    classes_np = best_classes.squeeze().cpu().numpy()

    detections = []
    scores_list = []
    classes_list = []

    # Parâmetros da rede (Pooling x3 = div por 8)
    STRIDE = 8
    WINDOW_SIZE = 28 # Tamanho que a rede "vê"

    rows, cols = scores_np.shape
    
    for y in range(rows):
        for x in range(cols):
            cls = classes_np[y, x]
            score = scores_np[y, x]
            
            # Ignora Classe 10 (Fundo) e confiança baixa
            if cls == 10 or score < args.threshold:
                continue
            
            # Mapear coordenadas do Heatmap de volta para a Imagem Original
            # Se usámos scale, temos de dividir para voltar ao tamanho original
            x_img = int((x * STRIDE) / args.scale)
            y_img = int((y * STRIDE) / args.scale)
            w_box = int(WINDOW_SIZE / args.scale)
            h_box = int(WINDOW_SIZE / args.scale)
            
            box = [x_img, y_img, x_img + w_box, y_img + h_box]
            
            detections.append(box)
            scores_list.append(float(score))
            classes_list.append(int(cls))

    # Limpar sobreposições
    final_boxes, final_scores, final_classes = nms_global(detections, scores_list, classes_list)
    
    print(f"[INFO] Tempo: {time.time() - start_time:.3f}s | Deteções: {len(final_boxes)}")

    # 5. Desenhar
    draw = ImageDraw.Draw(img_orig)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()

    # --- DESENHAR GROUND TRUTH (AZUL) ---
    gt_boxes = get_ground_truth_list(args.image_path)
    for gt in gt_boxes:
        # Desenha retângulo azul
        draw.rectangle(gt['box'], outline="blue", width=3)
        
        label_gt = f"GT: {gt['class']}"
        
        # ALTERAÇÃO: Usamos gt['box'][3] (fundo da caixa) + 5 pixeis de margem
        # Antes estava: gt['box'][1] - 15 (topo)
        text_x = gt['box'][0]
        text_y = gt['box'][3] + 5 
        
        draw.text((text_x, text_y), label_gt, fill="blue", font=font)
    # ------------------------------------------

    for box, cls, score in zip(final_boxes, final_classes, final_scores):
        # Desenha retângulo verde
        draw.rectangle(box, outline="lime", width=3)
        # Escreve a classe e confiança
        label = f"{cls} ({score:.2f})"
        draw.text((box[0], box[1] - 15), label, fill="lime", font=font)
        print(f" -> Encontrei um {cls} (conf: {score:.2f}) em {box}")

    img_orig.show()
    # Opcional: Salvar
    # img_orig.save("resultado_deteccao.png")

if __name__ == '__main__':
    main()


    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/experiments/FCN_Transfer_v2/best.pkl')
    parser.add_argument('-i', '--image_path', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_D/mnist_detection/test/images/85.png')
    parser.add_argument('-t', '--threshold', type=float, default=0.95)
    parser.add_argument('-s', '--scale', type=float, default=1.0) # Ajuste de tamanho
    args = parser.parse_args()