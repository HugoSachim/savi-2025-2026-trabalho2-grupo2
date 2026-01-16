#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
import os
import numpy as np
import time

from model import ModelFCN 

def get_ground_truth_list(image_path):
    img_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(img_dir)
    img_filename = os.path.basename(image_path)
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(parent_dir, "labels", label_filename)
    gt_list = []
    if not os.path.exists(label_path): return gt_list
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5: parts = line.strip().split()
                if len(parts) >= 5:
                    gt_list.append({'class': int(parts[0]), 'box': [int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]) )]})
    except: pass
    return gt_list

def nms_global(boxes, scores, classes, iou_threshold=0.3):
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
    parser.add_argument('-m', '--model_path', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/experiments/FCN_Experiment/best.pkl')
    parser.add_argument('-i', '--image_path', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_D/mnist_detection/test/images/1.png')
    parser.add_argument('-t', '--threshold', type=float, default=0.1, help="Confiança mínima")
    # NOVO: Argumento de escala
    parser.add_argument('-s', '--scale', type=float, default=1.0, help="Fator de redimensionamento")
    parser.add_argument('--no_invert', action='store_true', help="NUNCA inverter cores")
    # NOVO: Argumento para forçar inversão
    parser.add_argument('--force_invert', action='store_true', help="FORÇAR inversão de cores (útil para imagens preto-fundo)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] A carregar modelo: {args.model_path}")
    model = ModelFCN().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 1. Carregar Imagem Original
    img_orig = Image.open(args.image_path).convert('RGB')
    img_gray = img_orig.convert('L')

    # Lógica de Inversão Melhorada
    mean_val = np.mean(np.array(img_gray))
    print(f"[INFO] Intensidade média da imagem: {mean_val:.2f}")

    # Inverte SE: (Forçado pelo user) OU (Fundo é claro E user não proibiu)
    if args.force_invert or (not args.no_invert and mean_val > 127):
        print("[INFO] A inverter cores (Transformar em Fundo Branco)...")
        img_gray = ImageOps.invert(img_gray)
    else:
        print("[INFO] Cores mantidas originais.")

    # 2. Redimensionar para a rede (Rescaling)
    # Se treinaste com 28px e os dígitos na imagem têm 36px, precisas de scale ~0.77
    original_w, original_h = img_gray.size
    new_w = int(original_w * args.scale)
    new_h = int(original_h * args.scale)
    
    print(f"[INFO] Resizing image: {original_w}x{original_h} -> {new_w}x{new_h} (Scale: {args.scale})")
    img_input = img_gray.resize((new_w, new_h), Image.BILINEAR)

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_input).unsqueeze(0).to(device)

    # 3. Inferência
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        best_scores, best_classes = torch.max(probabilities, dim=1)

    best_scores_np = best_scores.squeeze().cpu().numpy()
    best_classes_np = best_classes.squeeze().cpu().numpy()

    # Diagnóstico rápido
    print(f"\n--- DIAGNÓSTICO (Escala {args.scale}) ---")
    print(f"Max Score: {best_scores_np.max():.4f}")
    print(f"Classes detetadas: {np.unique(best_classes_np)}")

    detections = []
    scores_list = []
    classes_list = []
    
    STRIDE = 8 
    # O tamanho da janela na imagem REDIMENSIONADA é 28
    WINDOW_SIZE = 28 

    rows, cols = best_scores_np.shape
    
    for y in range(rows):
        for x in range(cols):
            cls = best_classes_np[y, x]
            score = best_scores_np[y, x]
            
            if cls == 10 or score < args.threshold:
                continue
            
            # Coordenadas na imagem REDIMENSIONADA
            x_small = int(x * STRIDE)
            y_small = int(y * STRIDE)
            
            # Mapear de volta para a imagem ORIGINAL (dividir pelo scale)
            x_orig = int(x_small / args.scale)
            y_orig = int(y_small / args.scale)
            w_orig = int(WINDOW_SIZE / args.scale)
            h_orig = int(WINDOW_SIZE / args.scale)

            box = [x_orig, y_orig, x_orig + w_orig, y_orig + h_orig]
            detections.append(box)
            scores_list.append(float(score))
            classes_list.append(int(cls))

    final_boxes, final_scores, final_classes = nms_global(detections, scores_list, classes_list)
    print(f"[INFO] Deteções finais: {len(final_boxes)}")

    # Desenho
    draw = ImageDraw.Draw(img_orig)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()

    # GT a Azul
    gts = get_ground_truth_list(args.image_path)
    for gt in gts:
        draw.rectangle(gt['box'], outline="blue", width=3)
        draw.text((gt['box'][0], gt['box'][1]-15), f"GT: {gt['class']}", fill="blue", font=font)

    # Predições a Verde
    for box, cls, score in zip(final_boxes, final_classes, final_scores):
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], box[1] + 25), f"{cls} ({score:.2f})", fill="lime", font=font)

    img_orig.show()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
import os
import numpy as np
import time

# Importar o teu modelo e classes
from model import ModelFCN 

def get_ground_truth_list(image_path):
    img_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(img_dir)
    img_filename = os.path.basename(image_path)
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(parent_dir, "labels", label_filename)
    gt_list = []
    if not os.path.exists(label_path): return gt_list
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5: parts = line.strip().split()
                if len(parts) >= 5:
                    gt_list.append({'class': int(parts[0]), 'box': [int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]) )]})
    except: pass
    return gt_list

def nms_global(boxes, scores, classes, iou_threshold=0.3):
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

