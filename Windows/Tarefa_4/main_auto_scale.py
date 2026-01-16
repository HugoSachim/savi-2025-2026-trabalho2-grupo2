#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import numpy as np

from model import ModelFCN 

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
    parser.add_argument('-m', '--model_path', type=str, default='experiments/FCN_Experiment/best.pkl')
    parser.add_argument('-i', '--image_path', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_A/mnist_detection/test/images/1.png')
    parser.add_argument('-t', '--threshold', type=float, default=0.7, help="Confiança mínima")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] A carregar modelo: {args.model_path}")
    model = ModelFCN().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Carregar imagem original
    img_orig = Image.open(args.image_path).convert('RGB')
    img_gray_base = img_orig.convert('L')
    
    # NÃO invertemos cores (assumimos que o utilizador validou o dataset)
    # Se a imagem for Branco-sobre-Preto e o treino também, está ótimo.
    
    # Lista de escalas para testar automaticamente
    scales_to_test = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
    
    best_detections = []
    best_scale = 1.0
    
    print(f"[INFO] A testar multiescala...")

    to_tensor = transforms.ToTensor()

    for scale in scales_to_test:
        # Redimensionar
        w, h = img_gray_base.size
        new_w, new_h = int(w * scale), int(h * scale)
        img_input = img_gray_base.resize((new_w, new_h), Image.BILINEAR)
        
        img_tensor = to_tensor(img_input).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            best_scores, best_classes = torch.max(probabilities, dim=1)

        scores_np = best_scores.squeeze().cpu().numpy()
        classes_np = best_classes.squeeze().cpu().numpy()
        
        # Filtrar
        current_detections = []
        scores_list = []
        classes_list = []
        
        rows, cols = scores_np.shape
        STRIDE = 8
        WINDOW_SIZE = 28
        
        for y in range(rows):
            for x in range(cols):
                cls = classes_np[y, x]
                score = scores_np[y, x]
                
                if cls == 10 or score < args.threshold: continue
                
                # Coordenadas ajustadas à escala original
                x_orig = int((x * STRIDE) / scale)
                y_orig = int((y * STRIDE) / scale)
                w_orig = int(WINDOW_SIZE / scale)
                
                box = [x_orig, y_orig, x_orig + w_orig, y_orig + w_orig]
                current_detections.append(box)
                scores_list.append(float(score))
                classes_list.append(int(cls))
        
        print(f" -> Escala {scale:.1f}: Encontradas {len(current_detections)} caixas candidatas (Max Score: {scores_np.max():.2f})")
        
        if len(current_detections) > 0:
            # Guardamos tudo para fazer NMS final
            for i in range(len(current_detections)):
                best_detections.append({
                    'box': current_detections[i],
                    'score': scores_list[i],
                    'class': classes_list[i]
                })

    # Preparar NMS Final
    all_boxes = [d['box'] for d in best_detections]
    all_scores = [d['score'] for d in best_detections]
    all_classes = [d['class'] for d in best_detections]

    final_boxes, final_scores, final_classes = nms_global(all_boxes, all_scores, all_classes)
    
    print(f"\n[RESULTADO] Total final de deteções: {len(final_boxes)}")

    # Desenhar
    draw = ImageDraw.Draw(img_orig)
    try: font = ImageFont.truetype("arial.ttf", 20)
    except: font = ImageFont.load_default()

    for box, cls, score in zip(final_boxes, final_classes, final_scores):
        draw.rectangle(box, outline="lime", width=3)
        draw.text((box[0], box[1] - 20), f"{cls} ({score:.2f})", fill="lime", font=font)
        print(f"Detetado: {cls} com confiança {score:.2f} na posição {box}")

    img_orig.show()

if __name__ == '__main__':
    main()