#!/usr/bin/env python3

import os
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['KMP_WARNINGS'] = '0'
import torch
import argparse
import numpy as np
import platform
import subprocess
import time
import cv2  # Adicionado para visualização
from PIL import Image, ImageDraw, ImageOps, ImageFont
from torchvision import transforms
import torch.nn.functional as F
from model import ModelBetterCNN 

def calculate_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(area1 + area2 - interArea) if (area1 + area2 - interArea) > 0 else 0

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
                line = line.strip()
                if not line or line.startswith('label'): continue
                parts = line.split(',')
                if len(parts) == 5:
                    gt_list.append({'class': int(parts[0]), 'box': [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]})
    except: pass
    return gt_list

def nms_global(boxes, scores, classes, iou_threshold=0.1):
    if not boxes: return [], [], []
    b, s = torch.tensor(boxes, dtype=torch.float), torch.tensor(scores)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas, _, order = (x2 - x1) * (y2 - y1), None, s.sort(0, descending=True)[1]
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1: break
        xx1, yy1 = torch.clamp(x1[order[1:]], min=x1[i]), torch.clamp(y1[order[1:]], min=y1[i])
        xx2, yy2 = torch.clamp(x2[order[1:]], max=x2[i]), torch.clamp(y2[order[1:]], max=y2[i])
        w, h = torch.clamp(xx2 - xx1, min=0.0), torch.clamp(yy2 - yy1, min=0.0)
        iou = (w * h) / (areas[i] + areas[order[1:]] - (w * h))
        ids = (iou <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0: break
        order = order[ids + 1] if ids.dim() > 0 else order[ids + 1].unsqueeze(0)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [classes[i] for i in keep]

def is_centered(crop_np, threshold=40):
    edge = np.concatenate([crop_np[0, :], crop_np[-1, :], crop_np[:, 0], crop_np[:, -1]])
    return np.max(edge) <= threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='/home/hogu/Desktop/savi-2025-2026-trabalho2-grupoX/datasets/savi_experiments/Tarefa_1/best.pkl')
    parser.add_argument('-i', '--image_path', type=str, default='/home/hogu/Desktop/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_D/mnist_detection/test/images/56.png')
    parser.add_argument('-t', '--threshold', type=float, default=0.99)
    parser.add_argument('-s', '--stride', type=int, default=2) 
    parser.add_argument('-v', '--visualize', action='store_true', help='Ativar visualização em tempo real')
    args = parser.parse_args()

    gts = get_ground_truth_list(args.image_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelBetterCNN().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img_orig = Image.open(args.image_path).convert('RGB')
    gray_img = img_orig.convert('L')
    if np.mean(gray_img) > 127: gray_img = ImageOps.invert(gray_img)

    # Preparar imagem para OpenCV (BGR) se visualização estiver ativa
    img_cv = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)
    
    to_tensor = transforms.ToTensor()
    detections, scores, classes_list = [], [], []
    windows_scanned = 0
    window_sizes = [22, 28, 36] 

    print(f"\n[INFO] A iniciar scan da imagem: {os.path.basename(args.image_path)}")
    start_time = time.time()

    for win_size in window_sizes:
        for y in range(0, gray_img.height - win_size, args.stride):
            for x in range(0, gray_img.width - win_size, args.stride):
                windows_scanned += 1
                
                # Visualização em tempo real
                if args.visualize:
                    temp_img = img_cv.copy()
                    # Desenha caixa de scan atual (Branca)
                    cv2.rectangle(temp_img, (x, y), (x + win_size, y + win_size), (255, 255, 255), 1)
                    cv2.imshow("Processo de Sliding Window", temp_img)
                    # waitKey(1) permite que a janela processe o desenho. 
                    # Aumenta para 20 ou 50 se quiseres ver mais devagar.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        args.visualize = False # Para a visualização se premires 'q'

                crop = gray_img.crop((x, y, x + win_size, y + win_size))
                crop_np = np.array(crop)
                
                if np.max(crop_np) < 50 or not is_centered(crop_np): continue
                
                crop_t = to_tensor(crop.resize((28, 28))).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(crop_t)
                    conf, pred = torch.max(F.softmax(logits, dim=1), dim=1)
                
                if conf.item() > args.threshold:
                    detections.append([x, y, x + win_size, y + win_size])
                    scores.append(conf.item())
                    classes_list.append(pred.item())
                    
                    # Desenha deteção bruta na imagem base do OpenCV (Verde Fino)
                    if args.visualize:
                        cv2.rectangle(img_cv, (x, y), (x + win_size, y + win_size), (0, 255, 0), 1)

    if args.visualize: cv2.destroyAllWindows()

    final_boxes, final_scores, final_classes = nms_global(detections, scores, classes_list)
    execution_time = time.time() - start_time

    # DESENHO FINAL (PIL)
    draw = ImageDraw.Draw(img_orig)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except: font = ImageFont.load_default()

    for gt in gts:
        draw.rectangle(gt['box'], outline="blue", width=3)
        draw.text((gt['box'][0], gt['box'][1]-25), f"GT: {gt['class']}", fill="blue", font=font)

    for box, cls, score in zip(final_boxes, final_classes, final_scores):
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], box[1]-15), f"DET: {cls} ({score:.2f})", fill="lime", font=font)
        max_iou = max([calculate_iou(box, gt['box']) for gt in gts]) if gts else 0
        print(f"Deteção: {cls} | IoU: {max_iou:.4f} | Local: {box}")

    print(f"\nEficiência: {windows_scanned} janelas em {execution_time:.2f}s")
    img_orig.show()

if __name__ == '__main__':
    main()