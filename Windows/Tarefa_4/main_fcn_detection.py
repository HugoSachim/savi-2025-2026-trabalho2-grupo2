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
    parser.add_argument('-m', '--model_path', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/experiments/FCN_Transfer_v2/best.pkl', help="Caminho do modelo treinado")
    parser.add_argument('-i', '--image_path', type=str, default='C:/Users/matsd/OneDrive/Documentos/Ambiente de Trabalho/SAVI_2526/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_D/mnist_detection/test/images/5.png', help="Caminho da imagem para testar")
    parser.add_argument('-t', '--threshold', type=float, default=0.95, help="Confiança mínima (0.0 a 1.0)")
    parser.add_argument('-s', '--scale', type=float, default=1.0, help="Ajuste de tamanho (ex: 0.5 reduz a metade, 2.0 duplica)")
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