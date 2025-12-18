#!/usr/bin/env python3
# shebang line for linux / mac

import argparse
import os
import platform
import subprocess
import mnist
import pathlib
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use("Agg") 

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box
    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    # Compute intersection
    x1i = max(x1_t, x1_p)
    x2i = min(x2_t, x2_p)
    y1i = max(y1_t, y1_p)
    y2i = min(y2_t, y2_p)
    intersection = (x2i - x1i) * (y2i - y1i)

    # Compute union
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_t - x1_t) * (y2_t - y1_t)
    union = pred_area + gt_area - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def compute_iou_all(bbox, all_bboxes):
    ious = [0]
    for other_bbox in all_bboxes:
        ious.append(
            calculate_iou(bbox, other_bbox)
        )
    return ious


def tight_bbox(digit, orig_bbox, yes_or_no):
    xmin, ymin, xmax, ymax = orig_bbox
    if yes_or_no == 'yes':
        # xmin
        shift = 0
        for i in range(digit.shape[1]):
            if digit[:, i].sum() != 0:
                break
            shift += 1
        xmin += shift
        # xmax
        shift = 0
        for i in range(-1, -digit.shape[1], -1):
            if digit[:, i].sum() != 0:
                break
            shift += 1
        xmax -= shift
        ymin
        shift = 0
        for i in range(digit.shape[0]):
            if digit[i, :].sum() != 0:
                break
            shift += 1
        ymin += shift
        shift = 0
        for i in range(-1, -digit.shape[0], -1):
            if digit[i, :].sum() != 0:
                break
            shift += 1
        ymax -= shift
    else:
        pass
    return [xmin, ymin, xmax, ymax]


def dataset_exists(dirpath: pathlib.Path, num_images):
    if not dirpath.is_dir():
        return False
    for image_id in range(num_images):
        error_msg = f"MNIST dataset already generated in {dirpath}, \n\tbut did not find filepath:"
        error_msg2 = f"You can delete the directory by running: rm -r {dirpath.parent}"
        impath = dirpath.joinpath("images", f"{image_id}.png")
        assert impath.is_file(), f"{error_msg} {impath} \n\t{error_msg2}"
        label_path = dirpath.joinpath("labels", f"{image_id}.txt")
        assert label_path.is_file(),  f"{error_msg} {impath} \n\t{error_msg2}"
    return True


def generate_dataset(dirpath: pathlib.Path,
                     num_images: int,
                     max_digit_size: int,
                     min_digit_size: int,
                     imsize: int,
                     min_digits_per_image: int,
                     max_digits_per_image: int,
                     mnist_images: np.ndarray,
                     mnist_labels: np.ndarray):
    if dataset_exists(dirpath, num_images):
        return
    max_image_value = 255
    assert mnist_images.dtype == np.uint8
    image_dir = dirpath.joinpath("images")
    label_dir = dirpath.joinpath("labels")
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    for image_id in tqdm.trange(num_images, desc=f"Generating dataset, saving to: {dirpath}"):
        im = np.zeros((imsize, imsize), dtype=np.float32)
        labels = []
        bboxes = []
        num_images = np.random.randint(min_digits_per_image -1 , max_digits_per_image) #para incluir o numero minimo
        for _ in range(num_images+1):
            while True:
                width = np.random.randint(min_digit_size, max_digit_size + 1) #alterado para incluir o max, de outro modo nao inclui
                x0 = np.random.randint(0, imsize-width)
                y0 = np.random.randint(0, imsize-width)
                ious = compute_iou_all([x0, y0, x0+width, y0+width], bboxes)
                if max(ious) < 0.001: #sobreposicao reduzido para nao meter 
                    break
            digit_idx = np.random.randint(0, len(mnist_images))
            digit = mnist_images[digit_idx].astype(np.float32)
            digit = cv2.resize(digit, (width, width))
            label = mnist_labels[digit_idx]
            labels.append(label)
            assert im[y0:y0+width, x0:x0+width].shape == digit.shape, \
                f"imshape: {im[y0:y0+width, x0:x0+width].shape}, digit shape: {digit.shape}"
            bbox = tight_bbox(digit, [x0, y0, x0+width, y0+width], yes_or_no = 'no') 
            bboxes.append(bbox)

            im[y0:y0+width, x0:x0+width] += digit
            im[im > max_image_value] = max_image_value
        image_target_path = image_dir.joinpath(f"{image_id}.png")
        label_target_path = label_dir.joinpath(f"{image_id}.txt")
        im = im.astype(np.uint8)
        cv2.imwrite(str(image_target_path), im)
        with open(label_target_path, "w") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for l, bbox in zip(labels, bboxes):
                bbox = [str(_) for _ in bbox]
                to_write = f"{l}," + ",".join(bbox) + "\n"
                fp.write(to_write)

def visualize_dataset(dataset_dir, num_images=36):
    """
    Visualiza um mosaico de imagens com bounding boxes e mostra estatísticas do dataset.
    
    Args:
        dataset_dir (str / pathlib.Path): caminho para a pasta do dataset (com 'images' e 'labels')
        num_images (int): número de imagens a mostrar no mosaico
    """
    dataset_dir = pathlib.Path(dataset_dir)
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    # Todas as imagens para cálculo das estatísticas
    all_image_files = sorted(image_dir.glob("*.png"))
    total_images = len(all_image_files)
    
    # Para estatísticas globais
    class_counts = []
    digits_per_image = []
    digit_sizes = []

    for img_path in all_image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        label_path = label_dir / f"{img_path.stem}.txt"
        labels = pd.read_csv(label_path)
        digits_per_image.append(len(labels))
        for _, row in labels.iterrows():
            class_counts.append(row['label'])
            xmin, ymin, xmax, ymax = row[['xmin','ymin','xmax','ymax']].astype(int)
            digit_sizes.append(max(xmax-xmin, ymax-ymin))

    # Converter para arrays
    class_counts = np.array(class_counts)
    digits_per_image = np.array(digits_per_image)
    digit_sizes = np.array(digit_sizes)

    # ------------------------------
    # Criar mosaico de imagens aleatórias
    # ------------------------------
    selected_files = np.random.choice(all_image_files, min(num_images, total_images), replace=False)
    n_cols = int(np.sqrt(len(selected_files)))
    n_rows = int(np.ceil(len(selected_files)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()

    for ax, img_path in zip(axes, selected_files):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        label_path = label_dir / f"{img_path.stem}.txt"
        labels = pd.read_csv(label_path)
        for _, row in labels.iterrows():
            xmin, ymin, xmax, ymax = row[['xmin','ymin','xmax','ymax']].astype(int)
            rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                             linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    for ax in axes[len(selected_files):]:
        ax.axis('off')

    plt.tight_layout()
    path_mosaico_dataset = "./data/mosaico_dataset.png"
    plt.savefig(path_mosaico_dataset)
    plt.close()
    open_file(path_mosaico_dataset)

    # ------------------------------
    # Estatísticas globais e gráficos
    # ------------------------------
    # Distribuição das classes
    plt.figure(figsize=(8,5))
    sns.countplot(x=class_counts)
    plt.title("Distribuição de Classes")
    plt.xlabel("Classe")
    plt.ylabel("Contagem")
    plt.text(0.95, 0.95,
             f"Número total de imagens: {total_images}\n"
             f"Número total de dígitos: {len(class_counts)}",
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    path_dist_classes_with_stats = "./data/dist_classes_with_stats.png"
    plt.savefig(path_dist_classes_with_stats)
    plt.close()
    open_file(path_dist_classes_with_stats)

    # Número de dígitos por imagem
    plt.figure(figsize=(6,4))
    bins_digits = np.arange(digits_per_image.min() - 0.5, digits_per_image.max() + 1.5, 1)
    plt.hist(digits_per_image, bins=bins_digits, edgecolor='black', rwidth=0.8)
    plt.title("Número de dígitos por imagem")
    plt.xlabel("Dígitos por imagem")
    plt.ylabel("Número de imagens")
    plt.text(0.95, 0.95,
            f"Média de dígitos por imagem: {digits_per_image.mean():.2f}",
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    path_dig_per_image_with_stats = "./data/dig_per_image_with_stats.png"
    plt.savefig(path_dig_per_image_with_stats)
    plt.close()
    open_file(path_dig_per_image_with_stats)

    # Tamanho dos dígitos
    plt.figure(figsize=(6,4))
    bins_sizes = np.arange(int(digit_sizes.min()) - 0.5, int(digit_sizes.max()) + 1.5, 1)
    plt.hist(digit_sizes, bins=bins_sizes, edgecolor='black', rwidth=0.8)
    plt.title("Tamanho dos dígitos")
    plt.xlabel("Tamanho (pixels)")
    plt.ylabel("Número de dígitos")
    plt.text(0.95, 0.95,
            f"Tamanho médio dos dígitos: {digit_sizes.mean():.2f} px",
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    path_digit_sizes_with_stats = "./data/digit_sizes_with_stats.png"
    plt.savefig(path_digit_sizes_with_stats)
    plt.close()
    open_file(path_digit_sizes_with_stats)

    print("Visualização e estatísticas geradas.")

def open_file(path):
    try:
        system_name = platform.system()
        if system_name == "Windows":
            os.startfile(path)  # abre com programa padrão
        elif system_name == "Darwin":  # macOS
            subprocess.call(["open", path])
        else:  # Linux e outros
            subprocess.call(["xdg-open", path])
    except Exception as e:
        print("Não foi possível abrir a imagem:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path", default="data/mnist_detection", help='Path to save dataset.'
    )
    parser.add_argument(
        "--imsize", default=128, type=int, help='Image size.'
    )
    # foi retirado o tight box de modo a nao reduzir a boinding box
    parser.add_argument(
        "--max-digit-size", default=36, type=int, help='Max digit size.'
    )
    parser.add_argument(
        "--min-digit-size", default=22, type=int, help='Min digit size.'
    )
    parser.add_argument(
        "--num-train-images", default=60000, type=int, help='Number train images.'
    )
    parser.add_argument(
        "--num-test-images", default=10000, type=int, help='Number test images.'
    )
    parser.add_argument(
        "--min-digits-per-image", default=3, type=int, help='Min digits per image.'
    )
    parser.add_argument(
        "--max-digits-per-image", default=5, type=int, help='Max digits per image.'
    )
    parser.add_argument('-vs', '--visualization', action='store_true', help='Análise e Visualização do dataset.')
    parser.add_argument(
        '-pv', '--path-visualization', default="data/mnist_detection/train", help='Path to visualization dataset.'
    )

    # Versão A: Apenas 1 dígito por imagem mas em posição aleatória. min dig 1, max digit 1, min size 28, max size 28
    # Versão B: Apenas 1 dígito por imagem mas em posição aleatória e com diferenças de escala. min dig 1, max digit 1, min size 22, max size 36
    # Versão C: Múltiplos dígitos por imagem entre 3 a 5 dígitos. min dig 3, max digit 5, min size 28, max size 28
    # Versão D: Múltiplos dígitos por imagem entre 3 a 5 dígitos com diferenças de escala. min dig 3, max digit 5, min size 22, max size 36

    parser.add_argument('-vA', '--versao-A', action='store_true', help='Apenas 1 dígito por imagem mas em posição aleatória.')
    parser.add_argument('-vB', '--versao-B', action='store_true', help='Apenas 1 dígito por imagem mas em posição aleatória e com diferenças de escala.')
    parser.add_argument('-vC', '--versao-C', action='store_true', help='Múltiplos dígitos por imagem entre 3 a 5 dígitos.')
    parser.add_argument('-vD', '--versao-D', action='store_true', help='Múltiplos dígitos por imagem entre 3 a 5 dígitos com diferenças de escala.')

    args = parser.parse_args()

    if args.visualization:
        visualize_dataset(args.path_visualization, num_images=25)
    else:
        if args.versao_A:
            X_train, Y_train, X_test, Y_test = mnist.load()
            for dataset, (X, Y) in zip(["train", "test"], [[X_train, Y_train], [X_test, Y_test]]):
                num_images = args.num_train_images if dataset == "train" else args.num_test_images
                generate_dataset(
                    pathlib.Path(args.base_path, dataset),
                    num_images,
                    28,
                    28,
                    args.imsize,
                    1,
                    1,
                    X,
                    Y) 
        elif args.versao_B:
            X_train, Y_train, X_test, Y_test = mnist.load()
            for dataset, (X, Y) in zip(["train", "test"], [[X_train, Y_train], [X_test, Y_test]]):
                num_images = args.num_train_images if dataset == "train" else args.num_test_images
                generate_dataset(
                    pathlib.Path(args.base_path, dataset),
                    num_images,
                    36,
                    22,
                    args.imsize,
                    1,
                    1,
                    X,
                    Y) 
        elif args.versao_C:
            X_train, Y_train, X_test, Y_test = mnist.load()
            for dataset, (X, Y) in zip(["train", "test"], [[X_train, Y_train], [X_test, Y_test]]):
                num_images = args.num_train_images if dataset == "train" else args.num_test_images
                generate_dataset(
                    pathlib.Path(args.base_path, dataset),
                    num_images,
                    28,
                    28,
                    args.imsize,
                    3,
                    5,
                    X,
                    Y) 
        elif args.versao_D:
            X_train, Y_train, X_test, Y_test = mnist.load()
            for dataset, (X, Y) in zip(["train", "test"], [[X_train, Y_train], [X_test, Y_test]]):
                num_images = args.num_train_images if dataset == "train" else args.num_test_images
                generate_dataset(
                    pathlib.Path(args.base_path, dataset),
                    num_images,
                    36,
                    22,
                    args.imsize,
                    3,
                    5,
                    X,
                    Y) 
        else:
            X_train, Y_train, X_test, Y_test = mnist.load()
            for dataset, (X, Y) in zip(["train", "test"], [[X_train, Y_train], [X_test, Y_test]]):
                num_images = args.num_train_images if dataset == "train" else args.num_test_images
                generate_dataset(
                    pathlib.Path(args.base_path, dataset),
                    num_images,
                    args.max_digit_size,
                    args.min_digit_size,
                    args.imsize,
                    args.min_digits_per_image,
                    args.max_digits_per_image,
                    X,
                    Y) 
