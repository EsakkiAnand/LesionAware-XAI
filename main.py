import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

from utils.preprocessing import load_and_preprocess_image, load_segmentation_mask
from utils.gradcam import make_gradcam_heatmap, binarize_heatmap
from utils.metrics import calculate_iou

IMG_WIDTH, IMG_HEIGHT = 224, 224
LAST_CONV_LAYER_NAME = "conv5_block3_out"
CLASSIFIER_LAYER_NAME = "predictions"

def process_image(image_path, mask_path, model, threshold=0.5, save_dir="results/visualizations"):
    preprocessed_img, original_img = load_and_preprocess_image(image_path)
    mask = load_segmentation_mask(mask_path)

    preds = model.predict(preprocessed_img)
    decoded = decode_predictions(preds, top=1)
    pred_class = np.argmax(preds)

    heatmap, _ = make_gradcam_heatmap(
        preprocessed_img,
        model,
        LAST_CONV_LAYER_NAME,
        CLASSIFIER_LAYER_NAME,
        pred_index=pred_class
    )

    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    gradcam_mask = binarize_heatmap(heatmap_resized, threshold)

    if mask.shape[:2] != (IMG_WIDTH, IMG_HEIGHT):
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    if len(mask.shape) == 3:
        mask = mask.squeeze()

    iou = calculate_iou(gradcam_mask, mask)

    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(original_img / 255.0)
    axs[0].set_title(f"Original\n{decoded[0][0][1]} ({decoded[0][0][2]:.2f})")
    axs[0].axis('off')

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(original_img / 255.0)
    axs[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axs[2].set_title("Grad-CAM Overlay")
    axs[2].axis('off')

    axs[3].imshow(gradcam_mask, cmap='gray')
    axs[3].set_title(f"Binarized Grad-CAM\nIoU={iou:.2f}")
    axs[3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, os.path.basename(image_path).split('.')[0] + "_result.png")
    plt.savefig(save_path)
    plt.close()

    return iou

def run_pipeline(data_dir, threshold):
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    result_csv = os.path.join("results", "scores.csv")

    model = ResNet50(weights='imagenet')
    image_files = os.listdir(image_dir)
    scores = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        mask_name = img_file.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_file}, skipping.")
            continue

        iou = process_image(img_path, mask_path, model, threshold)
        scores.append({"image": img_file, "IoU": iou})

    pd.DataFrame(scores).to_csv(result_csv, index=False)
    print(f"Saved results to {result_csv}")

if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset_root")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing Grad-CAM")
    args = parser.parse_args()

    run_pipeline(args.data, args.threshold)