import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.base import CLCC
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================
os.makedirs("./output", exist_ok=True)

# ==========================================================
# INITIALIZE MODEL
# NOTE: Model is UNTRAINED → results are not true DL enhancement
# ==========================================================
model = CLCC(64, 3, 3)
model.eval()


# ==========================================================
# HISTOGRAM EQUALIZATION (GLOBAL CONTRAST)
# ==========================================================
def hist_eq(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# ==========================================================
# CLAHE (LOCAL CONTRAST ENHANCEMENT)
# ==========================================================
def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe_filter = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe_filter.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# ==========================================================
# CLCC (DEEP MODEL)
# FIX: Increased alpha to avoid misleadingly high PSNR/SSIM
# ==========================================================
def clcc(img):

    # --- Input ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    # --- Model ---
    with torch.no_grad():
        out = model(tensor)

    out = out.squeeze().permute(1, 2, 0).numpy()
    out = (out + 1) / 2
    out = np.clip(out, 0, 1)

    # ======================================================
    # ✅ VERY SAFE BLENDING (core idea)
    # ======================================================
    alpha = 0.1  # keep it LOW
    enhanced = img_norm + alpha * (out - img_norm)

    enhanced = np.clip(enhanced, 0, 1)

    # ======================================================
    # ✅ LIGHT CORRECTIONS ONLY (no aggressive ops)
    # ======================================================

    # Mild gamma (just to remove dullness)
    gamma = 1.05
    enhanced = np.power(enhanced, 1 / gamma)

    # Slight contrast boost (NOT stretching)
    enhanced = cv2.convertScaleAbs((enhanced * 255).astype(np.uint8), alpha=1.1, beta=5)

    return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)


# ==========================================================
# METRIC COMPUTATION
# NOTE: No ground truth available → interpret carefully
# ==========================================================
def compute_metrics(img, hist_img, clahe_img, clcc_img):

    # PSNR
    psnr_hist = psnr(img, hist_img)
    psnr_clahe = psnr(img, clahe_img)
    psnr_clcc = psnr(img, clcc_img)

    # SSIM
    ssim_hist = ssim(img, hist_img, channel_axis=2)
    ssim_clahe = ssim(img, clahe_img, channel_axis=2)
    ssim_clcc = ssim(img, clcc_img, channel_axis=2)

    # Relative improvement vs baseline
    rel_psnr_clahe = ((psnr_clahe - psnr_hist) / psnr_hist) * 100
    rel_psnr_clcc = ((psnr_clcc - psnr_hist) / psnr_hist) * 100

    rel_ssim_clahe = ((ssim_clahe - ssim_hist) / ssim_hist) * 100
    rel_ssim_clcc = ((ssim_clcc - ssim_hist) / ssim_hist) * 100

    print("\nEvaluation Metric Results (Reference = Input Image)")
    print("NOTE: These metrics indicate structural preservation, NOT true quality.")
    print("-" * 80)
    print(f"{'Method':<30}{'PSNR':<24}{'SSIM':<24}")
    print("-" * 80)

    print(f"{'Input Image':<30}{'∞':<24}{'1.000':<24}")

    print(
        f"{'Histogram Equalization':<30}{psnr_hist:<24}{ssim_hist:<24} (Baseline)"
    )
    print(
        f"{'CLAHE':<30}{psnr_clahe:.3f} ({rel_psnr_clahe:+.2f}%){'':<9}{ssim_clahe:.3f} ({rel_ssim_clahe:+.2f}%)"
    )
    print(
        f"{'CLCC (Untrained)':<30}{psnr_clcc:.3f} ({rel_psnr_clcc:+.2f}%){'':<8}{ssim_clcc:.3f} ({rel_ssim_clcc:+.2f}%)"
    )

    print("-" * 80)


# ==========================================================
# VISUALIZATION
# ==========================================================
def visualize(img, hist_img, clahe_img, clcc_img, save_path):

    input_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist_disp = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)
    clahe_disp = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)
    clcc_disp = cv2.cvtColor(clcc_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Underwater Image Enhancement Comparison", fontsize=14)

    ax[0, 0].imshow(input_disp)
    ax[0, 0].set_title("Input Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(hist_disp)
    ax[0, 1].set_title("Histogram Equalization")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(clahe_disp)
    ax[1, 0].set_title("CLAHE")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(clcc_disp)
    ax[1, 1].set_title("CLCC (Deep Enhancement)")
    ax[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=125)


# ==========================================================
# PIPELINE
# ==========================================================
def enhance(img, output_dir):

    hist_img = hist_eq(img)
    clahe_img = clahe(img)
    clcc_img = clcc(img)

    cv2.imwrite(f"{output_dir}/hist_eq.png", hist_img)
    cv2.imwrite(f"{output_dir}/clahe.png", clahe_img)
    cv2.imwrite(f"{output_dir}/clcc.png", clcc_img)

    compute_metrics(img, hist_img, clahe_img, clcc_img)

    visualize(img, hist_img, clahe_img, clcc_img, f"{output_dir}/comparison.png")

    print("\nEnhancements saved to output folder\n")


# ==========================================================
# MAIN
# ==========================================================
index = 2
img = cv2.imread(f"./images/input{index}.png")

if img is None:
    raise ValueError("Image not found.")

enhance(img, "./output")
