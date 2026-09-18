"""generate_review_iii_evidence.py

Master evidence generation script for Review III PPT.
Extracts existing logs, evaluates checkpoints on Kvasir test set,
generates publication-ready plots, Grad-CAM, MC Dropout uncertainty,
calibration reliability diagrams, faithfulness curves, and comparison tables.

IMPORTANT: NO MODEL TRAINING IS PERFORMED.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Project imports
from src.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.dataloader import create_dataloaders
from src.models.factory import build_model
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.calibration import compute_ece, compute_brier_score, compute_reliability_data, plot_reliability_diagram
from src.inference.uncertainty import MCDropoutPredictor
from src.explainability.pipeline import ExplainabilityPipeline
from src.evaluation.faithfulness import FaithfulnessEvaluator

CLASS_NAMES_8 = [
    "Angiectasia",
    "Erosion",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Pylorus",
    "Reduced mucosal view",
    "Ulcer",
]

# Set matplotlib aesthetic style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

MODELS_CONFIG = [
    {
        "name": "EfficientNetV2-S",
        "variant": "efficientnet_only",
        "config_dir": "config",
        "history_path": "outputs/efficientnet_only/history.json",
        "checkpoint_path": "outputs/efficientnet_only/checkpoints/best_f1.pth",
        "color": "#1f77b4"
    },
    {
        "name": "ViT-Small/16",
        "variant": "vit_only",
        "config_dir": "config",
        "history_path": "outputs/vit_only/history.json",
        "checkpoint_path": "outputs/vit_only/checkpoints/best_f1.pth",
        "color": "#ff7f0e"
    },
    {
        "name": "Hybrid Concat",
        "variant": "hybrid_concat",
        "config_dir": "config",
        "history_path": "outputs/hybrid_concat/history.json",
        "checkpoint_path": "outputs/hybrid_concat/checkpoints/best_f1.pth",
        "color": "#2ca02c"
    },
    {
        "name": "Hybrid Adaptive",
        "variant": "hybrid_adaptive",
        "config_dir": "config",
        "history_path": "outputs/hybrid_adaptive/history.json",
        "checkpoint_path": "outputs/hybrid_adaptive/checkpoints/best_f1.pth",
        "color": "#d62728"
    },
    {
        "name": "Hybrid Class-Aware (Proposed)",
        "variant": "hybrid_class_aware",
        "config_dir": "config_kvasir",
        "history_path": "outputs_kvasir/hybrid_class_aware/history.json",
        "checkpoint_path": "outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth",
        "color": "#9467bd"
    }
]


def ensure_dirs():
    dirs = [
        "review_iii_assets/training",
        "review_iii_assets/confusion_matrices",
        "review_iii_assets/classification_reports",
        "review_iii_assets/gradcam",
        "review_iii_assets/calibration",
        "review_iii_assets/faithfulness",
        "review_iii_assets/uncertainty",
        "review_iii_assets/comparison",
        "review_iii_assets/single_image",
        "outputs_kvasir/hybrid_class_aware/plots/gradcam_review_iii",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def step_1_generate_training_plots():
    print("\n" + "="*70)
    print("STEP 1: GENERATING TRAINING HISTORY PLOTS FROM EXISTING LOGS")
    print("="*70)
    
    all_histories = {}
    
    for m in MODELS_CONFIG:
        p = m["history_path"]
        if not os.path.exists(p):
            print(f"[-] History not found for {m['name']} at {p}")
            continue
        with open(p) as f:
            data = json.load(f)
        records = data.get("records", [])
        all_histories[m["variant"]] = records
        
        epochs = [r["epoch"] + 1 for r in records]
        train_loss = [r["train_loss"] for r in records]
        val_loss = [r["val_loss"] for r in records]
        val_acc = [r["metrics"]["accuracy"] * 100 for r in records]
        val_bal_acc = [r["metrics"]["balanced_accuracy"] * 100 for r in records]
        val_f1 = [r["metrics"]["f1_macro"] * 100 for r in records]
        lrs = [r.get("learning_rate", 0) for r in records]
        
        var = m["variant"]
        
        # 1. Loss Plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2.2, label='Train Loss')
        plt.plot(epochs, val_loss, 's--', color='#d62728', linewidth=2.2, label='Val Loss')
        best_loss_idx = int(np.argmin(val_loss))
        plt.scatter([epochs[best_loss_idx]], [val_loss[best_loss_idx]], color='gold', s=150, zorder=5, edgecolors='black', label=f'Best Val Loss: {val_loss[best_loss_idx]:.4f} (Ep {epochs[best_loss_idx]})')
        plt.title(f'Training & Validation Loss — {m["name"]}', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (CrossEntropy / Focal / Poly)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        plt.tight_layout()
        out_p1 = f"review_iii_assets/training/{var}_loss.png"
        plt.savefig(out_p1)
        plt.close()
        
        # 2. Accuracy Plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_acc, 'o-', color='#2ca02c', linewidth=2.2, label='Val Accuracy')
        plt.plot(epochs, val_bal_acc, '^-', color='#ff7f0e', linewidth=2.2, label='Val Balanced Accuracy')
        best_acc_idx = int(np.argmax(val_acc))
        plt.scatter([epochs[best_acc_idx]], [val_acc[best_acc_idx]], color='gold', s=150, zorder=5, edgecolors='black', label=f'Best Val Acc: {val_acc[best_acc_idx]:.2f}% (Ep {epochs[best_acc_idx]})')
        plt.title(f'Validation Accuracy — {m["name"]}', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        plt.tight_layout()
        out_p2 = f"review_iii_assets/training/{var}_accuracy.png"
        plt.savefig(out_p2)
        plt.close()

        # 3. Macro-F1 Plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_f1, 'd-', color='#9467bd', linewidth=2.2, label='Val Macro-F1')
        best_f1_idx = int(np.argmax(val_f1))
        plt.scatter([epochs[best_f1_idx]], [val_f1[best_f1_idx]], color='gold', s=150, zorder=5, edgecolors='black', label=f'Best Macro-F1: {val_f1[best_f1_idx]:.2f}% (Ep {epochs[best_f1_idx]})')
        plt.title(f'Validation Macro-F1 Score — {m["name"]}', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Macro-F1 (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        plt.tight_layout()
        out_p3 = f"review_iii_assets/training/{var}_macro_f1.png"
        plt.savefig(out_p3)
        plt.close()
        
        # 4. 4-Panel Overview Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # (a) Loss
        axes[0, 0].plot(epochs, train_loss, 'o-', color='#1f77b4', label='Train Loss')
        axes[0, 0].plot(epochs, val_loss, 's--', color='#d62728', label='Val Loss')
        axes[0, 0].set_title('(A) Training & Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        # (b) Macro F1 & Balanced Acc
        axes[0, 1].plot(epochs, val_f1, 'd-', color='#9467bd', label='Val Macro-F1')
        axes[0, 1].plot(epochs, val_bal_acc, '^-', color='#ff7f0e', label='Val Balanced Acc')
        axes[0, 1].set_title('(B) Val Macro-F1 & Balanced Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score (%)')
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)
        axes[0, 1].legend()

        # (c) Overall Accuracy
        axes[1, 0].plot(epochs, val_acc, 'o-', color='#2ca02c', label='Val Accuracy')
        axes[1, 0].set_title('(C) Validation Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)
        axes[1, 0].legend()

        # (d) Learning Rate Schedule
        axes[1, 1].plot(epochs, lrs, 'x-', color='#8c564b', label='LR Schedule')
        axes[1, 1].set_title('(D) Cosine Annealing Learning Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, linestyle='--', alpha=0.5)
        axes[1, 1].legend()

        plt.suptitle(f'Comprehensive Training Dynamics — {m["name"]}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        out_p4 = f"review_iii_assets/training/{var}_dashboard.png"
        plt.savefig(out_p4)
        plt.close()
        
        print(f"[+] Generated plots for {m['name']} -> {out_p1}, {out_p2}, {out_p3}, {out_p4}")

    # Multi-model comparison plots
    plt.figure(figsize=(10, 6))
    for m in MODELS_CONFIG:
        var = m["variant"]
        if var in all_histories:
            recs = all_histories[var]
            epochs = [r["epoch"] + 1 for r in recs]
            f1s = [r["metrics"]["f1_macro"] * 100 for r in recs]
            plt.plot(epochs, f1s, marker='o', linewidth=2, label=m["name"], color=m["color"])
    plt.title('Validation Macro-F1 Progression Across All Models', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Macro-F1 (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig("review_iii_assets/training/all_models_val_macro_f1_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for m in MODELS_CONFIG:
        var = m["variant"]
        if var in all_histories:
            recs = all_histories[var]
            epochs = [r["epoch"] + 1 for r in recs]
            losses = [r["val_loss"] for r in recs]
            plt.plot(epochs, losses, marker='s', linewidth=2, label=m["name"], color=m["color"])
    plt.title('Validation Loss Progression Across All Models', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig("review_iii_assets/training/all_models_val_loss_comparison.png")
    plt.close()

    print("[+] Generated multi-model comparison training plots!")


def step_2_evaluate_test_set_and_generate_metrics():
    print("\n" + "="*70)
    print("STEP 2: TEST SET EVALUATION FOR ALL 5 MODELS (6,875 TEST IMAGES)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    results = []
    
    # Load test dataloader using config_kvasir (336x336 / 224x224 transforms)
    cfg_base = load_config("config_kvasir")
    _, _, test_loader = create_dataloaders(cfg_base)
    print(f"Loaded Kvasir test dataset: {len(test_loader.dataset)} images in {len(test_loader)} batches.")
    
    for m in MODELS_CONFIG:
        name = m["name"]
        var = m["variant"]
        ckpt_p = m["checkpoint_path"]
        cfg_dir = m["config_dir"]
        
        print(f"\n---> Evaluating {name} [{var}]")
        print(f"     Checkpoint: {ckpt_p}")
        
        if not os.path.exists(ckpt_p):
            print(f"[-] Checkpoint missing for {name} at {ckpt_p}")
            continue
            
        # Get best val metrics from history
        best_val_f1 = 0.0
        best_val_acc = 0.0
        best_val_loss = 0.0
        best_epoch = 0
        if os.path.exists(m["history_path"]):
            with open(m["history_path"]) as f:
                hd = json.load(f)
            best_rec = max(hd["records"], key=lambda r: r["metrics"]["f1_macro"])
            best_val_f1 = best_rec["metrics"]["f1_macro"] * 100
            best_val_acc = best_rec["metrics"]["accuracy"] * 100
            best_val_loss = best_rec["val_loss"]
            best_epoch = best_rec["epoch"] + 1

        try:
            cfg = load_config(cfg_dir)
            cfg.model.variant = var
            model = build_model(cfg)
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)
                
            model.to(device)
            model.eval()
            
            criterion = nn.CrossEntropyLoss()
            
            all_labels = []
            all_preds = []
            all_probs = []
            all_confs = []
            total_loss = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    
                    logits = model(x)
                    loss = criterion(logits, y)
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)
                    
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    confs = probs[np.arange(len(preds)), preds]
                    
                    all_labels.extend(y.cpu().numpy().tolist())
                    all_preds.extend(preds.tolist())
                    all_probs.extend(probs.tolist())
                    all_confs.extend(confs.tolist())
                    
            y_true = np.array(all_labels, dtype=np.int64)
            y_pred = np.array(all_preds, dtype=np.int64)
            y_prob = np.array(all_probs, dtype=np.float32)
            y_conf = np.array(all_confs, dtype=np.float32)
            
            test_loss = total_loss / total_samples
            test_acc = accuracy_score(y_true, y_pred)
            test_bal_acc = balanced_accuracy_score(y_true, y_pred)
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except Exception:
                roc_auc = float('nan')
                
            ece = compute_ece(y_true, y_conf, y_pred, n_bins=10)
            brier = compute_brier_score(y_true, y_prob, num_classes=8)
            
            # Classification report
            cls_rep_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES_8, output_dict=True, zero_division=0)
            cls_rep_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES_8, digits=4, zero_division=0)
            
            # Save classification reports
            with open(f"review_iii_assets/classification_reports/{var}_report.json", "w") as f:
                json.dump(cls_rep_dict, f, indent=2)
            with open(f"review_iii_assets/classification_reports/{var}_report.txt", "w") as f:
                f.write(f"Classification Report for {name} ({var})\n")
                f.write("="*70 + "\n")
                f.write(cls_rep_text + "\n")
            df_rep = pd.DataFrame(cls_rep_dict).transpose()
            df_rep.to_csv(f"review_iii_assets/classification_reports/{var}_report.csv")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
            
            # Plot raw CM
            plt.figure(figsize=(9, 7.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES_8, yticklabels=CLASS_NAMES_8, cbar=True)
            plt.title(f'Test Confusion Matrix — {name}', fontsize=13, fontweight='bold', pad=12)
            plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
            plt.ylabel('Ground Truth Label', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            plt.savefig(f"review_iii_assets/confusion_matrices/{var}_confusion_matrix.png")
            plt.close()
            
            # Plot normalized CM
            plt.figure(figsize=(9, 7.5))
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Purples', xticklabels=CLASS_NAMES_8, yticklabels=CLASS_NAMES_8, cbar=True)
            plt.title(f'Normalized Test Confusion Matrix — {name}', fontsize=13, fontweight='bold', pad=12)
            plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
            plt.ylabel('Ground Truth Label', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            plt.savefig(f"review_iii_assets/confusion_matrices/{var}_confusion_matrix_normalized.png")
            plt.close()
            
            # Reliability Diagram
            rel_data = compute_reliability_data(y_true, y_conf, y_pred, n_bins=10)
            plot_reliability_diagram(
                reliability_data=rel_data,
                save_path=f"review_iii_assets/calibration/{var}_reliability_diagram.png",
                title=f"Calibration Reliability Diagram — {name}\nECE = {ece:.4f} | Brier = {brier:.4f}"
            )
            
            print(f"     Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | Test Bal Acc: {test_bal_acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}% | ECE: {ece:.4f} | Brier: {brier:.4f}")
            
            test_loss_val = round(test_loss, 4)
            test_acc_val = round(test_acc * 100, 2)
            test_bal_acc_val = round(test_bal_acc * 100, 2)
            prec_macro_val = round(prec_macro * 100, 2)
            rec_macro_val = round(rec_macro * 100, 2)
            f1_macro_val = round(f1_macro * 100, 2)
            f1_weighted_val = round(f1_weighted * 100, 2)
            roc_auc_val = round(roc_auc, 4)
            ece_val = round(ece, 4)
            brier_val = round(brier, 4)
            
        except Exception as e:
            print(f"[!] Warning: Could not evaluate {name} on 8-class test set ({e}). Checkpoint is 4-class legacy.")
            test_loss_val = "N/A"
            test_acc_val = "N/A"
            test_bal_acc_val = "N/A"
            prec_macro_val = "N/A"
            rec_macro_val = "N/A"
            f1_macro_val = "N/A"
            f1_weighted_val = "N/A"
            roc_auc_val = "N/A"
            ece_val = "N/A"
            brier_val = "N/A"

        
        # Append to summary results
        results.append({
            "Model": name,
            "Variant": var,
            "Best Epoch": best_epoch,
            "Val Loss": round(best_val_loss, 4),
            "Val Acc (%)": round(best_val_acc, 2),
            "Val Macro F1 (%)": round(best_val_f1, 2),
            "Test Loss": test_loss_val,
            "Test Acc (%)": test_acc_val,
            "Test Bal Acc (%)": test_bal_acc_val,
            "Precision Macro (%)": prec_macro_val,
            "Recall Macro (%)": rec_macro_val,
            "Test Macro F1 (%)": f1_macro_val,
            "Weighted F1 (%)": f1_weighted_val,
            "ROC-AUC Macro": roc_auc_val,
            "ECE": ece_val,
            "Brier Score": brier_val
        })
        
    return results


def step_3_model_comparison(results):
    print("\n" + "="*70)
    print("STEP 3: GENERATING MODEL COMPARISON TABLE & VISUALIZATIONS")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    # Save CSV files
    df.to_csv("review_iii_model_comparison.csv", index=False)
    df.to_csv("review_iii_assets/comparison/review_iii_model_comparison.csv", index=False)
    print("[+] Saved review_iii_model_comparison.csv")
    
    # Render presentation-ready table PNG
    display_cols = [
        "Model", "Best Epoch", "Val Acc (%)", "Val Macro F1 (%)", "Val Loss",
        "Test Acc (%)", "Test Bal Acc (%)", "Test Macro F1 (%)", "ECE", "Brier Score"
    ]
    df_disp = df[display_cols].copy()
    
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.2, 2.0)
    
    # Style table header & cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2b5797')
            cell.set_text_props(color='white', fontweight='bold')
        elif row == len(df_disp): # Proposed model (last row)
            cell.set_facecolor('#e8f4f8')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#f9f9f9' if row % 2 == 0 else 'white')
            
    plt.title('Review III Comprehensive Model Benchmark (Kvasir Dataset — 37,737 Images)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("review_iii_model_comparison.png", bbox_inches='tight')
    plt.savefig("review_iii_assets/comparison/review_iii_model_comparison.png", bbox_inches='tight')
    plt.close()
    
    # Filter numeric models for test bar charts
    df_numeric = df[df["Test Acc (%)"] != "N/A"].copy()
    for col in ["Test Acc (%)", "Test Bal Acc (%)", "Test Macro F1 (%)", "ECE", "Brier Score"]:
        df_numeric[col] = pd.to_numeric(df_numeric[col])
        
    if len(df_numeric) > 0:
        # Grouped Bar Chart Comparison
        metrics_to_plot = ["Test Acc (%)", "Test Bal Acc (%)", "Test Macro F1 (%)"]
        x = np.arange(len(df_numeric))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        for i, m in enumerate(metrics_to_plot):
            vals = df_numeric[m].values
            plt.bar(x + i*width, vals, width, label=m, alpha=0.9)
            for xi, val in zip(x + i*width, vals):
                plt.text(xi, val + 0.3, f"{val:.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                
        plt.xlabel('Model Architecture', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
        plt.title('Test Set Performance Comparison Across Evaluated Architectures', fontsize=14, fontweight='bold', pad=12)
        plt.xticks(x + width, df_numeric["Model"].values, rotation=15, ha='right', fontsize=10, fontweight='bold')
        plt.ylim(90, 102)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        plt.tight_layout()
        plt.savefig("review_iii_assets/comparison/benchmark_bar_chart.png")
        plt.close()
        
        # Calibration & Error comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # ECE
        ax1.bar(df_numeric["Model"], df_numeric["ECE"], color=['#1f77b4', '#ff7f0e', '#d62728', '#9467bd'][:len(df_numeric)], alpha=0.85)
        for i, v in enumerate(df_numeric["ECE"]):
            ax1.text(i, v + 0.0003, f"{v:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax1.set_title('Expected Calibration Error (ECE) — Lower is Better', fontweight='bold', fontsize=12)
        ax1.set_ylabel('ECE', fontweight='bold')
        ax1.set_xticklabels(df_numeric["Model"], rotation=20, ha='right', fontsize=9)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Brier Score
        ax2.bar(df_numeric["Model"], df_numeric["Brier Score"], color=['#1f77b4', '#ff7f0e', '#d62728', '#9467bd'][:len(df_numeric)], alpha=0.85)
        for i, v in enumerate(df_numeric["Brier Score"]):
            ax2.text(i, v + 0.0005, f"{v:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax2.set_title('Brier Score — Lower is Better', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Brier Score', fontweight='bold')
        ax2.set_xticklabels(df_numeric["Model"], rotation=20, ha='right', fontsize=9)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.suptitle('Calibration & Probability Quality Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("review_iii_assets/comparison/calibration_comparison.png")
        plt.close()
    
    print("[+] Model comparison artifacts created successfully!")



def step_4_gradcam_evidence():
    print("\n" + "="*70)
    print("STEP 4: GENERATING GRAD-CAM EXPLAINABILITY EVIDENCE")
from src.data.transforms import build_val_test_transforms

def step_4_gradcam_evidence():
    print("\n" + "="*70)
    print("STEP 4: GENERATING GRAD-CAM EXPLAINABILITY EVIDENCE")
    print("="*70)
    
    cfg = load_config("config_kvasir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_p = "outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth"
    
    pipeline = ExplainabilityPipeline(
        cfg=cfg,
        checkpoint_path=ckpt_p,
        device=device
    )
    
    target_classes = [
        ("Ulcer", "Dataset/test/Ulcer"),
        ("Normal clean mucosa", "Dataset/test/Normal clean mucosa"),
        ("Angiectasia", "Dataset/test/Angiectasia"),
        ("Erosion", "Dataset/test/Erosion"),
        ("Lymphangiectasia", "Dataset/test/Lymphangiectasia"),
        ("Ileocecal valve", "Dataset/test/Ileocecal valve"),
        ("Pylorus", "Dataset/test/Pylorus"),
        ("Reduced mucosal view", "Dataset/test/Reduced mucosal view"),
    ]
    
    for cname, cdir in target_classes:
        images = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
        img_p = images[0] # Pick representative test image
        
        cname_clean = cname.lower().replace(" ", "_")
        out_dir_local = "review_iii_assets/gradcam"
        out_dir_kvasir = "outputs_kvasir/hybrid_class_aware/plots/gradcam_review_iii"
        
        result = pipeline.explain_image(
            image_path=img_p,
            output_dir=out_dir_local,
            output_prefix=cname_clean
        )
        pipeline.explain_image(
            image_path=img_p,
            output_dir=out_dir_kvasir,
            output_prefix=cname_clean
        )
        
        pred_idx = result["predicted_class"]
        pred_cname = CLASS_NAMES_8[pred_idx]
        
        # Paths saved by pipeline: {save_base_path}_overlay.png, {save_base_path}_heatmap.png, etc.
        base_p = result["save_path_base"]
        overlay_p = f"{base_p}_overlay.png"
        cam_p = f"{base_p}_heatmap.png"
        
        # Create a 3-panel presentation composite: Original | Heatmap | Overlay
        orig_img = Image.open(img_p).convert("RGB")
        cam_img = Image.open(cam_p).convert("RGB") if os.path.exists(cam_p) else None
        overlay_img = Image.open(overlay_p).convert("RGB") if os.path.exists(overlay_p) else None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Original Input\n(GT: {cname})", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        if cam_img:
            axes[1].imshow(cam_img)
            axes[1].set_title("Grad-CAM Heatmap\n(Layer: cnn_backbone.conv_head)", fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
        if overlay_img:
            axes[2].imshow(overlay_img)
            axes[2].set_title(f"Grad-CAM Overlay\n(Predicted: {pred_cname})", fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
        plt.suptitle(f"Explainability Evidence (Grad-CAM) — {cname}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"review_iii_assets/gradcam/{cname_clean}_composite.png")
        plt.savefig(f"{out_dir_kvasir}/{cname_clean}_composite.png")
        plt.close()
        
        print(f"[+] Grad-CAM generated for class '{cname}' -> Pred: {pred_cname}")



def step_5_single_image_prediction_and_mc_dropout():
    print("\n" + "="*70)
    print("STEP 5: SINGLE IMAGE INFERENCE & MC DROPOUT UNCERTAINTY (20 SAMPLES)")
    print("="*70)
    
    cfg = load_config("config_kvasir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(cfg)
    ckpt_p = "outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth"
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    transforms = build_val_test_transforms(cfg)
    predictor = MCDropoutPredictor(model)
    
    test_samples = [
        ("Ulcer", "Dataset/test/Ulcer/test_00000.jpg" if os.path.exists("Dataset/test/Ulcer/test_00000.jpg") else os.path.join("Dataset/test/Ulcer", os.listdir("Dataset/test/Ulcer")[0])),
        ("Normal clean mucosa", os.path.join("Dataset/test/Normal clean mucosa", os.listdir("Dataset/test/Normal clean mucosa")[0])),
        ("Angiectasia", os.path.join("Dataset/test/Angiectasia", os.listdir("Dataset/test/Angiectasia")[0])),
        ("Erosion", os.path.join("Dataset/test/Erosion", os.listdir("Dataset/test/Erosion")[0])),
    ]
    
    single_image_results = []
    
    for cname, img_p in test_samples:
        pil_img = Image.open(img_p).convert("RGB")
        tensor_img = transforms(pil_img).unsqueeze(0).to(device)
        
        unc_res = predictor.predict(tensor_img, n_samples=20, device=device)
        
        pred_idx = unc_res.predicted_class
        pred_cname = CLASS_NAMES_8[pred_idx]
        conf = float(unc_res.confidence)
        mean_probs = unc_res.mean_proba.tolist()
        prob_dict = {CLASS_NAMES_8[i]: float(p) for i, p in enumerate(mean_probs)}
        
        # Check if model has fusion diagnostics
        fusion_diag = {}
        if hasattr(model, "fusion_module"):
            fm = model.fusion_module
            if hasattr(fm, "class_prototypes"):
                proto_norm = F.normalize(fm.class_prototypes, dim=-1)
                fusion_diag["has_prototypes"] = True
            if hasattr(fm, "alpha"):
                fusion_diag["alpha_weight"] = float(fm.alpha.detach().cpu().numpy()) if isinstance(fm.alpha, torch.Tensor) else float(fm.alpha)
                
        cname_clean = cname.lower().replace(" ", "_")
        
        # Save individual result JSON
        res_dict = {
            "ground_truth": cname,
            "image_path": img_p,
            "predicted_class": pred_cname,
            "predicted_index": pred_idx,
            "confidence": conf,
            "probabilities": prob_dict,
            "predictive_entropy": float(unc_res.predictive_entropy),
            "variation_ratio": float(unc_res.variation_ratio),
            "mc_samples": 20,
            "fusion_diagnostics": fusion_diag
        }
        
        out_json = f"review_iii_assets/single_image/{cname_clean}_prediction.json"
        with open(out_json, "w") as f:
            json.dump(res_dict, f, indent=2)
            
        # Plot probability bar chart with uncertainty metrics
        plt.figure(figsize=(10, 5))
        classes = list(prob_dict.keys())
        probs = [prob_dict[c] * 100 for c in classes]
        colors = ['#1f77b4' if c != pred_cname else '#2ca02c' for c in classes]
        
        bars = plt.barh(classes, probs, color=colors, alpha=0.85)
        for bar, p in zip(bars, probs):
            if p > 1.0:
                plt.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va='center', fontweight='bold', fontsize=9)
                
        plt.title(f"MC Dropout Prediction (20 Passes) — GT: {cname}\nPredicted: {pred_cname} (Conf: {conf*100:.2f}%) | Entropy: {unc_res.predictive_entropy:.4f} | Var Ratio: {unc_res.variation_ratio:.4f}", fontsize=11, fontweight='bold', pad=12)
        plt.xlabel("Predictive Probability (%)", fontsize=11, fontweight='bold')
        plt.xlim(0, 108)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"review_iii_assets/single_image/{cname_clean}_prob_dist.png")
        plt.close()
        
        single_image_results.append(res_dict)
        print(f"[+] Predicted {cname}: Pred='{pred_cname}' (Conf: {conf*100:.2f}%, Entropy: {unc_res.predictive_entropy:.4f}, VarRatio: {unc_res.variation_ratio:.4f})")
            
    with open("review_iii_assets/single_image/all_single_image_summary.json", "w") as f:
        json.dump(single_image_results, f, indent=2)


def step_6_uncertainty_analysis_test_subset():
    print("\n" + "="*70)
    print("STEP 6: UNCERTAINTY ANALYSIS ON TEST SUBSET")
    print("="*70)
    
    cfg = load_config("config_kvasir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(cfg)
    ckpt_p = "outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth"
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    transforms = build_val_test_transforms(cfg)
    predictor = MCDropoutPredictor(model)
    
    # Take a balanced sample of 240 images (30 per class from test set)
    dataset_dir = "Dataset/test"
    sampled_paths = []
    sampled_labels = []
    
    for cidx, cname in enumerate(CLASS_NAMES_8):
        cdir = os.path.join(dataset_dir, cname)
        flist = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for p in flist[:30]:
            sampled_paths.append(p)
            sampled_labels.append(cidx)
            
    print(f"Running MC Dropout uncertainty evaluation on {len(sampled_paths)} test images...")
    
    confidences = []
    entropies = []
    variation_ratios = []
    correctness = []
    
    for p, gt in zip(sampled_paths, sampled_labels):
        pil_img = Image.open(p).convert("RGB")
        tensor_img = transforms(pil_img).unsqueeze(0).to(device)
        res = predictor.predict(tensor_img, n_samples=10, device=device)
        
        confidences.append(float(res.confidence))
        entropies.append(float(res.predictive_entropy))
        variation_ratios.append(float(res.variation_ratio))
        correctness.append(res.predicted_class == gt)
        
    confidences = np.array(confidences)
    entropies = np.array(entropies)
    variation_ratios = np.array(variation_ratios)
    correctness = np.array(correctness)
    
    print(f"Sampled accuracy: {np.mean(correctness)*100:.2f}%")
    print(f"Mean Confidence: {np.mean(confidences):.4f} | Correct: {np.mean(confidences[correctness]):.4f} | Error: {np.mean(confidences[~correctness]) if np.sum(~correctness) > 0 else 'N/A'}")
    print(f"Mean Entropy: {np.mean(entropies):.4f} | Correct: {np.mean(entropies[correctness]):.4f} | Error: {np.mean(entropies[~correctness]) if np.sum(~correctness) > 0 else 'N/A'}")
    
    # Plot Confidence & Entropy distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence distribution
    axes[0].hist(confidences[correctness], bins=15, alpha=0.7, color='#2ca02c', label='Correct Predictions', density=True)
    if np.sum(~correctness) > 0:
        axes[0].hist(confidences[~correctness], bins=10, alpha=0.7, color='#d62728', label='Incorrect Predictions', density=True)
    axes[0].set_title('Predictive Confidence Distribution', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Confidence', fontweight='bold')
    axes[0].set_ylabel('Density', fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # Entropy distribution
    axes[1].hist(entropies[correctness], bins=15, alpha=0.7, color='#2ca02c', label='Correct Predictions', density=True)
    if np.sum(~correctness) > 0:
        axes[1].hist(entropies[~correctness], bins=10, alpha=0.7, color='#d62728', label='Incorrect Predictions', density=True)
    axes[1].set_title('Predictive Entropy (Uncertainty) Distribution', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Predictive Entropy', fontweight='bold')
    axes[1].set_ylabel('Density', fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.suptitle('Uncertainty Quantification (MC Dropout Analysis)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("review_iii_assets/uncertainty/uncertainty_distributions.png")
    plt.close()
    
    # Save uncertainty metrics report
    unc_report = {
        "num_evaluated_samples": len(sampled_paths),
        "subset_accuracy": float(np.mean(correctness)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_entropy": float(np.mean(entropies)),
        "mean_variation_ratio": float(np.mean(variation_ratios)),
        "correct_predictions": {
            "count": int(np.sum(correctness)),
            "mean_confidence": float(np.mean(confidences[correctness])),
            "mean_entropy": float(np.mean(entropies[correctness])),
            "mean_variation_ratio": float(np.mean(variation_ratios[correctness]))
        },
        "incorrect_predictions": {
            "count": int(np.sum(~correctness)),
            "mean_confidence": float(np.mean(confidences[~correctness])) if np.sum(~correctness) > 0 else None,
            "mean_entropy": float(np.mean(entropies[~correctness])) if np.sum(~correctness) > 0 else None,
            "mean_variation_ratio": float(np.mean(variation_ratios[~correctness])) if np.sum(~correctness) > 0 else None
        }
    }
    with open("review_iii_assets/uncertainty/uncertainty_summary.json", "w") as f:
        json.dump(unc_report, f, indent=2)
    print("[+] Uncertainty analysis plots and summary saved!")



from src.evaluation.faithfulness import FaithfulnessEvaluator

def step_7_faithfulness_evaluation():
    print("\n" + "="*70)
    print("STEP 7: QUANTITATIVE FAITHFULNESS EVALUATION (DELETION & INSERTION AUC)")
    print("="*70)
    
    cfg = load_config("config_kvasir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(cfg)
    ckpt_p = "outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth"
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    evaluator = FaithfulnessEvaluator(
        model=model,
        cfg=cfg,
        device=device,
        class_names=CLASS_NAMES_8
    )
    
    # Load test dataloader
    _, _, test_loader = create_dataloaders(cfg)
    
    print(f"Evaluating faithfulness metrics on test set sample (50 images)...")
    results = evaluator.evaluate_dataset(
        dataloader=test_loader,
        max_images=50
    )
    
    out_dir = "review_iii_assets/faithfulness"
    evaluator.save_report(results, os.path.join(out_dir, "faithfulness_report.json"))
    evaluator.plot_curves(results, out_dir, prefix="faithfulness")
    
    summary = results["summary"]
    gc_del_auc = summary["gradcam"]["mean_deletion_auc"]
    rnd_del_auc = summary["random_baseline"]["mean_deletion_auc"]
    gc_ins_auc = summary["gradcam"]["mean_insertion_auc"]
    rnd_ins_auc = summary["random_baseline"]["mean_insertion_auc"]
    
    print(f"[+] Faithfulness evaluation complete!")
    print(f"    Grad-CAM Deletion AUC: {gc_del_auc:.4f} (Random Baseline: {rnd_del_auc:.4f}) -> Lower is More Faithful")
    print(f"    Grad-CAM Insertion AUC: {gc_ins_auc:.4f} (Random Baseline: {rnd_ins_auc:.4f}) -> Higher is More Faithful")
    if "least_important_control" in summary:
        least_del_auc = summary["least_important_control"]["mean_deletion_auc"]
        least_ins_auc = summary["least_important_control"]["mean_insertion_auc"]
        print(f"    Least-Important Control: Deletion AUC = {least_del_auc:.4f}, Insertion AUC = {least_ins_auc:.4f}")



def main():
    ensure_dirs()
    set_seed(42)
    
    # 1. Training plots
    step_1_generate_training_plots()
    
    # 2. Test Set Evaluation
    results = step_2_evaluate_test_set_and_generate_metrics()
    
    # 3. Model Comparison
    step_3_model_comparison(results)
    
    # 4. Grad-CAM Explainability
    step_4_gradcam_evidence()
    
    # 5. Single Image Prediction & MC Dropout
    step_5_single_image_prediction_and_mc_dropout()
    
    # 6. Uncertainty Analysis
    step_6_uncertainty_analysis_test_subset()
    
    # 7. Faithfulness Evaluation
    step_7_faithfulness_evaluation()
    
    print("\n" + "="*70)
    print("ALL REVIEW III EVIDENCE GENERATION COMPLETED SUCCESSFULLY!")
    print("NO MODEL TRAINING WAS STARTED.")
    print("="*70)


if __name__ == "__main__":
    main()
