#train_p2p.py
"""
Iterative Crowd Counting Model Training Script (Coordinate Regression Hybrid)
"""
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import json # For logging config

from config_p2p import (
    DEVICE, SEED, TOTAL_ITERATIONS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    VALIDATION_INTERVAL, VALIDATION_BATCHES, SCHEDULER_PATIENCE, 
    IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, OUTPUT_DIR, LOG_FILE_PATH, BEST_MODEL_PATH,
    AUGMENTATION_SIZE, MODEL_INPUT_SIZE, GT_PSF_SIGMA
)
from utils import set_seed, find_and_sort_paths, split_train_val
from dataset_p2p import generate_batch, generate_train_sample
from model_p2p import VGG19FPNASPP
from losses_p2p import coordinate_regression_loss

def log_hyperparameters(log_file_path, config_module):
    """Logs hyperparameters from the config module to the log file."""
    hyperparams = {
        key: getattr(config_module, key)
        for key in dir(config_module)
        if not key.startswith("__") and not callable(getattr(config_module, key))
        and isinstance(getattr(config_module, key), (int, float, str, list, dict, tuple, bool, torch.device)) 
    }
    if 'DEVICE' in hyperparams and isinstance(hyperparams['DEVICE'], torch.device):
        hyperparams['DEVICE'] = str(hyperparams['DEVICE'])
        
    with open(log_file_path, "a") as log_file:
        log_file.write("--- Hyperparameters ---\n")
        log_file.write(json.dumps(hyperparams, indent=4))
        log_file.write("\n--- Training Log ---\n")

def train():
    print("Setting up training (Coordinate Regression Mode)...")
    set_seed(SEED)

    if os.path.exists(LOG_FILE_PATH):
        try: os.remove(LOG_FILE_PATH)
        except OSError as e: print(f"Warning: Could not remove existing log file: {e}")
    
    import config_p2p as cfg
    log_hyperparameters(LOG_FILE_PATH, cfg)

    sorted_image_paths_train_val = find_and_sort_paths(IMAGE_DIR_TRAIN_VAL, '*.jpg')
    sorted_gt_paths_train_val = find_and_sort_paths(GT_DIR_TRAIN_VAL, '*.mat') 
    if not sorted_gt_paths_train_val:
        print("Warning: GT paths list is empty. `split_train_val` might behave unexpectedly if it relies on len(gt_paths).")
        sorted_gt_paths_train_val = [None] * len(sorted_image_paths_train_val)

    if not sorted_image_paths_train_val:
        raise FileNotFoundError("Training/Validation images not found. Check paths in config.py.")

    train_image_paths, train_gt_paths, val_image_paths, val_gt_paths = split_train_val(
        sorted_image_paths_train_val, sorted_gt_paths_train_val, val_ratio=0.1, seed=SEED
    )
    if not train_image_paths or not val_image_paths:
        raise ValueError("Train or validation set is empty after splitting.")

    model = VGG19FPNASPP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_ITERATIONS, eta_min=1e-6) 
    print(f"Using CosineAnnealingLR scheduler with T_max={TOTAL_ITERATIONS}, eta_min=1e-6.")
    
    criterion = coordinate_regression_loss 
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print("Using Automatic Mixed Precision (AMP).")

    best_val_loss = float('inf')
    iterations_list, train_loss_list, val_loss_list = [], [], []

    print("Starting training...")
    pbar = tqdm(range(1, TOTAL_ITERATIONS + 1), desc=f"Iteration 1/{TOTAL_ITERATIONS}", unit="iter")
    train_loss_accum, samples_in_accum = 0.0, 0

    for iteration in pbar:
        model.train()
        img_batch, in_psf_batch, tgt_coords_batch = generate_batch(
            train_image_paths, train_gt_paths, BATCH_SIZE,
            generation_fn=generate_train_sample,
            augment_size=AUGMENTATION_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA
        )

        if img_batch is None:
            print(f"Warning: Failed to generate training batch at iter {iteration}. Skipping.")
            if scheduler: scheduler.step() 
            continue

        img_batch = img_batch.to(DEVICE)
        in_psf_batch = in_psf_batch.to(DEVICE)
        tgt_coords_batch = tgt_coords_batch.to(DEVICE)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            predicted_coords = model(img_batch, in_psf_batch)
            loss = criterion(predicted_coords, tgt_coords_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() 

        train_loss_accum += loss.item() * img_batch.size(0)
        samples_in_accum += img_batch.size(0)

        if iteration == 1 and BATCH_SIZE > 0 and predicted_coords is not None and tgt_coords_batch is not None:
            print("\n--- First Training Batch Coordinate Comparison (Normalized [0,1]) ---")
            num_to_print = min(5, predicted_coords.size(0))
            preds_to_print = predicted_coords.detach().cpu().numpy()[:num_to_print]
            tgts_to_print = tgt_coords_batch.detach().cpu().numpy()[:num_to_print]
            for i in range(num_to_print):
                print(f"  Sample {i+1}: Pred: [{preds_to_print[i,0]:.3f}, {preds_to_print[i,1]:.3f}], "
                      f"Target: [{tgts_to_print[i,0]:.3f}, {tgts_to_print[i,1]:.3f}]")
            print("------------------------------------------------------------")

        if iteration % VALIDATION_INTERVAL == 0:
            avg_train_loss = train_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0
            rng_state = {'random': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            
            model.eval()
            total_val_loss, total_val_samples = 0.0, 0
            total_val_pixel_error = 0.0  # Accumulator for pixel error
            printed_val_coords_this_interval = False # Flag to print coords once per validation

            with torch.no_grad():
                for val_batch_idx in range(VALIDATION_BATCHES):
                    set_seed(SEED + iteration + val_batch_idx) 
                    
                    val_img, val_in_psf, val_tgt_coords = generate_batch(
                        val_image_paths, val_gt_paths, BATCH_SIZE,
                        generation_fn=generate_train_sample,
                        augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE, psf_sigma=GT_PSF_SIGMA
                    )
                    if val_img is None: continue
                    val_img, val_in_psf, val_tgt_coords = val_img.to(DEVICE), val_in_psf.to(DEVICE), val_tgt_coords.to(DEVICE)

                    with autocast(enabled=use_amp):
                        val_pred_coords = model(val_img, val_in_psf)
                        batch_loss = criterion(val_pred_coords, val_tgt_coords)
                    
                    total_val_loss += batch_loss.item() * val_img.size(0)
                    total_val_samples += val_img.size(0)

                    # Calculate and accumulate pixel error
                    if MODEL_INPUT_SIZE > 1 and val_pred_coords is not None and val_tgt_coords is not None:
                        scale_factor = MODEL_INPUT_SIZE - 1
                        pred_px_x = val_pred_coords[:, 0] * scale_factor
                        pred_px_y = val_pred_coords[:, 1] * scale_factor
                        tgt_px_x = val_tgt_coords[:, 0] * scale_factor
                        tgt_px_y = val_tgt_coords[:, 1] * scale_factor
                        
                        pixel_errors_batch = torch.sqrt(
                            (pred_px_x - tgt_px_x)**2 + (pred_px_y - tgt_px_y)**2
                        )
                        total_val_pixel_error += pixel_errors_batch.sum().item()

                    # Print examples from the first validation batch of this interval
                    if not printed_val_coords_this_interval and val_pred_coords is not None and val_tgt_coords is not None and val_pred_coords.size(0) > 0:
                        print(f"\n--- Validation Coordinate Comparison (Iter {iteration}, Val Batch 1) ---")
                        num_to_print_val = min(5, val_pred_coords.size(0))
                        val_preds_norm = val_pred_coords.detach().cpu().numpy()[:num_to_print_val]
                        val_tgts_norm = val_tgt_coords.detach().cpu().numpy()[:num_to_print_val]
                        
                        for k_val in range(num_to_print_val):
                            pred_norm_x, pred_norm_y = val_preds_norm[k_val,0], val_preds_norm[k_val,1]
                            tgt_norm_x, tgt_norm_y = val_tgts_norm[k_val,0], val_tgts_norm[k_val,1]
                            
                            pred_px_str, tgt_px_str = "", ""
                            if MODEL_INPUT_SIZE > 1:
                                scale_f = MODEL_INPUT_SIZE - 1
                                pred_px_str = f"(Px: [{pred_norm_x*scale_f:.1f}, {pred_norm_y*scale_f:.1f}])"
                                tgt_px_str = f"(Px: [{tgt_norm_x*scale_f:.1f}, {tgt_norm_y*scale_f:.1f}])"

                            print(f"  Sample {k_val+1}: Pred_Norm: [{pred_norm_x:.3f}, {pred_norm_y:.3f}] {pred_px_str} --- "
                                  f"Target_Norm: [{tgt_norm_x:.3f}, {tgt_norm_y:.3f}] {tgt_px_str}")
                        print("-----------------------------------------------------------------------------------")
                        printed_val_coords_this_interval = True

            random.setstate(rng_state['random']); np.random.set_state(rng_state['numpy']); torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng_state['cuda'])
            set_seed(SEED + iteration + VALIDATION_BATCHES + 1)

            average_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else float('inf')
            average_val_pixel_error = total_val_pixel_error / total_val_samples if total_val_samples > 0 else float('inf')
            
            iterations_list.append(iteration); train_loss_list.append(avg_train_loss); val_loss_list.append(average_val_loss)

            log_message = (f"Iter [{iteration}/{TOTAL_ITERATIONS}] | Train Loss: {avg_train_loss:.4f} | "
                           f"Val Loss: {average_val_loss:.4f} | Val Pixel Err: {average_val_pixel_error:.2f}px | "
                           f"LR: {optimizer.param_groups[0]['lr']:.4e}")
            print(f"\n{log_message}")

            with open(LOG_FILE_PATH, "a") as log_file: log_file.write(log_message + "\n")
            
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"    -> New best model saved with Val Loss: {best_val_loss:.4f} (Avg Pixel Err: {average_val_pixel_error:.2f}px)")
            
            train_loss_accum, samples_in_accum = 0.0, 0
        pbar.set_description(f"Iter {iteration}/{TOTAL_ITERATIONS} | Last Batch Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    print("Training complete!")
    pbar.close()

    plt.figure(figsize=(10, 5))
    plt.plot(iterations_list, train_loss_list, label='Train Loss')
    plt.plot(iterations_list, val_loss_list, label='Validation Loss')
    plt.title("Training and Validation Loss (Coordinate Regression) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Coordinate Regression Loss (e.g., MSE)")
    plt.legend(); plt.grid(True); plt.ylim(bottom=0); plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_loss_plot_coordreg.png")
    plt.savefig(plot_path); plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Log file saved to: {LOG_FILE_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
