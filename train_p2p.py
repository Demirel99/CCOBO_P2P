#train.py
"""
Iterative Crowd Counting Model Training Script (Coordinate Regression Hybrid)
"""
import os
import torch
import torch.optim as optim
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
from losses_p2p import coordinate_regression_loss # MODIFIED

def log_hyperparameters(log_file_path, config_module):
    """Logs hyperparameters from the config module to the log file."""
    hyperparams = {
        key: getattr(config_module, key)
        for key in dir(config_module)
        if not key.startswith("__") and not callable(getattr(config_module, key))
        and isinstance(getattr(config_module, key), (int, float, str, list, dict, tuple, bool, torch.device)) # Filter for common types
    }
    # Convert device to string for JSON serialization
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
    
    # Log hyperparameters first
    import config_p2p as cfg # Import config module to pass to logger
    log_hyperparameters(LOG_FILE_PATH, cfg)


    sorted_image_paths_train_val = find_and_sort_paths(IMAGE_DIR_TRAIN_VAL, '*.jpg')
    # For GT paths, dataset.py handles matching, so a placeholder list or actual list can be used
    # If GTs are not 1-to-1 named with images, ensure dataset.py logic for finding GT is robust
    # For this example, assume find_and_sort_paths for GTs is also somewhat meaningful or dataset.py's fallback works.
    sorted_gt_paths_train_val = find_and_sort_paths(GT_DIR_TRAIN_VAL, '*.mat') 
    if not sorted_gt_paths_train_val: # If GT dir is empty/not found, provide a dummy list for split_train_val
        print("Warning: GT paths list is empty. `split_train_val` might behave unexpectedly if it relies on len(gt_paths).")
        sorted_gt_paths_train_val = [None] * len(sorted_image_paths_train_val)


    if not sorted_image_paths_train_val: # Removed check for sorted_gt_paths_train_val here as it might be dummy
        raise FileNotFoundError("Training/Validation images not found. Check paths in config.py.")

    train_image_paths, train_gt_paths, val_image_paths, val_gt_paths = split_train_val(
        sorted_image_paths_train_val, sorted_gt_paths_train_val, val_ratio=0.1, seed=SEED
    )
    if not train_image_paths or not val_image_paths:
        raise ValueError("Train or validation set is empty after splitting.")

    model = VGG19FPNASPP().to(DEVICE)
    
    # For potential future improvement: Different learning rates for backbone vs. other parts
    # Example:
    # optimizer = optim.AdamW([
    #     {'params': model.image_encoder.parameters(), 'lr': LEARNING_RATE * 0.1},
    #     {'params': model.mask_encoder.parameters()},
    #     {'params': model.fusion_conv_c5.parameters()},
    #     {'params': model.aspp_c5.parameters()},
    #     {'params': model.fpn_decoder.parameters()},
    #     {'params': model.coordinate_head.parameters()}
    # ], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=SCHEDULER_PATIENCE, verbose=True)
    
    criterion = coordinate_regression_loss # MODIFIED (function itself, not instance if stateless)
                                         # Or if it's a class: criterion = CoordinateRegressionLoss().to(DEVICE)
                                         # Our current loss is a function, so this is fine.

    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print("Using Automatic Mixed Precision (AMP).")

    best_val_loss = float('inf')
    iterations_list, train_loss_list, val_loss_list = [], [], []
    # Log file is now handled by log_hyperparameters and append mode later

    print("Starting training...")
    pbar = tqdm(range(1, TOTAL_ITERATIONS + 1), desc=f"Iteration 1/{TOTAL_ITERATIONS}", unit="iter")
    train_loss_accum, samples_in_accum = 0.0, 0

    for iteration in pbar:
        model.train()
        # MODIFIED: tgt_psf_batch -> tgt_coords_batch
        img_batch, in_psf_batch, tgt_coords_batch = generate_batch(
            train_image_paths, train_gt_paths, BATCH_SIZE,
            generation_fn=generate_train_sample,
            augment_size=AUGMENTATION_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA # Still used for input PSF generation
        )

        if img_batch is None:
            print(f"Warning: Failed to generate training batch at iter {iteration}. Skipping.")
            continue

        img_batch = img_batch.to(DEVICE)
        in_psf_batch = in_psf_batch.to(DEVICE)
        tgt_coords_batch = tgt_coords_batch.to(DEVICE) # MODIFIED

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            predicted_coords = model(img_batch, in_psf_batch) # MODIFIED output name
            loss = criterion(predicted_coords, tgt_coords_batch) # MODIFIED

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_accum += loss.item() * img_batch.size(0)
        samples_in_accum += img_batch.size(0)

        if iteration == 1 and BATCH_SIZE > 0: # Print for the first batch
            print("Sample Predicted Coords (first 5):", predicted_coords.detach().cpu().numpy()[:5])
            print("Sample Target Coords (first 5):", tgt_coords_batch.detach().cpu().numpy()[:5])

        if iteration % VALIDATION_INTERVAL == 0:
            avg_train_loss = train_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0
            rng_state = {'random': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            
            model.eval()
            total_val_loss, total_val_samples = 0.0, 0
            with torch.no_grad():
                for i in range(VALIDATION_BATCHES):
                    # Validation set seeding: The current method set_seed(SEED + iteration + i)
                    # re-seeds for each validation batch. This means augmentations for validation
                    # samples can vary slightly across validation phases. This can provide a more
                    # robust average validation score over time.
                    # If perfectly identical validation augmentations are needed for each validation
                    # phase, seed once before the validation loop and ensure dataset generation is deterministic.
                    set_seed(SEED + iteration + i) 
                    
                    # MODIFIED: val_tgt_psf -> val_tgt_coords
                    val_img, val_in_psf, val_tgt_coords = generate_batch(
                        val_image_paths, val_gt_paths, BATCH_SIZE,
                        generation_fn=generate_train_sample,
                        augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE, psf_sigma=GT_PSF_SIGMA
                    )
                    if val_img is None: continue
                    val_img, val_in_psf, val_tgt_coords = val_img.to(DEVICE), val_in_psf.to(DEVICE), val_tgt_coords.to(DEVICE) # MODIFIED

                    with autocast(enabled=use_amp):
                        val_pred_coords = model(val_img, val_in_psf) # MODIFIED output name
                        batch_loss = criterion(val_pred_coords, val_tgt_coords) # MODIFIED
                    
                    total_val_loss += batch_loss.item() * val_img.size(0)
                    total_val_samples += val_img.size(0)

            random.setstate(rng_state['random']); np.random.set_state(rng_state['numpy']); torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng_state['cuda'])
            set_seed(SEED + iteration + VALIDATION_BATCHES + 1) # Re-seed for subsequent training

            average_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else float('inf')
            iterations_list.append(iteration); train_loss_list.append(avg_train_loss); val_loss_list.append(average_val_loss)

            log_message = (f"Iter [{iteration}/{TOTAL_ITERATIONS}] | Train Loss: {avg_train_loss:.4f} | "
                           f"Val Loss: {average_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.4e}")
            print(f"\n{log_message}")

            with open(LOG_FILE_PATH, "a") as log_file: log_file.write(log_message + "\n")
            
            scheduler.step(average_val_loss)
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"    -> New best model saved with Val Loss: {best_val_loss:.4f}")
            
            train_loss_accum, samples_in_accum = 0.0, 0
        pbar.set_description(f"Iter {iteration}/{TOTAL_ITERATIONS} | Last Batch Loss: {loss.item():.4f}")

    print("Training complete!")
    pbar.close()

    plt.figure(figsize=(10, 5))
    plt.plot(iterations_list, train_loss_list, label='Train Loss')
    plt.plot(iterations_list, val_loss_list, label='Validation Loss')
    plt.title("Training and Validation Loss (Coordinate Regression) over Iterations") # MODIFIED
    plt.xlabel("Iteration")
    plt.ylabel("Coordinate Regression Loss (e.g., MSE)") # MODIFIED
    plt.legend(); plt.grid(True); plt.ylim(bottom=0); plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_loss_plot_coordreg.png") # MODIFIED
    plt.savefig(plot_path); plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Log file saved to: {LOG_FILE_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()