import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from datasets.coco import COCOTrainDataset
from models.tinyclip_student import TinyCLIPStudent
from models.clip_teacher import CLIPTeacher
from scripts.paths import Config
from scripts.seed_everything import set_seed
from scripts.plots import plot_loss_curve_dual
from losses.semantic_alignment_loss import SemanticAlignmentLoss

# Affinity Loss
def affinity_loss(student_embed: torch.Tensor, teacher_embed: torch.Tensor) -> torch.Tensor:
    assert student_embed.shape == teacher_embed.shape, \
        f"Shape mismatch: {student_embed.shape} vs {teacher_embed.shape}"
    student_embed = F.normalize(student_embed, dim=-1)
    teacher_embed = F.normalize(teacher_embed, dim=-1)
    cosine_sim = F.cosine_similarity(student_embed, teacher_embed, dim=-1)
    loss = 1 - cosine_sim.mean()
    return loss


# TokenSim Loss
def tokensim_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid_labels_mask = (labels != -1)
    if valid_labels_mask.sum() > 0:
        return F.cross_entropy(logits[valid_labels_mask], labels[valid_labels_mask])
    else:
        return torch.tensor(0.0).to(logits.device)

def main():
    cfg = Config()
    set_seed(cfg.get_seed())
    device = torch.device(cfg.get_device())
    print(f"[INFO] Using device: {device}")

    total_loss_history = []
    affinity_loss_history = []
    tokensim_loss_history = []
    sas_loss_history = []
    cosine_sim_history = []
    hyperparams = cfg.get_hyperparams()
    model_params = cfg.get_model_params()
    training_flags = cfg.get_training_flags()
    batch_size = hyperparams["batch_size"]
    num_epochs = hyperparams["num_epochs"]
    lr = hyperparams["learning_rate"]
    lambda_affinity = hyperparams["lambda_affinity"]
    lambda_tokensim = hyperparams["lambda_tokensim"]
    lambda_sas = hyperparams["lambda_sas"]
    enable_sas_after_epoch = training_flags["enable_sas_after_epoch"]
    image_size = model_params["image_size"]


    # Verify Stage 1 Configuration
    if lambda_tokensim != 0.0 or lambda_sas != 0.0 or enable_sas_after_epoch <= num_epochs:
        print(" Config not seem set for Stage 1.")
        print(f" lambda_tokensim=0.0, lambda_sas=0.0, enable_sas_after_epoch > num_epochs")
        print(
            f"  Found: lambda_tokensim={lambda_tokensim}, lambda_sas={lambda_sas}, enable_sas_after_epoch={enable_sas_after_epoch}")
        print(" Ensure config is correctly set for Stage 1.")


    # Path settings
    checkpoints_dir = cfg.get_output_path("checkpoints")
    logs_dir = cfg.get_output_path("logs")
    figures_dir = cfg.get_output_path("figures")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)


    # Log file path settings
    log_path_epoch = os.path.join(logs_dir, "train_log_epoch_stage1.txt")
    log_path_step = os.path.join(logs_dir, "train_tokensim_log_steps_stage1.txt")


    # Refresh log file
    with open(log_path_epoch, "w", encoding="utf-8") as f:
        f.write("")
    with open(log_path_step, "w", encoding="utf-8") as f:
        f.write("")


    # TensorBoard log settings
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_log_dir = os.path.join(logs_dir, "runs", "train_stage1", current_time)
    writer = SummaryWriter(tb_log_dir)
    print(f"[INFO] TensorBoard logs saved to: {tb_log_dir}")


    # Checkpoint Paths for Stage 1
    resume_checkpoint_path = os.path.join(checkpoints_dir, "last_TinyCLIP_student_stage1.pt")
    temp_best_model_path = os.path.join(checkpoints_dir,
                                        "stage1_temp_best_TinyCLIP_student.pt")
    final_stage1_best_model_path = os.path.join(checkpoints_dir,
                                                "TinyCLIP_student_stage1_best.pt")

    start_epoch = 0
    train_dataset = COCOTrainDataset(image_size=image_size)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(f"[INFO] Dataset loaded. Number of training images: {len(train_dataset)}")


    # Models
    teacher = CLIPTeacher().to(device).eval()
    student = TinyCLIPStudent(
        embed_dim=model_params["embed_dim"],
        num_classes=model_params["num_classes"],
        freeze_layers=model_params["freeze_layers"]
    ).to(device)
    print(f"[INFO] Student Model: {student}")
    print(f"[INFO] Teacher Model: {teacher}")


    # Stage 1 Model Loading Logic
    print("[INFO] Starting Stage 1 Training")
    if os.path.exists(resume_checkpoint_path):
        print(f"[INFO] Resuming Stage 1 training from: {resume_checkpoint_path}")
        student.load_state_dict(torch.load(resume_checkpoint_path, map_location=device))
    else:
        print("[INFO] No checkpoint found for Stage 1.")

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=hyperparams["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    # Initialize loss function
    feature_map_channels = 128
    embed_dim_for_sas_loss = model_params["embed_dim"]
    semantic_loss_fn = SemanticAlignmentLoss(feature_map_channels, embed_dim_for_sas_loss).to(device)
    scaler = GradScaler()
    print("[INFO] Starting training")


    # Early stopping variables
    best_metric = -float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 7
    min_delta_for_early_stopping = 0.0001

    for epoch in range(start_epoch, num_epochs):
        student.train()
        current_epoch_total_loss_sum = 0.0
        current_epoch_affinity_loss_sum = 0.0
        current_epoch_tokensim_loss_sum = 0.0
        current_epoch_sas_loss_sum = 0.0
        current_epoch_cosine_sim_sum = 0.0
        start_time_epoch = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Stage 1)")

        with open(log_path_step, "a", encoding="utf-8") as step_log_file:
            for step_idx, (images, texts_list, labels, img_ids) in enumerate(pbar):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with autocast():
                    student_img_embed, student_feature_map, student_logits = student(
                        images, return_features=True, return_logits=True
                    )

                    with torch.no_grad():
                        teacher_text_embed = teacher.forward_text(texts_list)
                    l_affinity = affinity_loss(student_img_embed, teacher_text_embed)
                    l_tokensim = tokensim_loss_fn(student_logits, labels)
                    l_sas = torch.tensor(0.0).to(device)

                    if epoch >= enable_sas_after_epoch:
                        l_sas = semantic_loss_fn(student_feature_map, student_img_embed)

                    loss = (
                            lambda_affinity * l_affinity +
                            lambda_tokensim * l_tokensim +
                            lambda_sas * l_sas
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                current_epoch_total_loss_sum += loss.item()
                current_epoch_affinity_loss_sum += l_affinity.item()
                current_epoch_tokensim_loss_sum += l_tokensim.item()
                current_epoch_sas_loss_sum += l_sas.item()


                # Calculate and accumulate cosine similarity
                with torch.no_grad():
                    sim_batch = F.cosine_similarity(
                        F.normalize(student_img_embed, dim=-1),
                        F.normalize(teacher_text_embed, dim=-1)
                    ).mean().item()
                    current_epoch_cosine_sim_sum += sim_batch


                # Update tqdm progress bar display
                pbar.set_postfix({
                    'TotalL': f'{loss.item():.4f}',
                    'AffL': f'{l_affinity.item():.4f}',
                    'TSL': f'{l_tokensim.item():.4f}',
                    'SASL': f'{l_sas.item():.4f}',
                    'Sim': f'{sim_batch:.4f}'
                })


                # every step log writes to file
                log_line_step = (f"Epoch {epoch + 1}, Step {step_idx}, "
                                 f"TotalL: {loss.item():.4f}, "
                                 f"AffL: {l_affinity.item():.4f}, "
                                 f"TSL: {l_tokensim.item():.4f}, "
                                 f"SASL: {l_sas.item():.4f}, "
                                 f"Sim: {sim_batch:.4f}, "
                                 f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                step_log_file.write(log_line_step + "\n")
                step_log_file.flush()


                # TensorBoard every step level Logging
                global_step = epoch * len(dataloader) + step_idx
                writer.add_scalar('Stage1/Loss/Step/Total', loss.item(), global_step)
                writer.add_scalar('Stage1/Loss/Step/Affinity', l_affinity.item(), global_step)
                writer.add_scalar('Stage1/Loss/Step/TokenSim', l_tokensim.item(), global_step)
                writer.add_scalar('Stage1/Loss/Step/SAS', l_sas.item(), global_step)
                writer.add_scalar('Stage1/Metrics/Step/Cosine_Similarity', sim_batch, global_step)
                writer.add_scalar('Stage1/LearningRate/Step', optimizer.param_groups[0]['lr'], global_step)


        # Epoch level summary
        avg_total_loss_epoch = current_epoch_total_loss_sum / len(dataloader)
        avg_affinity_loss_epoch = current_epoch_affinity_loss_sum / len(dataloader)
        avg_tokensim_loss_epoch = current_epoch_tokensim_loss_sum / len(dataloader)
        avg_sas_loss_epoch = current_epoch_sas_loss_sum / len(dataloader)
        avg_cosine_sim_epoch = current_epoch_cosine_sim_sum / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        time_taken_epoch = time.time() - start_time_epoch

        print(f"\n--- Epoch {epoch + 1} Summary (Stage 1) ---")
        print(f"Avg Total Loss: {avg_total_loss_epoch:.4f}")
        print(f"Avg Affinity Loss: {avg_affinity_loss_epoch:.4f}")
        print(f"Avg TokenSim Loss: {avg_tokensim_loss_epoch:.4f}")
        print(f"Avg SAS Loss: {avg_sas_loss_epoch:.4f}")
        print(f"Avg Student-Teacher Cosine Similarity: {avg_cosine_sim_epoch:.4f}")
        print(f"Current Learning Rate: {current_lr:.6f}")
        print(f"Time taken for Epoch {epoch + 1}: {time_taken_epoch:.2f}s")
        print("-" * 30)


        # Write Epoch to log file
        log_line_epoch = (f"Epoch {epoch + 1}, TotalL: {avg_total_loss_epoch:.4f}, "
                          f"AffL: {avg_affinity_loss_epoch:.4f}, TSL: {avg_tokensim_loss_epoch:.4f}, "
                          f"SASL: {avg_sas_loss_epoch:.4f}, "
                          f"Sim: {avg_cosine_sim_epoch:.4f}, LR: {current_lr:.6f}, "
                          f"Time: {time_taken_epoch:.2f}s")
        with open(log_path_epoch, "a", encoding="utf-8") as f:
            f.write(log_line_epoch + "\n")
            f.flush()


        # Record historical data for drawing
        total_loss_history.append(avg_total_loss_epoch)
        affinity_loss_history.append(avg_affinity_loss_epoch)
        tokensim_loss_history.append(avg_tokensim_loss_epoch)
        sas_loss_history.append(avg_sas_loss_epoch)
        cosine_sim_history.append(avg_cosine_sim_epoch)


        # TensorBoard Epoch level Logging
        writer.add_scalar('Stage1/Loss/Epoch/Total', avg_total_loss_epoch, epoch)
        writer.add_scalar('Stage1/Loss/Epoch/Affinity', avg_affinity_loss_epoch, epoch)
        writer.add_scalar('Stage1/Loss/Epoch/TokenSim', avg_tokensim_loss_epoch, epoch)
        writer.add_scalar('Stage1/Loss/Epoch/SAS', avg_sas_loss_epoch, epoch)
        writer.add_scalar('Stage1/Metrics/Epoch/Avg_Cosine_Similarity', avg_cosine_sim_epoch, epoch)
        writer.add_scalar('Stage1/LearningRate/Epoch', current_lr, epoch)


        # Early Stopping Logic
        if avg_cosine_sim_epoch > best_metric + min_delta_for_early_stopping:
            best_metric = avg_cosine_sim_epoch
            epochs_no_improve = 0
            torch.save(student.state_dict(), temp_best_model_path)
            print(
                f"[INFO] Metric improved to {best_metric:.4f}. Saving best model to: {temp_best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] Metric did not improve for {epochs_no_improve} epoch(s). Best was {best_metric:.4f}.")
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"[INFO] Early stopping triggered No improvement for {early_stopping_patience} consecutive epochs.")
                break


        # Save the latest checkpoint for resuming Stage 1 if interrupted
        torch.save(student.state_dict(), resume_checkpoint_path)
        print(f"[INFO] Saved latest Stage 1 checkpoint: {resume_checkpoint_path}")

        scheduler.step()

    print("\n[INFO] Stage 1 training complete.")

    if os.path.exists(temp_best_model_path):
        os.rename(temp_best_model_path, final_stage1_best_model_path)
        print(f"[INFO] Final Stage 1 best model saved: {final_stage1_best_model_path}")
    else:
        print(
            f"Temporary best model not found at '{temp_best_model_path}'. Final Stage 1 best model not created.")


    # Plotting and TensorBoard closing
    plot_loss_curve_dual(total_loss_history, cosine_sim_history,
                         save_path=os.path.join(figures_dir, "train_total_loss_and_sim_curve_stage1.png"),
                         title="Total Loss and Cosine Similarity over Epochs (Stage 1)",
                         ylabel1="Total Loss", ylabel2="Cosine Similarity")
    print("[INFO] Saved Stage 1 Loss and Cosine Similarity Curve Plot.")

    writer.close()
    print("[INFO] TensorBoard writer closed.")

if __name__ == "__main__":
    main()