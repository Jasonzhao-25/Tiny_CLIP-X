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

def auxiliary_loss_fn(aux_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid_labels_mask = (labels != -1)
    if valid_labels_mask.sum() > 0:
        return F.cross_entropy(aux_logits[valid_labels_mask], labels[valid_labels_mask])
    else:
        return torch.tensor(0.0).to(aux_logits.device)

def main():
    cfg = Config('D:/Project/Tiny CLIP/CODE/configs/config_stage2.yaml')
    set_seed(cfg.get_seed())
    device = torch.device(cfg.get_device())
    print(f"[INFO] Using device: {device}")


    # History lists for plotting
    total_loss_history = []
    affinity_loss_history = []
    tokensim_loss_history = []
    sas_loss_history = []
    aux_loss_history = []
    cosine_sim_history = []


    # Get parameters from the Stage 2 config
    hyperparams = cfg.get_hyperparams()
    model_params = cfg.get_model_params()
    training_flags = cfg.get_training_flags()
    batch_size = hyperparams["batch_size"]
    num_epochs = hyperparams["num_epochs"]
    lr = hyperparams["learning_rate"]
    lambda_affinity = hyperparams["lambda_affinity"]
    lambda_tokensim = hyperparams["lambda_tokensim"]
    lambda_sas = hyperparams["lambda_sas"]
    lambda_aux = hyperparams["lambda_aux"]
    image_size = model_params["image_size"]
    stage1_checkpoint_path = training_flags["load_from_stage1_checkpoint"]

    if not os.path.exists(stage1_checkpoint_path):
        print(f" Stage 1 checkpoint not found: {stage1_checkpoint_path}")
        print("  Ensure the path is correct and the file exists.")
        exit()

    checkpoints_dir = cfg.get_output_path("checkpoints")
    logs_dir = cfg.get_output_path("logs")
    figures_dir = cfg.get_output_path("figures")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    log_path_epoch = os.path.join(logs_dir, "train_log_epoch_stage2.txt")
    log_path_step = os.path.join(logs_dir, "train_tokensim_log_steps_stage2.txt")

    with open(log_path_epoch, "w", encoding="utf-8") as f:
        f.write("")
    with open(log_path_step, "w", encoding="utf-8") as f:
        f.write("")

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_log_dir = os.path.join(logs_dir, "runs", "train_stage2", current_time)
    writer = SummaryWriter(tb_log_dir)
    print(f"[INFO] TensorBoard logs will be saved to: {tb_log_dir}")

    resume_checkpoint_path = os.path.join(checkpoints_dir, "last_TinyCLIP_student_stage2.pt")
    final_stage2_best_model_path = os.path.join(checkpoints_dir, "TinyCLIP_student_stage2_best.pt")
    start_epoch = 0


    # Dataset and Loader
    train_dataset = COCOTrainDataset(image_size=image_size)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f"[INFO] Dataset loaded. Number of training images: {len(train_dataset)}")


    # Models
    teacher = CLIPTeacher().to(device).eval()
    student = TinyCLIPStudent(
        embed_dim=model_params["embed_dim"],
        num_classes=model_params["num_classes"],
        freeze_layers=model_params["freeze_layers"]
    ).to(device)

    print(f"[INFO] Initializing Stage 2 Training")
    print(f"[INFO] Loading best model weights from Stage 1: {stage1_checkpoint_path}")
    student.load_state_dict(torch.load(stage1_checkpoint_path, map_location=device), strict=False)
    print("[INFO] Stage 1 weights loaded success.")

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=hyperparams["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    semantic_loss_fn = SemanticAlignmentLoss(128, model_params["embed_dim"]).to(device)
    scaler = GradScaler()
    print("[INFO] Starting Stage 2 training")

    best_metric = -float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 7
    min_delta_for_early_stopping = 0.0001

    for epoch in range(start_epoch, num_epochs):
        student.train()


        # Reset accumulators for the epoch
        current_epoch_total_loss_sum = 0.0
        current_epoch_affinity_loss_sum = 0.0
        current_epoch_tokensim_loss_sum = 0.0
        current_epoch_sas_loss_sum = 0.0
        current_epoch_aux_loss_sum = 0.0
        current_epoch_cosine_sim_sum = 0.0
        start_time_epoch = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Stage 2)")

        for step_idx, (images, texts_list, labels, img_ids) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast():
                student_img_embed, student_feature_map, student_logits, student_aux_logits = student(
                    images, return_features=True, return_logits=True, return_aux_logits=True
                )

                with torch.no_grad():
                    teacher_text_embed = teacher.forward_text(texts_list)

                l_affinity = affinity_loss(student_img_embed, teacher_text_embed)
                l_tokensim = tokensim_loss_fn(student_logits, labels)
                l_sas = semantic_loss_fn(student_feature_map, student_img_embed)
                l_aux = auxiliary_loss_fn(student_aux_logits, labels)
                loss = (
                        lambda_affinity * l_affinity +
                        lambda_tokensim * l_tokensim +
                        lambda_sas * l_sas +
                        lambda_aux * l_aux
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # Accumulate losses
            current_epoch_total_loss_sum += loss.item()
            current_epoch_affinity_loss_sum += l_affinity.item()
            current_epoch_tokensim_loss_sum += l_tokensim.item()
            current_epoch_sas_loss_sum += l_sas.item()
            current_epoch_aux_loss_sum += l_aux.item()
            with torch.no_grad():
                sim_batch = F.cosine_similarity(
                    F.normalize(student_img_embed, dim=-1),
                    F.normalize(teacher_text_embed, dim=-1)
                ).mean().item()
                current_epoch_cosine_sim_sum += sim_batch


            # Update tqdm progress bar with all losses
            pbar.set_postfix({
                'TotalL': f'{loss.item():.4f}',
                'AffL': f'{l_affinity.item():.4f}',
                'TSL': f'{l_tokensim.item():.4f}',
                'SASL': f'{l_sas.item():.4f}',
                'AuxL': f'{l_aux.item():.4f}',
                'Sim': f'{sim_batch:.4f}'
            })

        avg_total_loss_epoch = current_epoch_total_loss_sum / len(dataloader)
        avg_affinity_loss_epoch = current_epoch_affinity_loss_sum / len(dataloader)
        avg_tokensim_loss_epoch = current_epoch_tokensim_loss_sum / len(dataloader)
        avg_sas_loss_epoch = current_epoch_sas_loss_sum / len(dataloader)
        avg_aux_loss_epoch = current_epoch_aux_loss_sum / len(dataloader)
        avg_cosine_sim_epoch = current_epoch_cosine_sim_sum / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        time_taken_epoch = time.time() - start_time_epoch


        print(f"\n--- Epoch {epoch + 1} Summary (Stage 2) ---")
        print(f"Avg Total Loss: {avg_total_loss_epoch:.4f}")
        print(f"Avg Affinity Loss: {avg_affinity_loss_epoch:.4f}, Avg TokenSim Loss: {avg_tokensim_loss_epoch:.4f}")
        print(f"Avg SAS Loss: {avg_sas_loss_epoch:.4f},      Avg Auxiliary Loss: {avg_aux_loss_epoch:.4f}")
        print(f"Avg Student-Teacher Cosine Similarity: {avg_cosine_sim_epoch:.4f}")
        print(f"Current Learning Rate: {current_lr:.6f}")
        print(f"Time taken for Epoch {epoch + 1}: {time_taken_epoch:.2f}s\n" + "-" * 30)

        log_line_epoch = (f"Epoch {epoch + 1}, TotalL: {avg_total_loss_epoch:.4f}, "
                          f"AffL: {avg_affinity_loss_epoch:.4f}, TSL: {avg_tokensim_loss_epoch:.4f}, "
                          f"SASL: {avg_sas_loss_epoch:.4f}, AuxL: {avg_aux_loss_epoch:.4f}, "
                          f"Sim: {avg_cosine_sim_epoch:.4f}, LR: {current_lr:.6f}, "
                          f"Time: {time_taken_epoch:.2f}s")
        with open(log_path_epoch, "a", encoding="utf-8") as f:
            f.write(log_line_epoch + "\n")


        # Append to history lists for plotting
        aux_loss_history.append(avg_aux_loss_epoch)
        writer.add_scalar('Stage2/Loss/Epoch/Total', avg_total_loss_epoch, epoch)
        writer.add_scalar('Stage2/Loss/Epoch/Affinity', avg_affinity_loss_epoch, epoch)
        writer.add_scalar('Stage2/Loss/Epoch/TokenSim', avg_tokensim_loss_epoch, epoch)
        writer.add_scalar('Stage2/Loss/Epoch/SAS', avg_sas_loss_epoch, epoch)
        writer.add_scalar('Stage2/Loss/Epoch/Auxiliary', avg_aux_loss_epoch, epoch)
        writer.add_scalar('Stage2/Metrics/Epoch/Avg_Cosine_Similarity', avg_cosine_sim_epoch, epoch)
        writer.add_scalar('Stage2/LearningRate/Epoch', current_lr, epoch)


        # Early Stopping Logic
        if avg_cosine_sim_epoch > best_metric + min_delta_for_early_stopping:
            best_metric = avg_cosine_sim_epoch
            epochs_no_improve = 0
            torch.save(student.state_dict(), final_stage2_best_model_path)
            print(
                f"[INFO] Metric improved to {best_metric:.4f}. Saving best Stage 2 model to: {final_stage2_best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] Metric not improve for {epochs_no_improve} epoch(s). Best was {best_metric:.4f}.")
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"[INFO] Early stopping triggered No improvement for {early_stopping_patience} consecutive epochs.")
                break

        torch.save(student.state_dict(), resume_checkpoint_path)
        scheduler.step()

    print("\n[INFO] Stage 2 training complete.")

    plot_loss_curve_dual(total_loss_history, cosine_sim_history,
                         save_path=os.path.join(figures_dir, "train_total_loss_and_sim_curve_stage2.png"),
                         title="Total Loss and Cosine Similarity over Epochs (Stage 2)",
                         ylabel1="Total Loss", ylabel2="Cosine Similarity")
    print("[INFO] Saved Stage 2 Loss and Cosine Similarity Curve Plot.")

    writer.close()
    print("[INFO] TensorBoard writer closed.")


if __name__ == "__main__":
    main()