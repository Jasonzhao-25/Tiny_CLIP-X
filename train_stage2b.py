import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from datasets.coco import COCOTrainDataset
from models.tinyclip_student import TinyCLIPStudent
from models.clip_teacher import CLIPTeacher
from scripts.paths import Config
from scripts.seed_everything import set_seed
from scripts.plots import plot_loss_curve_dual
from losses.semantic_alignment_loss import SemanticAlignmentLoss


def affinity_loss(student_embed: torch.Tensor, teacher_embed: torch.Tensor) -> torch.Tensor:
    student_embed = F.normalize(student_embed, dim=-1)
    teacher_embed = F.normalize(teacher_embed, dim=-1)
    return 1 - F.cosine_similarity(student_embed, teacher_embed, dim=-1).mean()

def tokensim_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid_labels_mask = (labels != -1)
    if valid_labels_mask.sum() > 0:
        return F.cross_entropy(logits[valid_labels_mask], labels[valid_labels_mask])
    return torch.tensor(0.0).to(logits.device)

def auxiliary_loss_fn(aux_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid_labels_mask = (labels != -1)
    if valid_labels_mask.sum() > 0:
        return F.cross_entropy(aux_logits[valid_labels_mask], labels[valid_labels_mask])
    return torch.tensor(0.0).to(aux_logits.device)


def main():
    cfg = Config('../configs/config_stage2b.yaml')
    set_seed(cfg.get_seed())
    device = torch.device(cfg.get_device())
    print(f"[INFO] Using device: {device}")

    total_loss_history, affinity_loss_history, tokensim_loss_history = [], [], []
    sas_loss_history, aux_loss_history, cosine_sim_history = [], [], []
    hyperparams = cfg.get_hyperparams()
    model_params = cfg.get_model_params()
    training_flags = cfg.get_training_flags()

    batch_size, num_epochs, lr = hyperparams["batch_size"], hyperparams["num_epochs"], hyperparams["learning_rate"]
    lambda_affinity, lambda_tokensim, lambda_sas, lambda_aux = hyperparams["lambda_affinity"], hyperparams[
        "lambda_tokensim"], hyperparams["lambda_sas"], hyperparams["lambda_aux"]
    stage1_checkpoint_path = training_flags["load_from_stage1_checkpoint"]
    checkpoints_dir = cfg.get_output_path("checkpoints")
    logs_dir = cfg.get_output_path("logs")
    figures_dir = cfg.get_output_path("figures")

    log_path_epoch = os.path.join(logs_dir, "train_log_epoch_stage2b.txt")
    log_path_step = os.path.join(logs_dir, "train_log_steps_stage2b.txt")

    with open(log_path_epoch, "w", encoding="utf-8") as f:
        f.write("")
    with open(log_path_step, "w", encoding="utf-8") as f:
        f.write("")

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_log_dir = os.path.join(logs_dir, "runs", "train_stage2b", current_time)
    writer = SummaryWriter(tb_log_dir)
    print(f"[INFO] TensorBoard logs will be saved to: {tb_log_dir}")

    resume_checkpoint_path = os.path.join(checkpoints_dir, "last_TinyCLIP_student_stage2b.pt")
    final_best_model_path = os.path.join(checkpoints_dir, "TinyCLIP_student_stage2b_best.pt")
    start_epoch = 0


    # Dataset and model loading
    train_dataset = COCOTrainDataset(image_size=model_params["image_size"])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    teacher = CLIPTeacher().to(device).eval()
    student_init_params = {'embed_dim': model_params['embed_dim'], 'num_classes': model_params['num_classes'],
                           'freeze_layers': 0}
    student = TinyCLIPStudent(**student_init_params).to(device)

    print(f"[INFO] Initializing Stage 2b Training")
    print(f"[INFO] Loading best model weights from Stage 1: {stage1_checkpoint_path}")
    student.load_state_dict(torch.load(stage1_checkpoint_path, map_location=device), strict=False)
    print("[INFO] Stage 1 weights loaded success.")


    # Optimizer, Scheduler, Loss Function
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=hyperparams["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    semantic_loss_fn = SemanticAlignmentLoss(128, model_params["embed_dim"]).to(device)
    scaler = GradScaler()
    best_metric = -float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 7
    min_delta = 0.0001

    print("[INFO] Starting Stage 2b training")

    # training loop
    for epoch in range(start_epoch, num_epochs):
        student.train()
        epoch_total_loss, epoch_aff_loss, epoch_tsl_loss, epoch_sas_loss, epoch_aux_loss, epoch_sim = 0, 0, 0, 0, 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Stage 2b)")

        with open(log_path_step, "a", encoding="utf-8") as step_log_file:
            for step_idx, (images, texts_list, labels, img_ids) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
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
                    loss = (lambda_affinity * l_affinity + lambda_tokensim * l_tokensim +
                            lambda_sas * l_sas + lambda_aux * l_aux)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    sim_batch = F.cosine_similarity(student_img_embed, teacher_text_embed).mean().item()

                epoch_total_loss += loss.item()
                epoch_aff_loss += l_affinity.item()
                epoch_tsl_loss += l_tokensim.item()
                epoch_sas_loss += l_sas.item()
                epoch_aux_loss += l_aux.item()
                epoch_sim += sim_batch

                pbar.set_postfix({
                    'TotalL': f'{loss.item():.4f}', 'AffL': f'{l_affinity.item():.4f}',
                    'TSL': f'{l_tokensim.item():.4f}', 'SASL': f'{l_sas.item():.4f}',
                    'AuxL': f'{l_aux.item():.4f}', 'Sim': f'{sim_batch:.4f}'
                })

                global_step = epoch * len(dataloader) + step_idx
                writer.add_scalar('Stage2b/Loss/Step/Total', loss.item(), global_step)

        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_aff_loss = epoch_aff_loss / len(dataloader)
        avg_tsl_loss = epoch_tsl_loss / len(dataloader)
        avg_sas_loss = epoch_sas_loss / len(dataloader)
        avg_aux_loss = epoch_aux_loss / len(dataloader)
        avg_sim = epoch_sim / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        time_taken = time.time() - pbar.start_t

        print(f"\n Epoch {epoch + 1} Summary (Stage 2b)")
        print(f"Avg Total Loss: {avg_total_loss:.4f}")
        print(f"Avg Affinity Loss: {avg_aff_loss:.4f}, Avg TokenSim Loss: {avg_tsl_loss:.4f}")
        print(f"Avg SAS Loss: {avg_sas_loss:.4f},      Avg Auxiliary Loss: {avg_aux_loss:.4f}")
        print(f"Avg Student-Teacher Cosine Similarity: {avg_sim:.4f}")
        print(f"Current Learning Rate: {current_lr:.6f}")
        print(f"Time taken for Epoch {epoch + 1}: {time_taken:.2f}s\n" + "-" * 30)

        log_line_epoch = (f"Epoch {epoch + 1}, TotalL: {avg_total_loss:.4f}, AffL: {avg_aff_loss:.4f}, "
                          f"TSL: {avg_tsl_loss:.4f}, SASL: {avg_sas_loss:.4f}, AuxL: {avg_aux_loss:.4f}, "
                          f"Sim: {avg_sim:.4f}, LR: {current_lr:.6f}, Time: {time_taken:.2f}s")
        with open(log_path_epoch, "a", encoding="utf-8") as f:
            f.write(log_line_epoch + "\n")

        total_loss_history.append(avg_total_loss);
        cosine_sim_history.append(avg_sim)

        writer.add_scalar('Stage2b/Loss/Epoch/Total', avg_total_loss, epoch)
        writer.add_scalar('Stage2b/Loss/Epoch/Affinity', avg_aff_loss, epoch)
        writer.add_scalar('Stage2b/Loss/Epoch/TokenSim', avg_tsl_loss, epoch)
        writer.add_scalar('Stage2b/Loss/Epoch/SAS', avg_sas_loss, epoch)
        writer.add_scalar('Stage2b/Loss/Epoch/Auxiliary', avg_aux_loss, epoch)
        writer.add_scalar('Stage2b/Metrics/Epoch/Avg_Cosine_Similarity', avg_sim, epoch)
        writer.add_scalar('Stage2b/LearningRate/Epoch', current_lr, epoch)


        # Early stopping and model saving
        if avg_sim > best_metric + min_delta:
            best_metric = avg_sim
            epochs_no_improve = 0
            torch.save(student.state_dict(), final_best_model_path)
            print(
                f"[INFO] Metric improved to {best_metric:.4f}. Saving best Stage 2b model to: {final_best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] Metric did not improve for {epochs_no_improve} epoch(s). Best was {best_metric:.4f}.")
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"[INFO] Early stopping triggered No improvement for {early_stopping_patience} consecutive epochs.")
                break

        torch.save(student.state_dict(), resume_checkpoint_path)
        scheduler.step()

    print("\n[INFO] Stage 2b training complete.")

    plot_loss_curve_dual(total_loss_history, cosine_sim_history,
                         save_path=os.path.join(figures_dir, "train_loss_sim_curve_stage2b.png"),
                         title="Total Loss and Cosine Similarity over Epochs (Stage 2b)",
                         ylabel1="Total Loss", ylabel2="Cosine Similarity")
    print("[INFO] Saved Stage 2b Loss and Cosine Similarity Curve Plot.")

    writer.close()
    print("[INFO] TensorBoard writer closed.")


if __name__ == "__main__":
    main()