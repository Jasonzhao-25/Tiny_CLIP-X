import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.tinyclip_student import TinyCLIPStudent
from models.clip_teacher import CLIPTeacher
from scripts.paths import Config
from scripts.seed_everything import set_seed


def affinity_loss(student_embed, teacher_embed):
    student_embed = F.normalize(student_embed, dim=-1)
    teacher_embed = F.normalize(teacher_embed, dim=-1)
    return 1 - F.cosine_similarity(student_embed, teacher_embed, dim=-1).mean()


def main():
    cfg = Config('../configs/config_stage2b.yaml')
    set_seed(cfg.get_seed())
    device = torch.device(cfg.get_device())
    print(f"[INFO] Using device: {device}")


    # Parameter settings
    num_epochs = 20
    lr = 0.0001
    batch_size = cfg.get_hyperparams()["batch_size"]
    s2b_checkpoint_path = "D:/Project/Tiny CLIP/CODE/train/outputs/checkpoints/tinyclip_student_stage2b_best.pt"
    final_s3_model_path = "D:/Project/Tiny CLIP/CODE/train/outputs/checkpoints/tinyclip_student_stage3_final.pt"
    print(f"[INFO] Initializing Stage 3 Training")


    # Load dataset
    from datasets.coco import COCOTrainDataset
    train_dataset = COCOTrainDataset(image_size=cfg.get_model_params()["image_size"])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Load model
    teacher = CLIPTeacher().to(device).eval()
    model_params = cfg.get_model_params()
    student_init_params = {'embed_dim': model_params['embed_dim'], 'num_classes': model_params['num_classes'],
                           'freeze_layers': 0}
    student = TinyCLIPStudent(**student_init_params).to(device)

    print(f"[INFO] Loading best model weights from Stage 2b: {s2b_checkpoint_path}")
    student.load_state_dict(torch.load(s2b_checkpoint_path, map_location=device), strict=False)
    print("[INFO] Stage 2b weights loaded success.")


    # Freeze the backbone network
    print("[INFO] Freezing backbone.")
    for name, param in student.named_parameters():
        if 'projector' in name:
            param.requires_grad = True
            print(f"  > Training parameter: {name}")
        else:
            param.requires_grad = False


    # Optimizer and training loop
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    best_sim = -float('inf')

    for epoch in range(num_epochs):
        student.train()
        total_loss = 0
        total_sim = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Stage 3)")

        for images, texts_list, labels, img_ids in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            with autocast():
                student_img_embed = student(images)
                with torch.no_grad():
                    teacher_text_embed = teacher.forward_text(texts_list)

                loss = affinity_loss(student_img_embed, teacher_text_embed)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                sim_batch = F.cosine_similarity(student_img_embed, teacher_text_embed).mean().item()

            total_loss += loss.item()
            total_sim += sim_batch
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Sim': f'{sim_batch:.4f}'})

        avg_loss = total_loss / len(dataloader)
        avg_sim = total_sim / len(dataloader)
        print(f" Epoch {epoch + 1} Summary (Stage 3)")
        print(f"Avg Loss: {avg_loss:.4f}, Avg Sim: {avg_sim:.4f}")

        if avg_sim > best_sim:
            best_sim = avg_sim
            torch.save(student.state_dict(), final_s3_model_path)
            print(f"[INFO] Best similarity improved to {best_sim:.4f}. Saving final model to {final_s3_model_path}")

        scheduler.step()

    print("\n[INFO] Stage 3 training complete.")
    print(f"Final model saved to: {final_s3_model_path}")


if __name__ == "__main__":
    main()