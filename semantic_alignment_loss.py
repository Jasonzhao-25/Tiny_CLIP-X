import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentLoss(nn.Module):
    def __init__(self, feature_map_channels: int, embed_dim: int):
        super(SemanticAlignmentLoss, self).__init__()
        self.projection = nn.Linear(feature_map_channels, embed_dim)

    def forward(self, feature_map: torch.Tensor, student_embed: torch.Tensor) -> torch.Tensor:
        try:
            B, C, H, W = feature_map.shape
            D = student_embed.shape[1]

            assert self.projection.in_features == C and self.projection.out_features == D, \
                f"Projection layer dimensions mismatch. Expected ({self.projection.in_features}, {self.projection.out_features}), got ({C}, {D})"

            self.projection = self.projection.to(student_embed.device)

            pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).view(B, C)
            projected = self.projection(pooled)  # [B, D]
            projected = F.normalize(projected + 1e-6, dim=1)
            student_embed = F.normalize(student_embed + 1e-6, dim=1)
            sim = F.cosine_similarity(projected, student_embed, dim=1)

            if torch.isnan(sim).any() or torch.isinf(sim).any():
                print("Semantic sim contains NaN")
                return torch.tensor(0.0, device=student_embed.device, requires_grad=True)

            return 1 - sim.mean()

        except Exception as e:
            print(f"SemanticAlignmentLoss forward failed: {e}")
            return torch.tensor(0.0, device=student_embed.device, requires_grad=True)