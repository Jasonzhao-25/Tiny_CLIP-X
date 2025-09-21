import open_clip
import torch
import torch.nn as nn


class CLIPTeacher(nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.visual = self.model.visual

    def forward(self, images: torch.Tensor):
        img_embed = self.model.encode_image(images)
        return img_embed

    def forward_image(self, images: torch.Tensor):
        return self.model.encode_image(images)

    def forward_text(self, texts: list):
        device = next(self.parameters()).device
        if isinstance(texts[0], list):
            all_text_embeds_for_batch = []
            for cap_list_for_one_image in texts:
                tokens = self.tokenizer(cap_list_for_one_image).to(device)
                embeds = self.model.encode_text(tokens)
                all_text_embeds_for_batch.append(embeds.mean(dim=0))
            return torch.stack(all_text_embeds_for_batch)

        else:
            tokens = self.tokenizer(texts).to(device)
            embeds = self.model.encode_text(tokens)
            return embeds