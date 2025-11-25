import torch
import torch.nn as nn
import math
from model.mamba import MambaLMHeadModel

class PropertyProjector(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64)))

        self.mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x [batch_size, 1]
        sinusoid_inp = torch.ger(x.view(-1), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return self.mlp(pos_emb).unsqueeze(1) # [B, 1, H]
    
class PoeticMamba(MambaLMHeadModel):
    def __init__(self, config, device=None, dtype=None):
        super().__init__(config, device=device, dtype=dtype)
        self.prop_projector = PropertyProjector(config.d_model).to(device)
    
    def forward(self, ref_input_ids, target_input_ids, ref_prop, target_prop):
        ref_emb = self.backbone.embedding(ref_input_ids)
        target_emb = self.backbone.embedding(target_input_ids)

        delta = target_prop - ref_prop
        delta_emb = self.prop_projector(delta)       # [B, 1, H]
        target_prop_emb = self.prop_projector(target_prop) # [B, 1, H]

        inputs_embeds = torch.cat([ref_emb, delta_emb, target_prop_emb, target_emb], dim=1)
        hidden_states = self.backbone(inputs_embeds=inputs_embeds)

        ref_len = ref_input_ids.size(1)
        delta_hidden = hidden_states[:, ref_len, :]
        prop_hidden = hidden_states[:, ref_len + 1, :]

        pred_delta = self.prop_projector.reg_head(delta_hidden)
        pred_prop = self.prop_projector.reg_head(prop_hidden)

        loss_aux = torch.nn.functional.mse_loss(pred_delta, delta) + torch.nn.functional.mse_loss(pred_prop, target_prop)

        predictive_hidden = hidden_states[:, ref_len + 1 : -1, :]
        logits = self.lm_head(predictive_hidden)
        loss_gen = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            target_input_ids.reshape(-1), 
            ignore_index=self.padding_token_id
        )
        
        total_loss = loss_gen + 0.1 * loss_aux

        return logits, total_loss