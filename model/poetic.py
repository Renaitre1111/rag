import torch
import torch.nn as nn
import math
from model.mamba import MambaLMHeadModel

class MoleculeProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x: [Batch, EGNN_Dim] -> [Batch, 1, Model_Dim]
        return self.net(x).unsqueeze(1)

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
    def __init__(self, config, egnn_dim=128, device=None, dtype=None):
        super().__init__(config, device=device, dtype=dtype)

        self.mol_projector = MoleculeProjector(input_dim=egnn_dim, output_dim=config.d_model).to(device)
        self.prop_projector = PropertyProjector(hidden_dim=config.d_model).to(device)
        self.delta_projector = PropertyProjector(hidden_dim=config.d_model).to(device)
    
    def forward(self, ref_egnn_vecs, target_input_ids, ref_prop, target_prop):
        ref_emb = self.mol_projector(ref_egnn_vecs)
        target_prop_emb = self.prop_projector(target_prop)

        delta = target_prop - ref_prop
        delta_emb = self.delta_projector(delta)

        target_token_emb = self.backbone.embedding(target_input_ids)

        inputs_embeds = torch.cat([ref_emb, target_prop_emb, delta_emb, target_token_emb], dim=1)
        hidden_states = self.backbone(inputs_embeds=inputs_embeds)

        prop_hidden = hidden_states[:, 1, :] 
        delta_hidden = hidden_states[:, 2, :]

        pred_prop = self.prop_projector.reg_head(prop_hidden)
        pred_delta = self.delta_projector.reg_head(delta_hidden)

        loss_aux = torch.nn.functional.mse_loss(pred_prop, target_prop) + \
                   torch.nn.functional.mse_loss(pred_delta, delta)
        
        prefix_len = 3
        predictive_hidden = hidden_states[:, prefix_len-1 : -1, :]

        logits = self.lm_head(predictive_hidden) # [B, L, Vocab]

        loss_gen = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            target_input_ids.reshape(-1), 
            ignore_index=self.padding_token_id
        )
        
        total_loss = loss_gen + 0.1 * loss_aux

        return logits, total_loss