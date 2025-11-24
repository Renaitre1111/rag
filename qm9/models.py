import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

import numpy as np
from egnn.models import EGNN_dynamics_QM9, EGNN_encoder_QM9, EGNN_decoder_QM9

from equivariant_diffusion.en_diffusion import EnVariationalDiffusion, EnHierarchicalVAE, EnLatentDiffusion
from equivariant_diffusion import utils as diffusion_utils

import pickle
from os.path import join

class RAGAggregator(nn.Module):
    def __init__(self, mol_emb_dim, num_prop, agg_dim):
        super().__init__()
        self.mol_emb_dim = mol_emb_dim
        self.num_prop = num_prop
        self.agg_dim = agg_dim

        self.Q_mlp = nn.Sequential(
            nn.Linear(num_prop, agg_dim),
            nn.SiLU()
        )

        self.K_mlp = nn.Sequential(
            nn.Linear(mol_emb_dim + num_prop * 2, agg_dim),
            nn.SiLU()
        )

        self.V_mlp = nn.Sequential(
            nn.Linear(mol_emb_dim + num_prop * 2, agg_dim),
            nn.SiLU()
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, target_val, ret_mol_embs, ret_prop_vals, ret_dist_to_targs, k_mask=None):

        Q = self.Q_mlp(target_val).unsqueeze(1) # (bs, 1, agg_dim)
        K_V_input = torch.cat([ret_mol_embs, ret_prop_vals, ret_dist_to_targs], dim=-1)
        K = self.K_mlp(K_V_input)  # (bs, k, agg_dim)
        V = self.V_mlp(K_V_input)  # (bs, k, agg_dim)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)).squeeze(1)  # (bs, k)

        if k_mask is not None:
            attn_scores = attn_scores.masked_fill(~k_mask, -torch.inf)
            
        attn_weights = self.softmax(attn_scores).unsqueeze(1)  # (bs, 1, k)

        rag_emb = torch.bmm(attn_weights, V).squeeze(1)  # (bs, agg_dim)

        return rag_emb
    
def _create_rag_aggregator(args):
    if hasattr(args, 'use_rag') and args.use_rag:
        if not hasattr(args, 'rag_mol_emb_dim') or \
           not hasattr(args, 'rag_num_prop') or \
           not hasattr(args, 'rag_agg_dim'):
            raise ValueError("RAG is enabled but missing 'rag_mol_emb_dim', 'rag_num_prop', or 'rag_agg_dim' in args.")
        return RAGAggregator(args.rag_mol_emb_dim, args.rag_num_prop, args.rag_agg_dim)
    return None

def get_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    rag_c_nf = args.rag_agg_dim if hasattr(args, 'use_rag') and args.use_rag else 0

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        rag_c_nf=rag_c_nf)

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )

        rag_aggregator = _create_rag_aggregator(args)
        return vdm, nodes_dist, prop_dist, rag_aggregator

    else:
        raise ValueError(args.probabilistic_model)


def get_autoencoder(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    # if args.condition_time:
    #     dynamics_in_node_nf = in_node_nf + 1
    # else:
    print('Autoencoder models are _not_ conditioned on time.')
        # dynamics_in_node_nf = in_node_nf
    
    rag_c_nf = args.rag_agg_dim if hasattr(args, 'use_rag') and args.use_rag else 0
    encoder = EGNN_encoder_QM9(
        in_node_nf=in_node_nf, context_node_nf=args.context_node_nf, out_node_nf=args.latent_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=1,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        include_charges=args.include_charges, rag_c_nf=rag_c_nf
        )
    
    decoder = EGNN_decoder_QM9(
        in_node_nf=args.latent_nf, context_node_nf=args.context_node_nf, out_node_nf=in_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        include_charges=args.include_charges, rag_c_nf=rag_c_nf
        )

    vae = EnHierarchicalVAE(
        encoder=encoder,
        decoder=decoder,
        in_node_nf=in_node_nf,
        n_dims=3,
        latent_node_nf=args.latent_nf,
        kl_weight=args.kl_weight,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges
        )

    rag_aggregator = _create_rag_aggregator(args)
    return vae, nodes_dist, prop_dist, rag_aggregator


def get_latent_diffusion(args, device, dataset_info, dataloader_train):

    # Create (and load) the first stage model (Autoencoder).
    if args.ae_path is not None:
        with open(join(args.ae_path, 'args.pickle'), 'rb') as f:
            first_stage_args = pickle.load(f)
    else:
        first_stage_args = args
    
    # CAREFUL with this -->
    if not hasattr(first_stage_args, 'normalization_factor'):
        first_stage_args.normalization_factor = 1
    if not hasattr(first_stage_args, 'aggregation_method'):
        first_stage_args.aggregation_method = 'sum'
    if hasattr(args, 'use_rag'):
        first_stage_args.use_rag = args.use_rag
    if hasattr(args, 'rag_agg_dim'):
        first_stage_args.rag_agg_dim = args.rag_agg_dim
    if hasattr(args, 'rag_mol_emb_dim'):
        first_stage_args.rag_mol_emb_dim = args.rag_mol_emb_dim
    if hasattr(args, 'rag_num_prop'):
        first_stage_args.rag_num_prop = args.rag_num_prop
    if not hasattr(first_stage_args, 'no_cuda'):
        first_stage_args.no_cuda = args.no_cuda

    first_stage_model, nodes_dist, prop_dist, rag_aggregator = get_autoencoder(
        first_stage_args, device, dataset_info, dataloader_train)
    first_stage_model.to(device)

    if args.ae_path is not None:
        fn = 'generative_model_ema.npy' if first_stage_args.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(args.ae_path, fn),
                                        map_location=device)
        first_stage_model.load_state_dict(flow_state_dict)

    # Create the second stage model (Latent Diffusions).
    args.latent_nf = first_stage_args.latent_nf
    in_node_nf = args.latent_nf

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    rag_c_nf = args.rag_agg_dim if hasattr(args, 'use_rag') and args.use_rag else 0
    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        rag_c_nf=rag_c_nf)

    if args.probabilistic_model == 'diffusion':
        vdm = EnLatentDiffusion(
            vae=first_stage_model,
            trainable_ae=args.trainable_ae,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )

        return vdm, nodes_dist, prop_dist, rag_aggregator

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
