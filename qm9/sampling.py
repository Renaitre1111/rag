import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from qm9.analyze import check_stability
import qm9.utils as qm9utils
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel

def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None, rag_aggregator=None, rag_db=None):
    flow = flow.module if isinstance(flow, (DDP, DataParallel)) else flow

    n_samples = 1
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device) # (n_samples, n_nodes, context_node_nf)
        #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    adaln_ctx = None
    if args.use_rag:
        if context is not None:
            target_norm = context[:, :1, :] # (bs, num_prop)
        else: # Fallback if no context conditioning at all, sample new target values for RAG
            target_norm = prop_dist.sample(n_nodes).to(device)

        target_unnorm = qm9utils.unnormalize_context(target_norm, args.conditioning, prop_dist.normalizer)
        target_unnorm_ret = target_unnorm[:, 0, :].cpu().numpy() # (bs, num_prop)
        target_norm = target_norm.squeeze(1) # (bs, num_prop)
        ret_mol_embs, ret_prop_vals, ret_dist_to_targs, _ = rag_db.retrieve_batch(target_unnorm_ret, k=args.rag_k)

        ret_mol_embs = torch.from_numpy(ret_mol_embs).to(device, dtype=torch.float32)
        ret_prop_vals = torch.from_numpy(ret_prop_vals).to(device, dtype=torch.float32)
        ret_dist_to_targs = torch.from_numpy(ret_dist_to_targs).to(device, dtype=torch.float32)

        adaln_ctx = rag_aggregator(target_norm, ret_mol_embs, ret_prop_vals, ret_dist_to_targs) # (1, agg_dim)
        
        adaln_ctx = adaln_ctx.unsqueeze(1).repeat(1, n_nodes, 1) * node_mask # (bs, max_n_nodes, agg_dim)


    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False, rag_aggregator=None, rag_db=None):
    generative_model = generative_model.module if isinstance(generative_model, (DDP, DataParallel)) else generative_model

    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask # (bs, max_n_nodes, context_node_nf)
    else:
        context = None
    
    adaln_ctx = None
    if args.use_rag:
        if context is not None:
            target_norm = context[:, :1, :] # (bs, 1, num_prop)
        else: # Fallback if no context conditioning at all, sample new target values for RAG
            target_norm = prop_dist.sample_batch(nodesxsample).to(device)
            
        target_unnorm = qm9utils.unnormalize_context(target_norm, args.conditioning, prop_dist.normalizer)
        target_unnorm_ret = target_unnorm[:, 0, :].cpu().numpy() # (bs, num_prop)
        target_norm = target_norm.squeeze(1) # (bs, num_prop)
        ret_mol_embs, ret_prop_vals, ret_dist_to_targs, _ = rag_db.retrieve_batch(target_unnorm_ret, k=args.rag_k)

        ret_mol_embs = torch.from_numpy(ret_mol_embs).to(device, dtype=torch.float32)
        ret_prop_vals = torch.from_numpy(ret_prop_vals).to(device, dtype=torch.float32)
        ret_dist_to_targs = torch.from_numpy(ret_dist_to_targs).to(device, dtype=torch.float32)

        adaln_ctx = rag_aggregator(target_norm, ret_mol_embs, ret_prop_vals, ret_dist_to_targs) # (1, agg_dim)
        
        adaln_ctx = adaln_ctx.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask # (bs, max_n_nodes, agg_dim)
        

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise, adaln_ctx=adaln_ctx)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask


def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100, rag_aggregator=None, rag_db=None):
    generative_model = generative_model.module if isinstance(generative_model, (DDP, DataParallel)) else generative_model
    
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True, rag_aggregator=rag_aggregator, rag_db=rag_db)
    return one_hot, charges, x, node_mask
