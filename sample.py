import argparse
import torch
import numpy as np
from torch.distributions.categorical import Categorical
import os

class PropertyDistribution:
    def __init__(self, values, num_bins=1000):
        self.num_bins = num_bins
        values = torch.tensor([v for v in values if not (np.isnan(v) or np.isinf(v))], dtype=torch.float)

        self.probs, self.params = self._create_prob_dist(values)
        self.m = Categorical(self.probs)

    def _create_prob_dist(self, values):
        n_bins = self.num_bins
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12

        histogram = torch.zeros(n_bins)

        for val in values:
            # Calculate bin index
            i = int((val - prop_min) / prop_range * n_bins)
            
            # Boundary check
            if i == n_bins:
                i = n_bins - 1
            
            histogram[i] += 1

        probs = histogram / torch.sum(histogram)

        params = [prop_min, prop_max]

        return probs, params
    
    def _idx2value(self, idx, params, n_bins):
        prop_min = params[0]
        prop_max = params[1]
        prop_range = prop_max - prop_min

        left = float(idx) / n_bins * prop_range + prop_min
        right = float(idx + 1) / n_bins * prop_range + prop_min

        val = torch.rand(1) * (right - left) + left
        return val
    
    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        vals = []
        for i in idx:
            val = self._idx2value(i, self.params, self.num_bins)
            vals.append(val)
            
        return torch.tensor(vals)
    
def main():
    parser = argparse.ArgumentParser(description="Sample properties from training distribution.")
    parser.add_argument('--prop_path', type=str, required=True, help="Path to training set properties (.txt)")
    parser.add_argument('--save_path', type=str, default='data/sampled_gap.txt', help="Output path")
    parser.add_argument('--num_samples', type=int, default=10000, help="How many samples to generate")
    parser.add_argument('--num_bins', type=int, default=1000, help="Number of histogram bins")
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading properties from {args.prop_path}.")
    with open(args.prop_path, 'r') as f:
        raw_values = [float(line.strip().split()[0]) for line in f if line.strip()]
    
    print(f"Loaded {len(raw_values)} values.")
    
    # Build distribution model
    dist_model = PropertyDistribution(raw_values, num_bins=args.num_bins)
    
    # Sample values
    print(f"Sampling {args.num_samples} values.")
    sampled_values = dist_model.sample(args.num_samples)
    
    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, 'w') as f:
        for val in sampled_values:
            f.write(f"{val.item():.2f}\n")
            
    print(f"Saved to {args.save_path}")

if __name__ == '__main__':
    main()