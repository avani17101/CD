#!/bin/bash
#SBATCH --job-name=sv_vs_cd
#SBATCH --output=sv_out/sv_vs_cd.%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
## Steering vectors (ActAdd) vs CD-lite on ColorMNIST. Submit from src/: sbatch scripts/run_sv_vs_cd.sh
## (add -p/--qos for your cluster). Needs dep/data/ColorMNIST (mnist/ColorMNIST/00_make_data.py) and data/MNIST.
for s in 0 1 2; do
  python sv_debias.py --seed $s --out sv_out/sv_seed$s.json                  # ActAdd variants + random/projection controls
  python sv_layer_sweep.py --seed $s --out sv_out/sweep_seed$s.json            # layer x coeff x n_pairs sweep, random control per cell
  python sv_vs_cd_efficiency.py --seed $s --out sv_out/eff_seed$s.json       # timed SV vs CD-lite
done
