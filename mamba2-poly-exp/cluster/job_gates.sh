#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_gates
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# The gate work at proper scale. Everything in GATES.md was measured on 6 x 1024
# tokens on a laptop CPU, and the SiLU margin results swing over 40 orders of
# magnitude -- exactly the regime where 6 blocks of interval measurement is not
# enough. This re-measures on 64 blocks and re-runs the sweeps.
source cluster/env.sh
source cluster/job_common.sh

echo "=== 1. re-measure every gate input at scale (64 x 1024, was 6) ==="
python collect_gate_stats.py --backend official --device cuda \
    --blocks 64 --seq-len 1024 --outdir runs/gate_stats_gpu

echo "=== 2. gate-by-gate evaluation on the better intervals ==="
python eval_gates.py --backend official --device cuda --degree 4 \
    --blocks 120 --batch-size 4 \
    --gate-stats runs/gate_stats_gpu/gate_stats.json \
    --outdir runs/gates_eval_gpu

echo "=== 3. does more measurement data fix the SiLU fragility? ==="
for M in 0.0 0.02 0.05 0.10 0.20; do
  echo "--- silu margin $M ---"
  python eval_gates.py --backend official --device cuda --degree 4 \
      --blocks 60 --batch-size 4 --silu-margin "$M" \
      --gate-stats runs/gate_stats_gpu/gate_stats.json \
      --outdir "runs/gates_margin_gpu/m$M" 2>&1 | grep -E "SiLU|FUSED|ALL FOUR"
done

echo "=== 4. degree sweep -- is 4 the right band? ==="
for D in 2 3 4 6; do
  echo "--- degree $D ---"
  python eval_gates.py --backend official --device cuda --degree "$D" \
      --blocks 60 --batch-size 4 --silu-margin 0.05 \
      --gate-stats runs/gate_stats_gpu/gate_stats.json \
      --outdir "runs/gates_degree_gpu/d$D" 2>&1 | grep -E "softplus only|SiLU|FUSED|ALL FOUR"
done

echo "gates job finished at $(date)"
