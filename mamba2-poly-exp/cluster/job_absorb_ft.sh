#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_absorb_ft
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# NO `#$ -M` / `#$ -m` DIRECTIVE ON PURPOSE -- pass it at submit time:
#     qsub -M you@nd.edu -m abe cluster/job_absorb_ft.sh
#
# RMSNorm ABSORPTION + FINE-TUNING, round 1 (norm/absorb_ft.py).
# The 24 residual pre-norms become per-layer constants folded into in_proj (zero
# levels), and the WHOLE model is fine-tuned to live with them -- THE-X's recipe
# with the backbone unfrozen, which Path A deliberately did not do.
#
# 1M tokens, the same budget as Path A's +39 ppl (constants only), so the two
# compare directly. Per learning rate: a CONTROL (exact model, same recipe), then
# absorbed with and without the v-spread penalty. The verdict is absorbed vs
# control, paired per block:  python norm/compare_absorb.py <absorbed> <control>
#
# Override with -v:  ABSORB_LRS, ABSORB_VSPREADS, ABSORB_BUDGET
: "${FHEMAMBA_CONDA_ENV:=$HOME/.conda/envs/pdpo}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

: "${ABSORB_LRS:=2e-5 1e-4}"
: "${ABSORB_VSPREADS:=0 0.1}"
: "${ABSORB_BUDGET:=1000000}"
COMMON="--backend official --device cuda --stage A1 --token-budget $ABSORB_BUDGET \
        --train-data wikitext103 --eval-lengths 512 2048 --eval-blocks 40"
echo "[absorb] lrs='$ABSORB_LRS' vspreads='$ABSORB_VSPREADS' budget=$ABSORB_BUDGET"

for LR in $ABSORB_LRS; do
  echo; echo "############ CONTROL  lr=$LR ############"
  python norm/absorb_ft.py $COMMON --control --lr-backbone "$LR" \
      --outdir "runs/norm_absorb/control_lr${LR}_${ABSORB_BUDGET}tok"
  for VS in $ABSORB_VSPREADS; do
    echo; echo "############ ABSORBED  lr=$LR  vspread=$VS ############"
    python norm/absorb_ft.py $COMMON --lr-backbone "$LR" --lambda-vspread "$VS" \
        --outdir "runs/norm_absorb/absorb_A1_lr${LR}_vs${VS}_${ABSORB_BUDGET}tok"
    python norm/compare_absorb.py "runs/norm_absorb/absorb_A1_lr${LR}_vs${VS}_${ABSORB_BUDGET}tok" \
        "runs/norm_absorb/control_lr${LR}_${ABSORB_BUDGET}tok"
  done
done
echo "absorb_ft finished at $(date)"
