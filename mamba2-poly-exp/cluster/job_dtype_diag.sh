#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_dtypediag
#$ -M YOUR_NETID@nd.edu
#$ -m abe
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
export PATH=${FHEMAMBA_CONDA_ENV}/bin:$PATH
export HF_HUB_OFFLINE=1
${FHEMAMBA_CONDA_ENV}/bin/python cluster/diag/dtype_diag.py
