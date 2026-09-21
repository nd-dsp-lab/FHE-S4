# Running this project on CRC (Notre Dame)

## Set these first

The job scripts are parameterised so no one's NetID or paths are baked in. Export
these once (e.g. in `~/.bashrc` on the login node):

```bash
export NETID=your_netid
export FHEMAMBA_GROUP=/groups/<your_group>/$NETID      # 5 TB group space
export FHEMAMBA_ROOT=$FHEMAMBA_GROUP/FHEMAMBA          # where this repo lives
export FHEMAMBA_CONDA_ENV=$HOME/.conda/envs/<env>      # must have torch + mamba_ssm
```

For the notification address, **do not edit the job scripts.** A command-line
`-M` overrides the `#$ -M` directive, so pass it at submit time:

```bash
qsub -M your_netid@nd.edu cluster/job_norm_phase1.sh
```

Editing the `#$ -M YOUR_NETID@nd.edu` line works too, but it makes every later
`git pull` abort with "local changes would be overwritten" — which is exactly what
happened the first time these scripts were used. The placeholder is deliberately
fake so a forgotten `-M` fails loudly instead of mailing nobody.

The values this work actually used are in the project's own notes, not here.


## Settings, fixed in one place

| setting | value | where |
|---|---|---|
| queue | `gpu@@jung_gpu` | the `#$ -q` line of every job script |
| GPUs | `-l gpu=1` | one RTX 6000 (24 GB) — what this project is sized for |
| cores | `-pe smp 1` | |
| node | `rtx6k-019` (4 RTX 6000 cards) | reached via the `@@jung_gpu` hostgroup |
| email | `YOUR_NETID@nd.edu` | `#$ -M`, with `#$ -m abe` (mail on abort/begin/end) |

To pin one specific host instead of the hostgroup, change the queue line to
`#$ -q gpu@rtx6k-019.crc.nd.edu`. To ask for more than one card, raise `-l gpu=N`
(up to 4 on that node) — **nothing in this repo needs more than one**, so the
default stays at 1.

Interactive session with the same resources:

```bash
qrsh -q gpu@@jung_gpu -l gpu=1 -pe smp 1
```

## The jobs, in order

```bash
qsub cluster/job_preflight.sh  # 0. RUN FIRST: node + env + all tests + a short eval
qsub cluster/job_parity.sh     # 1. proves our forward == the official kernel
qsub cluster/job_eval.sh       # 2. Parts 6/7/8 with the official backend + real GPU memory
qsub cluster/job_finetune.sh   # 3. Part 9 at 1M tokens, modes A/B/C, plus the control
qsub -v BUDGET=5000000 cluster/job_finetune.sh
qsub cluster/job_distill.sh    # 4. Part 10
```

Monitoring: `qstat -u ${NETID}`, `qstat -j <jobid>`, and the logs land in
`logs/<jobname>.<jobid>.{out,err}`.

**Job 1 is the gate.** Everything in `README.md` currently rests on our reference
SSD matching the official `ssd_minimal_discrete` maths (1e-12) plus a plausible
perplexity — not on a direct comparison against the fused Triton kernel, because
that comparison needs CUDA. If job 1 fails, stop and fix it before reading any
other number.

## Environment: reuse `pdpo`, do not build a new one

**Verified on the node, 2026-09-18.** `${FHEMAMBA_CONDA_ENV}` already has
everything this project needs:

| package | version |
|---|---|
| python | 3.10.18 |
| torch | 2.3.1+cu121 (`cuda available: True` on the node) |
| triton | 2.3.1 |
| **mamba_ssm** | **2.2.2** — the official backend works |
| transformers / einops / pyarrow / numpy | present |
| causal_conv1d | **missing — and that is fine**, `real_mamba/nn_ref.py` has a pure-torch conv |

**Do not `conda create` a new env, and never install from inside a batch job.**

This is not a style preference. On 2026-09-19 `cluster/env.sh` did exactly that:
it ran `conda create` + `pip install torch` on a `$HOME` with 5.5 GB free, built
3.6 GB of a ~7 GB environment, filled the filesystem to **0 bytes**, and died with
`OSError: [Errno 28] No space left on device`. A full `$HOME` also endangers every
other job running at the time. `env.sh` now **verifies and refuses**; it never
installs. If a package is genuinely missing, install it deliberately, from an
interactive session, after checking `df -h $HOME`.

There is no `/scratch365/${NETID}`.

## Data: pre-fetch on the login node

`crcfe01` has internet; assume compute nodes may not. Everything is already
cached under `~/.cache/huggingface` (the 130M checkpoint, the gpt-neox-20b
tokenizer, wikitext-2 train+validation), and the jobs run with `HF_HUB_OFFLINE=1`.
To add wikitext-103 for the larger fine-tuning budgets, fetch it from `crcfe01`
first.

## The node, as it actually is

```
gpu@qa-rtx6k-019.crc.nd.edu   24 slots, 4 x Quadro RTX 6000, 23040 MiB each, sm_75
```

Note the real hostname carries a `qa-` prefix. `@@jung_gpu` resolves to it, so the
queue line in the job scripts is correct as written.

**Contention warning.** The DP-GRPO project submits 4-task array jobs that each
request `gpu_card=1` with `h_rt=36h`, which occupies **all 4 cards and all 24
slots**. While those run, FHEMAMBA jobs sit in `qw`. Check before you promise a
turnaround:

```bash
qstat -f -q gpu@@jung_gpu | head -4      # resv/used/tot slots
qhost -F gpu_card -h qa-rtx6k-019.crc.nd.edu | tail -1   # hc:gpu_card=N free
```

## Where this lives on CRC

**`${FHEMAMBA_ROOT}`** — the group space, not `$HOME`.

| filesystem | size | used | why |
|---|---|---|---|
| `/groups` | 5.0 T | 4% | code, `runs/`, `logs/`, and this project's HF cache |
| `$HOME` | 100 G | 56% | conda envs only; it has been at 96%+ twice |

`$HOME` does **not** hold a copy any more. `~/FHEMAMBA` is a *different*, older
repo (github `jennazhao7/FHEMAMBA`) — never rsync over it.

The HF cache at `${FHEMAMBA_ROOT}/hf-cache` is **this project's own**,
holding only `state-spaces/mamba2-130m`, the gpt-neox-20b tokenizer and wikitext-2
(258 MB). It is deliberately separate from `~/.cache/huggingface`, because the
DP-GRPO project sets `HF_HOME` itself in several of its job scripts and sharing a
cache across projects on a nearly-full filesystem is how you get a surprise.
`cluster/env.sh` points `HF_HOME` at it and falls back to `$HOME` with a warning
if it is missing.

```bash
rsync -av --exclude .venv --exclude runs --exclude '*.pyc' \
      <your local checkout>/ ${NETID}@crcfe01.crc.nd.edu:${FHEMAMBA_ROOT}/
```

Every job script uses `#$ -cwd`, so `qsub` from `${FHEMAMBA_ROOT}`
and outputs land beside the code.

### A quoting trap worth knowing

`ssh host 'cmd with $VAR'` runs `cmd` through the **remote login shell**, which
expands `$VAR` before your `bash -c` ever sees it. That is how a `for d in ...; cp
$d` loop silently copied `hub/` into `hub/`. For anything with variables, write a
script file and `scp` it.

If a shared SSH master is already open (see below), `rsync` and `scp` reuse it
automatically and will not ask for a password again.

## The shared SSH connection

You authenticate once, by hand, in your own terminal:

```bash
ssh -o ControlMaster=yes \
    -o ControlPath=~/.ssh/cm-crcfe01.sock \
    -o ControlPersist=8h \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    ${NETID}@crcfe01.crc.nd.edu
```

After that, any later `ssh`/`scp`/`rsync` to the same host that points at the same
`ControlPath` reuses that authenticated session and needs no password. That socket
**is** live access to your account for as long as it persists — treat it like a
key, keep it in `~/.ssh/` rather than `/tmp`, and close it when you are done:

```bash
ssh -O check -o ControlPath=~/.ssh/cm-crcfe01.sock ${NETID}@crcfe01.crc.nd.edu   # alive?
ssh -O exit  -o ControlPath=~/.ssh/cm-crcfe01.sock ${NETID}@crcfe01.crc.nd.edu   # close it
```

### If remote commands hang at `Loading CRC_default/1.1`

That is the login shell sourcing `/etc/profile.d`, and it stalls when something
in there wants a terminal. Two fixes:

```bash
ssh ... -T 'bash --noprofile --norc -c "cd ~/research/FHEMAMBA && qstat -u ${NETID}"'
ssh ... -tt 'hostname'      # force a pty when a command really does need one
```

A hang here usually means the login node is loaded or the master died, not that
anything is misconfigured. Re-check with `-O check` before debugging further.
