#!/usr/bin/env bash
# BTI-Net, its ablations and the compared methods on BUSI, BRISC and HAM10000 with seeds 42, 43 and 44.
#   bash scripts/run_experiments.sh [btinet|ablations|multitask|references]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
WHAT="${1:-all}"
DATASETS="BUSI BRISC HAM10000"
SEEDS="42 43 44"

run () {  # run <output group> <name> <command...>
  local group=$1 name=$2; shift 2
  for ds in $DATASETS; do for s in $SEEDS; do
    out="results/$group/$ds/$name/seed$s"
    [ -f "$out/results.json" ] || "$@" "$ds" "$s" "$out"
  done; done
}

btinet () { python -u train_btinet.py "$1" "$2" "$3" ${ABL:-}; }
compared () { python -u train_compared.py "$1" "$M" "$2" "$3"; }

if [[ "$WHAT" == all || "$WHAT" == btinet ]]; then
  ABL= run btinet full btinet
fi
if [[ "$WHAT" == all || "$WHAT" == ablations ]]; then
  for ABL in no_tim no_c2s no_s2c gate_one gate_const no_gsup gate_aiw no_gctx_pres; do
    run ablations "$ABL" btinet
  done
fi
if [[ "$WHAT" == all || "$WHAT" == multitask ]]; then
  for M in mtan mti_net densemtl mtanet mtl_oca aiw; do run multitask "$M" compared; done
fi
if [[ "$WHAT" == all || "$WHAT" == references ]]; then
  for M in unet att_unet unet_pp transunet swin_unet missformer stl_seg stl_clf; do run references "$M" compared; done
fi
