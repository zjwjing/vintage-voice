#!/bin/bash
# =============================================================================
# CosyVoice2-0.5B — Cajun + PRAIRIE (St. Landry) LLM finetune  (single 8GB GPU)
# Merged corpus: existing 8.1h Cajun  +  prairie 1.03h UPWEIGHTED x3.
# Fresh finetune from base llm.pt (same recipe that produced ep2), so CV loss
# picks the best epoch. Stops the llamas for the GPU run, restores on exit.
# =============================================================================
set -u
exec >/tmp/prairie_train.log 2>&1

REPO=/home/scott/vintage-voice/models/cosyvoice-repo
MODEL=/home/scott/vintage-voice/models/CosyVoice2-0.5B
EIGHT=/home/scott/vintage-voice/data/cosyvoice_cajun_8h
PRAIRIE=/home/scott/vintage-voice/data/cosyvoice_prairie
DATA=/home/scott/vintage-voice/data/cosyvoice_cajun_prairie
PY=/home/scott/vintage-voice/venv-cosy/bin/python
RESTORE=/home/scott/vintage-voice/data/llama_restore_prairie_train.sh
UPWEIGHT=3
log(){ echo "[$(date +%H:%M:%S)] $*"; }

restore_llamas(){
  systemctl --user start openclaw-llama 2>/dev/null || true
  if ! pgrep -f 'llama-server.*port 8082' >/dev/null && [ -f "$RESTORE" ]; then
    bash "$RESTORE" 2>/dev/null || true
  fi
  log "llamas restored ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"
}
trap restore_llamas EXIT

# --- build merged data dir ---------------------------------------------------
mkdir -p "$DATA"
cp -r "$EIGHT/conf" "$DATA/conf"
# train.data.list: 8h tars once + prairie tars x UPWEIGHT (absolute paths)
EIGHT_TRAIN=$(sed 's|^data/|/home/scott/vintage-voice/data/|' "$EIGHT/train/parquet/data.list")
PRAIRIE_TRAIN=$(cat "$PRAIRIE/train/parquet/data.list")
{ echo "$EIGHT_TRAIN"; for i in $(seq 1 $UPWEIGHT); do echo "$PRAIRIE_TRAIN"; done; } | grep -v '^$' > "$DATA/train.data.list"
EIGHT_DEV=$(sed 's|^data/|/home/scott/vintage-voice/data/|' "$EIGHT/dev/parquet/data.list")
PRAIRIE_DEV=$(cat "$PRAIRIE/dev/parquet/data.list")
{ echo "$EIGHT_DEV"; echo "$PRAIRIE_DEV"; } | grep -v '^$' > "$DATA/dev.data.list"
log "merged train tars: $(wc -l < "$DATA/train.data.list")  dev tars: $(wc -l < "$DATA/dev.data.list")  (prairie x$UPWEIGHT)"

# --- free GPU (stop llamas) --------------------------------------------------
P8082=$(pgrep -f 'llama-server.*port 8082' || true)
[ -n "$P8082" ] && ps -o args= -p $P8082 | sed 's/^/nohup /; s|$| >> /tmp/llama_restored.log 2>\&1 \&|' > "$RESTORE"
log "stopping llamas"
systemctl --user stop openclaw-llama 2>/dev/null || true
[ -n "$P8082" ] && { kill $P8082 2>/dev/null; sleep 4; kill -9 $P8082 2>/dev/null || true; }
for i in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 800 ] && break; sleep 5; done
log "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

# --- env (per cosyvoice_train_llm.sh) ----------------------------------------
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${REPO}:${REPO}/third_party/Matcha-TTS:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
export MASTER_ADDR=localhost MASTER_PORT=29537
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
train_engine=torch_ddp

cd "$DATA"
log "=== TRAIN START (llm, merged cajun+prairie) ==="
${PY} ${REPO}/cosyvoice/bin/train.py \
  --train_engine ${train_engine} \
  --config ${DATA}/conf/cosyvoice2_8gb.yaml \
  --train_data ${DATA}/train.data.list \
  --cv_data ${DATA}/dev.data.list \
  --qwen_pretrain_path ${MODEL}/CosyVoice-BlankEN \
  --onnx_path ${MODEL} \
  --model llm \
  --checkpoint ${MODEL}/llm.pt \
  --model_dir /home/scott/vintage-voice/exp/cajun_prairie/llm \
  --tensorboard_dir /home/scott/vintage-voice/exp/cajun_prairie/llm/tensorboard \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
RC=$?
log "=== TRAIN DONE rc=$RC ==="
ls -la /home/scott/vintage-voice/exp/cajun_prairie/llm/*.pt 2>/dev/null
log "TRAIN COMPLETE"
