#!/usr/bin/env bash
# Reproducible Colab CLI training workflow for the ATF Fraud Detection Case.
# Training runs as a BACKGROUND process on the VM writing /content/train.log, so you can watch it live:
#   - in this terminal (this script streams new log lines every POLL seconds), and/or
#   - in a second terminal:   watch -n 10 'colab exec -s atf-fraud -f scripts/colab_tail.py --env N=30'
#   - session event history:  colab log -s atf-fraud -n 20      |  status: colab status -s atf-fraud
set -euo pipefail

SESSION="${SESSION:-atf-fraud}"
COLAB_BIN="${COLAB_BIN:-$HOME/.local/bin/colab}"
POLL="${POLL:-15}"
JOB="${JOB:-train}"   # train | ablation
GPU="${GPU:-T4}"   # Colab accelerator (T4, L4, G4, A100, H100); set GPU=none for a CPU-only run
MAX_WAIT_SEC="${MAX_WAIT_SEC:-7200}"
case "$JOB" in
  train)
    PACK="services/ml-inference-service/train.py data/transacoes_historico_2025.csv data/transacoes_jan2026.csv"
    RUN_CMD="python3 -u services/ml-inference-service/train.py --n-jobs \$(nproc --all) __GPU_FLAG__ --log-file /content/train.log > /content/train.stdout 2>&1"
    OUT_PATHS="models/ deliverables/" ;;
  ablation)
    GPU="${GPU_OVERRIDE:-none}"   # CPU models only; no GPU needed
    PACK="services/ml-inference-service/train.py scripts/ablation_protocol_drop.py data/transacoes_historico_2025.csv"
    RUN_CMD="python3 -u scripts/ablation_protocol_drop.py > /content/train.log 2>&1"
    OUT_PATHS="deliverables/ablation_protocol_drop.json deliverables/ablation_protocol_drop.md" ;;
  *) echo "Unknown JOB=$JOB (use train or ablation)"; exit 2 ;;
esac
WORK="$(mktemp -d)"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[1/6] Packaging code and datasets..."
tar -czf "$WORK/atf_fraud_colab_input.tgz" $PACK

echo "[2/6] Provisioning High-RAM Colab VM ($SESSION)..."
if [ "$GPU" = "none" ]; then
    "$COLAB_BIN" new -s "$SESSION" --high-mem
    GPU_FLAG=""
else
    "$COLAB_BIN" new -s "$SESSION" --gpu "$GPU" --high-mem
    GPU_FLAG="--gpu"
fi
trap '"$COLAB_BIN" stop -s "$SESSION" || true' EXIT

echo "[3/6] Uploading bundle and installing aligned package versions..."
"$COLAB_BIN" upload -s "$SESSION" "$WORK/atf_fraud_colab_input.tgz" /content/atf_fraud_colab_input.tgz
"$COLAB_BIN" install -s "$SESSION" numpy==2.5.5 pandas==3.1.1 scikit-learn==1.9.1 xgboost==3.4.1 lightgbm==4.7.0 catboost==1.2.10 joblib==1.6.0 matplotlib==3.11.2 scipy==1.18.1

echo "[4/6] Launching training in the background on the VM..."
cat << PY > "$WORK/launch.py"
import subprocess, pathlib
for f in ("/content/train.log", "/content/train.exit"):
    pathlib.Path(f).unlink(missing_ok=True)
cmd = (
    "mkdir -p /content/atf/deliverables /content/atf/models && tar -xzf /content/atf_fraud_colab_input.tgz -C /content/atf && cd /content/atf && "
    "$RUN_CMD; code=\$?; "
    "if [ \$code -eq 0 ]; then tar -czf /content/atf_fraud_colab_output.tgz $OUT_PATHS; code=\$?; fi; "
    "echo \$code > /content/train.exit"
)
subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
print("launched; log -> /content/train.log")
PY
sed -i "s|__GPU_FLAG__|$GPU_FLAG|" "$WORK/launch.py"
"$COLAB_BIN" exec -s "$SESSION" --timeout 120 -f "$WORK/launch.py"

cat <<MSG

>>> To follow the log from ANOTHER terminal:
    watch -n 10 '$COLAB_BIN exec -s $SESSION -f $HERE/colab_tail.py --env N=30'
    $COLAB_BIN log -s $SESSION -n 20

MSG

echo "[5/6] Streaming log (every ${POLL}s) until the run finishes..."
offset=0
waited=0
while true; do
    out="$("$COLAB_BIN" exec -s "$SESSION" --timeout 120 -f "$HERE/colab_tail.py" --env "OFFSET=$offset" || true)"
    printf '%s\n' "$out" | grep -v -E '^__(OFFSET|EXIT)__=' || true
    new_offset="$(printf '%s\n' "$out" | sed -n 's/^__OFFSET__=//p' | tail -n1)"
    state="$(printf '%s\n' "$out" | sed -n 's/^__EXIT__=//p' | tail -n1)"
    [ -n "$new_offset" ] && offset="$new_offset"
    if [ -n "$state" ] && [ "$state" != "RUNNING" ]; then
        [ "$state" = "0" ] || { echo "Training FAILED on the VM (exit $state). Inspect /content/train.stdout via: $COLAB_BIN download -s $SESSION /content/train.stdout $WORK/train.stdout"; exit 1; }
        break
    fi
    waited=$((waited + POLL)); [ "$waited" -ge "$MAX_WAIT_SEC" ] && { echo "Timed out after ${MAX_WAIT_SEC}s"; exit 1; }
    sleep "$POLL"
done

echo "[6/6] Downloading joblib artifacts, metrics, and classified Jan-2026 dataset..."
"$COLAB_BIN" download -s "$SESSION" /content/atf_fraud_colab_output.tgz "$WORK/atf_fraud_colab_output.tgz"
tar -xzf "$WORK/atf_fraud_colab_output.tgz"
echo "Done! Artifacts extracted to models/ and deliverables/."
