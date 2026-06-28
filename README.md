# Fashion-MNIST — CascadedDP with Non-Uniform Partitioning

## Overview

This project trains a Fashion-MNIST classifier across a two-node HPE Swarm Learning setup using **CascadedDP**: training begins with Differential Privacy active, and DP is automatically dropped once convergence is detected via a quorum vote across all nodes. Data is partitioned using a **Dirichlet distribution** to simulate non-IID heterogeneity.

**Model file:** `fashion-mnist_nonuniform.py`

**Stack:** TensorFlow · TensorFlow Privacy · HPE Swarm Learning

---

## Project Structure

```text
fashion-mnist/
├── cert/
├── ml-context/
├── model/
│   └── fashion-mnist_nonuniform.py
├── results/
│   ├── *.json
│   └── *.log
├── tmp/
│   ├── sl1/
│   ├── sl2/
│   └── shared_scratch/
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MAX_EPOCHS` | `50` | Total training epochs |
| `MIN_PEERS` | `2` | Minimum Swarm peers before sync |
| `NODE_ID` | `0` | Node index (0-based) |
| `NUM_NODES` | `2` | Total number of nodes |
| `OPTIMIZER` | `sgd` | `sgd` or `adam` |
| `LEARNING_RATE` | `0` | Override LR (0 = auto: 0.001 for Adam, 0.01 for SGD) |
| `DP_ENABLED` | `false` | Enable Differential Privacy |
| `NOISE_MULTIPLIER` | `0.0` | Gaussian noise multiplier for DP-SGD/DP-Adam |
| `L2_NORM_CLIP` | `1.0` | Gradient clipping norm for DP |
| `MICROBATCHES` | `32` | Microbatch size for DP optimizer |
| `CASCADED_DP` | `false` | Enable convergence-triggered DP drop |
| `DP_DROP_WINDOW` | `5` | Rolling window size for convergence signals |
| `MIN_DP_EPOCHS` | `5` | Minimum epochs before DP drop is considered |
| `DP_SLOPE_THRESHOLD` | `0.01` | Relative gradient norm slope threshold for convergence |
| `ACC_PLATEAU_THRESHOLD` | `0.0005` | Validation accuracy variance threshold for convergence |
| `DIRICHLET_ALPHA` | `inf` | Dirichlet alpha for partitioning (`inf` = IID) |
| `SCRATCH_DIR` | `/platform/scratch` | Shared directory for quorum vote files |
| `RESULT_FILE` | `results.json` | Output JSON filename under `/results/` |

### Dirichlet Alpha Guide

| `DIRICHLET_ALPHA` | Distribution |
|---|---|
| `inf` | True IID (uniform equal split) |
| `1.0` | Mild heterogeneity |
| `0.5` | Moderate heterogeneity |
| `0.1` | Strong non-IID |
| `0.01` | Extreme non-IID |
| `0.005` | Near-degenerate non-IID |

---

## Setup

### 0. Clone Project Repository into Workspace

```bash
cd ~/swarm-learning/workspace/

git clone https://github.com/clearlynew/Fashion-MNIST-using-Cascaded-DP.git fashion-mnist
```

### 1. Generate Certificates

```bash
cd ~/swarm-learning/

cp -r examples/utils/gen-cert workspace/fashion-mnist/

./workspace/fashion-mnist/gen-cert -e fashion-mnist -i 1
./workspace/fashion-mnist/gen-cert -e fashion-mnist -i 2
```

### 2. Remove SWOP/SWCI Certificates

```bash
cd workspace/fashion-mnist/cert
rm swop-* swci-*
cd ../../../
```

### 3. Create Docker Network

```bash
docker network create host-1-net
```

### 4. Create Required Directories

```bash
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/sl1
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/sl2
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch
mkdir -p ~/swarm-learning/workspace/fashion-mnist/results

chmod -R 777 ~/swarm-learning/workspace/fashion-mnist/tmp
chmod -R 777 ~/swarm-learning/workspace/fashion-mnist/results
```

> `shared_scratch` is required for the quorum vote files used by CascadedDP. Both nodes must mount the same directory.

### 5. Copy SwarmLearning Wheel

```bash
cp ~/swarm-learning/lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl \
~/swarm-learning/workspace/fashion-mnist/ml-context/swarmlearning-0.0.1-py3-none-manylinux_2_24_x86_64.whl
```

### 6. Build ML Docker Image

```bash
docker build -t fashion-ml-env \
~/swarm-learning/workspace/fashion-mnist/ml-context
```

### 7. Run APLS

```bash
docker run -d \
--name apls \
--network host-1-net \
-v apls-volume:/hpe \
-p 5814:5814 \
--restart unless-stopped \
hub.myenterpriselicense.hpe.com/hpe_eval/autopass/apls:9.19
```

### 8. Set Environment Variables

```bash
export HOST_IP=172.1.1.1
export SN_IP=172.1.1.1
export APLS_IP=172.1.1.1
export SN_API_PORT=30304
```

### 9. Run Swarm Network Node

```bash
cd ~/swarm-learning

./scripts/bin/run-sn -d --name=sn1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sentinel \
--sn-api-port=${SN_API_PORT} \
--key=workspace/fashion-mnist/cert/sn-1-key.pem \
--cert=workspace/fashion-mnist/cert/sn-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--apls-ip=${APLS_IP}
```

Wait until the following appears in logs before launching SL nodes:

```bash
docker logs -f sn1
# Wait for: swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
```

---

## Running Experiments

> Before each new experiment, stop old containers:
> ```bash
> docker rm -f sn1 sl1 sl2 ml1 ml2 2>/dev/null
> ```

---

### Experiment 1 — Baseline (No DP)

#### SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_nonuniform.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_baseline_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e DIRICHLET_ALPHA=0.5 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=false \
--apls-ip=${APLS_IP}
```

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_baseline_ml1.log 2>&1 &
```

#### SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_nonuniform.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_baseline_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e DIRICHLET_ALPHA=0.5 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=false \
--apls-ip=${APLS_IP}
```

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_baseline_ml2.log 2>&1 &
```

---

### Experiment 2 — Full DP (No Drop)

#### SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_nonuniform.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_full_dp_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e DIRICHLET_ALPHA=0.5 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=false \
--apls-ip=${APLS_IP}
```

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_full_dp_ml1.log 2>&1 &
```

#### SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_nonuniform.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_full_dp_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e DIRICHLET_ALPHA=0.5 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=false \
--apls-ip=${APLS_IP}
```

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_full_dp_ml2.log 2>&1 &
```

---

### Experiment 3 — CascadedDP (Convergence-Triggered DP Drop)

#### SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_nonuniform.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_cascaded_dp_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e DIRICHLET_ALPHA=0.5 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=true \
--ml-e DP_DROP_WINDOW=5 \
--ml-e MIN_DP_EPOCHS=5 \
--ml-e DP_SLOPE_THRESHOLD=0.01 \
--ml-e ACC_PLATEAU_THRESHOLD=0.0005 \
--apls-ip=${APLS_IP}
```

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_cascaded_dp_ml1.log 2>&1 &
```

#### SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_nonuniform.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_cascaded_dp_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e DIRICHLET_ALPHA=0.5 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=true \
--ml-e DP_DROP_WINDOW=5 \
--ml-e MIN_DP_EPOCHS=5 \
--ml-e DP_SLOPE_THRESHOLD=0.01 \
--ml-e ACC_PLATEAU_THRESHOLD=0.0005 \
--apls-ip=${APLS_IP}
```

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_cascaded_dp_ml2.log 2>&1 &
```

---

## How CascadedDP Works

Training starts with the DP optimizer active. After `MIN_DP_EPOCHS` epochs, each node monitors two signals every epoch:

- **Relative gradient norm slope** — the fractional change in the rolling mean of gradient norms across the last `DP_DROP_WINDOW` epochs.
- **Validation accuracy variance** — variance of validation accuracy over the same window.

When both fall below their respective thresholds (`DP_SLOPE_THRESHOLD` and `ACC_PLATEAU_THRESHOLD`), a node writes a vote file to `SCRATCH_DIR`. Once all `NUM_NODES` vote files are present (full quorum), every node simultaneously replaces the DP optimizer with a standard SGD/Adam optimizer and training continues without DP.

Privacy accounting (`epsilon`) is computed only for the epochs in which DP was active, so the reported epsilon reflects the true privacy cost.

---

## Notes

- TensorFlow: `2.7.0`
- TensorFlow Privacy: `0.7.3`
- TensorFlow Probability: `0.15.0`
- Both nodes must mount the **same physical directory** as `/tmp/scratch` for the quorum vote mechanism to work.
- Adjust `DIRICHLET_ALPHA` to sweep heterogeneity levels across experiments. Lower values produce more skewed class distributions.
