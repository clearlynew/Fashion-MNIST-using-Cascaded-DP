# Fashion-MNIST — CascadedDP with Non-Uniform Partitioning

=========================================================

This example trains a Fashion-MNIST classifier across a two-node HPE Swarm Learning setup using **CascadedDP**: training begins with Differential Privacy (DP) active, and DP is automatically dropped once convergence is detected via a decentralized weight parameter consensus protocol. Data is partitioned using a **Dirichlet distribution** to simulate non-IID heterogeneity.

The ML program is in `workspace/fashion-mnist/model` and is called `fashion-mnist_nonuniform.py`.

**Stack:** TensorFlow · TensorFlow Privacy · HPE Swarm Learning

This example shows the Swarm training of a Fashion-MNIST classifier using two Machine Learning (ML) nodes launched directly via `run-sl`, without SWOP or SWCI. All nodes run on a single host. This example also shows how private data, scratch spaces, and model files can be mounted to Machine Learning nodes for Swarm training.

## Cluster Setup

The cluster setup for this example uses only one host, as shown in the figure below:

* host-1: 172.1.1.1

|<img width="1372" height="771" alt="fraud-detection-cluster-setup" src="https://github.com/user-attachments/assets/52e11724-6b98-4ca7-9763-3b12f7edbcb1" />|
|:--:|
|<b>Figure 1: Cluster setup for the Credit card fraud detection example</b>|


1. This example uses one Swarm Network (SN) node. The name of the docker container representing this node is **sn1**. sn1 is also the Sentinel Node. sn1 runs on host 172.1.1.1.
2. Two Swarm Learning (SL) and two Machine Learning (ML) nodes are launched directly using `run-sl`. The names of the docker containers representing these nodes are **sl1** and **sl2**, with associated ML containers **ml1** and **ml2**. Both run on host 172.1.1.1.
3. Training begins automatically once both SL nodes are up and the `MIN_PEERS` quorum is satisfied — no SWCI node is required.
4. This example assumes that a License Server (APLS) already runs on host 172.1.1.1. All Swarm nodes connect to the License Server on its default port 5814.

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
│   └── sl2/
└── README.md

```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_EPOCHS` | `50` | Total training epochs |
| `MIN_PEERS` | `2` | Minimum Swarm peers before sync |
| `NODE_ID` | `0` | Node index (0-based) |
| `NUM_NODES` | `2` | Total number of nodes |
| `OPTIMIZER` | `sgd` | `sgd` or `adam` |
| `LEARNING_RATE` | `0` | Override LR (0 = auto: 0.001 for Adam, 0.01 for SGD) |
| `DP_ENABLED` | `true` | Enable Differential Privacy |
| `NOISE_MULTIPLIER` | `0.0` | Gaussian noise multiplier for DP-SGD/DP-Adam |
| `L2_NORM_CLIP` | `1.0` | Gradient clipping norm for DP |
| `MICROBATCHES` | `32` | Microbatch size for DP optimizer |
| `CASCADED_DP` | `false` | Enable convergence-triggered DP drop |
| `DP_DROP_WINDOW` | `5` | Rolling window size for convergence signals |
| `MIN_DP_EPOCHS` | `5` | Minimum epochs before DP drop is considered |
| `DP_SLOPE_THRESHOLD` | `0.01` | Relative gradient norm slope threshold for convergence |
| `ACC_PLATEAU_THRESHOLD` | `0.0005` | Validation accuracy variance threshold for convergence |
| `DIRICHLET_ALPHA` | `inf` | Dirichlet alpha for partitioning (`inf` = IID) |
| `RESULT_FILE` | `results.json` | Output JSON filename under `/results/` |

### Dirichlet Alpha Guide

| `DIRICHLET_ALPHA` | Distribution |
| --- | --- |
| `inf` | True IID (uniform equal split) |
| `1.0` | Mild heterogeneity |
| `0.5` | Moderate heterogeneity |
| `0.1` | Strong non-IID |
| `0.01` | Extreme non-IID |
| `0.005` | Near-degenerate non-IID |

## Running the Fashion-MNIST CascadedDP Example

### Setup

1. *On host-1*:
Clone the project repository into the workspace directory.
```bash
cd ~/swarm-learning/workspace/
git clone https://github.com/clearlynew/Fashion-MNIST-using-Cascaded-DP.git fashion-mnist

```


2. *On host-1*:
Copy the `gen-cert` utility into the cloned project and generate certificates for each Swarm component.
```bash
cd ~/swarm-learning/
cp -r examples/utils/gen-cert workspace/fashion-mnist/
./workspace/fashion-mnist/gen-cert -e fashion-mnist -i 1
./workspace/fashion-mnist/gen-cert -e fashion-mnist -i 2

```


3. *On host-1*:
Remove the SWOP and SWCI certificates that were auto-generated but are not needed for this setup.
```bash
cd workspace/fashion-mnist/cert
rm swop-* swci-*
cd ../../../

```


4. *On host-1*:
Create a network called `host-1-net` using the docker network create command. This network will be used for SN, SL, and ML containers. Please ignore this step if this network is already created.
```bash
docker network create host-1-net

```


5. *On host-1*:
Create separate temporary mount directories for each SL node and a results directory. Set appropriate permissions.
```bash
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/sl1
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/sl2
mkdir -p ~/swarm-learning/workspace/fashion-mnist/results
chmod -R 777 ~/swarm-learning/workspace/fashion-mnist/tmp
chmod -R 777 ~/swarm-learning/workspace/fashion-mnist/results

```


6. *On host-1*:
Copy the SwarmLearning wheel file into the ML Docker build context.
```bash
cp ~/swarm-learning/lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl \
~/swarm-learning/workspace/fashion-mnist/ml-context/swarmlearning-0.0.1-py3-none-manylinux_2_24_x86_64.whl

```


7. *On host-1*:
Build the ML Docker image that will be used to run the Fashion-MNIST model inside the ML containers.
```bash
docker build -t fashion-ml-env \
~/swarm-learning/workspace/fashion-mnist/ml-context

```


8. *On host-1*:
Run the APLS license server container if it is not already running or not connected.
```bash
docker run -d \
--name apls \
--network host-1-net \
-v apls-volume:/hpe \
-p 5814:5814 \
--restart unless-stopped \
hub.myenterpriselicense.hpe.com/hpe_eval/autopass/apls:9.19

```


9. *On host-1*:
Declare and assign values to the environment variables. The values mentioned here are for illustration purposes only. Use appropriate values as per your swarm network (check your machine's IP using `hostname -I`).
```bash
export HOST_IP=172.1.1.1
export SN_IP=172.1.1.1
export APLS_IP=172.1.1.1
export SN_API_PORT=30304

```


10. *On host-1*:
Run the Swarm Network node (sn1) — this is the Sentinel node.
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


Use the docker logs command to monitor the Sentinel SN node and wait for the node to finish initializing. The Sentinel node is ready when this message appears in the log output:
```
swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304

```



---

### Running Experiments

> ***NOTE :*** Before starting each new experiment, stop and remove containers from the previous run:
> ```bash
> docker rm -f sn1 sl1 sl2 ml1 ml2 2>/dev/null
> 
> ```
> 
> 
> Then re-run the SN node (step 10 above) and wait for it to initialize before launching SL nodes.

---

### Experiment 1 — Baseline (No DP)

11. *On host-1*: Run SL1
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
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
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


12. *On host-1*: Run SL2
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
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
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

13. *On host-1*: Run SL1
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
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
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


14. *On host-1*: Run SL2
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
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
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

15. *On host-1*: Run SL1
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
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
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


16. *On host-1*: Run SL2
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
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
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


Swarm training will end with the following log message at the end —
`SwarmCallback : INFO : All peers and Swarm training rounds finished. Final Swarm model was loaded.`
Result JSON files will be saved under `workspace/fashion-mnist/results/`. To clean up, stop and remove all containers and remove the `workspace` directory.

## How CascadedDP Works

Training starts with the DP optimizer active. After `MIN_DP_EPOCHS` epochs, each node monitors two local signals every epoch:

* **Relative gradient norm slope** — the fractional change in the rolling mean of gradient norms across the last `DP_DROP_WINDOW` epochs.
* **Validation accuracy variance** — variance of validation accuracy over the same window.

When both signals fall below their respective thresholds (`DP_SLOPE_THRESHOLD` and `ACC_PLATEAU_THRESHOLD`), the node trips its consensus flag by changing an internal, non-trainable tracking model parameter from `0.0` to `1.0` via a custom model layer (`ConvergenceFlagLayer`).

During Swarm parameter synchronization rounds, the Swarm learning mechanism naturally aggregates these tracking weights via mathematical averaging across all participants:

$$\text{Global Consensus Value} = \frac{1}{N} \sum_{i=1}^{N} \text{Node Flag}_i$$

Once all `NUM_NODES` achieve convergence, the unified network model average evaluates to exactly `1.0`. Detecting this network quorum consensus, every node simultaneously re-compiles its operational execution path, unlinking the DP wrapper to continue execution utilizing standard SGD/Adam optimization.

Privacy accounting (`epsilon`) is computed only for the exact epochs in which DP was actively running, reflecting the true privacy cost.

## Notes

* TensorFlow: `2.7.0`
* TensorFlow Privacy: `0.7.3`
* TensorFlow Probability: `0.15.0`
* No shared volume or scratch directory mounting is required for voting; consensus state resolution is calculated natively via the model parameter synchronization framework.
* Adjust `DIRICHLET_ALPHA` to sweep heterogeneity levels across experiments. Lower values produce more skewed class distributions.
