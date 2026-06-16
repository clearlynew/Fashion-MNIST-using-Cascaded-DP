import os
import glob
import json
import time
import pickle
import numpy as np
import tensorflow as tf
from collections import deque
from sklearn.metrics import f1_score

from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy
)

from swarmlearning.tf import SwarmCallback

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

batchSize       = 32
defaultMaxEpoch = 20
defaultMinPeers = 2


# ─────────────────────────────────────────────────────────────────────────────
# EPOCH SYNCHRONIZATION BARRIER (for noniid_unequal)
# ─────────────────────────────────────────────────────────────────────────────

class EpochBarrierCallback(tf.keras.callbacks.Callback):
    """
    Lightweight epoch-level barrier: each node writes a heartbeat file
    at the END of every epoch, then busy-waits until ALL peers have
    written that same epoch's heartbeat before proceeding.

    This keeps the fast (small-data) node in lockstep with the slow
    (large-data) node so the CascadedDP quorum window is always
    evaluated at the same logical training epoch across the cluster.
    """

    def __init__(self, node_id, num_nodes, scratch_dir,
                 poll_interval=2.0, timeout=7200.0):
        super().__init__()
        self.node_id       = node_id
        self.num_nodes     = num_nodes
        self.scratch_dir   = scratch_dir
        self.poll_interval = poll_interval   # seconds between fs polls
        self.timeout       = timeout         # max wait per epoch (safety valve)

        # Clean up stale heartbeats from previous runs
        for f in glob.glob(os.path.join(scratch_dir, ".epoch_barrier_*")):
            try:
                os.remove(f)
            except Exception:
                pass

    def _heartbeat_path(self, peer_id, epoch):
        return os.path.join(
            self.scratch_dir,
            f".epoch_barrier_node_{peer_id}_epoch_{epoch}"
        )

    def on_epoch_end(self, epoch, logs=None):
        # 1. Announce this node has finished epoch `epoch`
        hb = self._heartbeat_path(self.node_id, epoch)
        with open(hb, 'w') as f:
            f.write(f"node={self.node_id} epoch={epoch} done")

        print(f"  [EpochBarrier] Node {self.node_id} waiting at epoch {epoch + 1} ...")

        # 2. Busy-wait until every peer has written its heartbeat
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            votes = sum(
                1 for pid in range(self.num_nodes)
                if os.path.exists(self._heartbeat_path(pid, epoch))
            )
            if votes == self.num_nodes:
                print(f"  [EpochBarrier] All {self.num_nodes} nodes at epoch {epoch + 1} — proceeding.")
                return
            time.sleep(self.poll_interval)

        # Timeout is non-fatal; log and continue so training isn't permanently stalled
        print(f"  [EpochBarrier] WARNING: timeout waiting for peers at epoch {epoch + 1}. Proceeding anyway.")


# ─────────────────────────────────────────────────────────────────────────────
# DECENTRALIZED CASCADED DP CALLBACK (SWARM CONSENSUS)
# ─────────────────────────────────────────────────────────────────────────────

class CascadedDPCallback(tf.keras.callbacks.Callback):
    """
    Drops differential privacy across ALL swarm nodes simultaneously using a
    fully decentralized Quorum-based Peer Consensus mechanism.
    """

    def __init__(
        self,
        val_ds,
        node_id,
        num_nodes,
        scratch_dir,
        optimizer_type='sgd',
        learning_rate=0.01,
        window_size=5,
        slope_threshold=0.015,
        acc_plateau_threshold=0.0005,
        min_dp_epochs=5,
    ):
        super().__init__()

        self.val_ds                = val_ds
        self.node_id               = node_id
        self.num_nodes             = num_nodes
        self.scratch_dir           = scratch_dir
        self.optimizer_type        = optimizer_type
        self.learning_rate         = learning_rate
        self.window_size           = window_size
        self.slope_threshold       = slope_threshold
        self.acc_plateau_threshold = acc_plateau_threshold
        self.min_dp_epochs         = min_dp_epochs

        # Tracking windows
        self.grad_norm_window = deque(maxlen=window_size)
        self.acc_window        = deque(maxlen=window_size)

        # Telemetry history tracking
        self.grad_history     = []
        self.rolling_history  = []
        self.acc_history      = []

        self.dp_active        = True
        self.dp_drop_epoch    = None
        self.dp_drop_reason   = None

        # Consensus directory setup
        self.vote_file = os.path.join(self.scratch_dir, f".vote_drop_dp_node_{self.node_id}")

        # Clean up old votes from previous experimental runs at bootup
        if os.path.exists(self.vote_file):
            try:
                os.remove(self.vote_file)
            except Exception:
                pass

        self._measure_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def _compute_grad_norm(self):
        """Compute stable raw gradient norm dynamically sampled across 5 val batches."""
        norms = []
        for x, y in self.val_ds.take(5):
            with tf.GradientTape() as tape:
                preds    = self.model(x, training=True)
                loss_val = self._measure_loss(y, preds)

            grads     = tape.gradient(loss_val, self.model.trainable_variables)
            grad_norm = tf.linalg.global_norm(grads).numpy()
            norms.append(float(grad_norm))
        return float(np.mean(norms))

    def _drop_dp(self, epoch):
        """Swap to standard optimizer + loss, recompile, and force rebuild the runtime graph."""

        print(f"\n***** CascadedDP: [Node {self.node_id}] SWARM QUORUM UNLOCKED *****")
        print(f"***** CascadedDP: dropping DP globally at epoch {epoch + 1} *****")

        if self.optimizer_type == 'adam':
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            new_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)

        new_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # Recompile model structure
        self.model.compile(
            loss=new_loss,
            optimizer=new_optimizer,
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
        )

        # ── FORCE REBUILD OF LOW-LEVEL KERAS RUNTIME GRAPH FUNCTIONS ──
        if hasattr(self.model, 'train_function'):
            self.model.train_function = None
            self.model.test_function = None
            self.model.predict_function = None

            # Run immediate graph tracing session to bypass NoneType crash
            for x_sample, y_sample in self.val_ds.take(1):
                self.model.make_train_function()
                self.model.make_test_function()
                self.model.make_predict_function()

        self.dp_active     = False
        self.dp_drop_epoch = epoch + 1

        self.dp_drop_reason = {
            "epoch": epoch + 1,
            "slope_threshold": self.slope_threshold,
            "acc_variance_threshold": self.acc_plateau_threshold,
            "rolling_mean": float(np.mean(self.grad_norm_window)),
            "val_acc_window": list(self.acc_window)
        }

        print(f"***** CascadedDP: low-level execution graphs forcefully rebuilt *****")
        print(f"***** CascadedDP: model recompiled with standard optimizer *****\n")

    def on_epoch_end(self, epoch, logs=None):
        if not self.dp_active:
            return

        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        if val_acc is None:
            return

        grad_norm = self._compute_grad_norm()
        self.grad_norm_window.append(grad_norm)

        rolling_mean = np.mean(self.grad_norm_window)
        self.acc_window.append(val_acc)

        self.grad_history.append(float(grad_norm))
        self.rolling_history.append(float(rolling_mean))
        self.acc_history.append(float(val_acc))

        print(
            f"  [CascadedDP] Node={self.node_id} | epoch={epoch + 1} | "
            f"grad_norm={grad_norm:.6f} | rolling_mean={rolling_mean:.6f} | val_acc={val_acc:.4f}"
        )

        # 1. Guard check: Minimum epochs restriction
        if epoch + 1 < self.min_dp_epochs or len(self.grad_norm_window) < self.window_size:
            return

        # 2. Local Node Audit Evaluation
        if len(self.rolling_history) >= 2:
            prev_mean = self.rolling_history[-2]
            curr_mean = self.rolling_history[-1]
            relative_slope = abs(prev_mean - curr_mean) / prev_mean
        else:
            relative_slope = 1.0

        acc_variance = float(np.var(self.acc_window))

        # If this specific node converges, it casts its vote to the cluster volume
        if relative_slope < self.slope_threshold and acc_variance < self.acc_plateau_threshold:
            if not os.path.exists(self.vote_file):
                try:
                    with open(self.vote_file, 'w') as f:
                        f.write(f"Node {self.node_id} converged at epoch {epoch + 1}")
                    print(f"  [CascadedDP-Consensus] Node {self.node_id} posted drop vote to shared scratch.")
                except Exception as e:
                    print(f"  [CascadedDP-Consensus] Error writing vote file: {e}")

        # 3. Scan cluster volume for total cast votes
        total_votes = 0
        for peer_id in range(self.num_nodes):
            peer_vote_path = os.path.join(self.scratch_dir, f".vote_drop_dp_node_{peer_id}")
            if os.path.exists(peer_vote_path):
                total_votes += 1

        print(f"  [CascadedDP-Quorum] Node {self.node_id} reporting cluster status: {total_votes}/{self.num_nodes} votes collected.")

        # 4. Strict Consensus Check: Trigger drop ONLY when ALL nodes agree
        if total_votes == self.num_nodes:
            # Short sleep delay matching node rank to prevent file system access race conditions
            time.sleep(self.node_id * 0.2)
            self._drop_dp(epoch)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_optimizer(
    optimizer_type,
    dp_enabled,
    learning_rate,
    l2_norm_clip,
    noise_multiplier,
    microbatches
):
    """Return DP or standard optimizer."""
    if dp_enabled:
        if optimizer_type == 'adam':
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
            print("***** Using DP-Adam optimizer")
            return DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate or 0.001
            )
        else:
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
            print("***** Using DP-SGD optimizer")
            return DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate or 0.01
            )
    else:
        if optimizer_type == 'adam':
            print("***** Using Adam optimizer")
            return tf.keras.optimizers.Adam(learning_rate=learning_rate or 0.001)
        else:
            print("***** Using SGD optimizer")
            return tf.keras.optimizers.SGD(learning_rate=learning_rate or 0.01, momentum=0.9, nesterov=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def get_metrics():
    return [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    modelName = 'fashion-mnist'

    # Read env vars
    scratchDir      = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch        = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers        = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))
    dpEnabled       = os.getenv('DP_ENABLED', 'false').lower() == 'true'
    noiseMultiplier = float(os.getenv('NOISE_MULTIPLIER', '0.0'))
    l2NormClip      = float(os.getenv('L2_NORM_CLIP', '1.0'))
    microbatches    = int(os.getenv('MICROBATCHES', str(batchSize)))

    if batchSize % microbatches != 0:
        raise ValueError(f"MICROBATCHES ({microbatches}) must perfectly divide batchSize ({batchSize}).")

    optimizerType = os.getenv('OPTIMIZER', 'sgd').lower()
    learningRate  = float(os.getenv('LEARNING_RATE', '0'))
    actual_lr     = learningRate or (0.001 if optimizerType == 'adam' else 0.01)

    # CascadedDP env vars
    cascadedDp    = os.getenv('CASCADED_DP', 'false').lower() == 'true'
    dpDropWindow  = int(os.getenv('DP_DROP_WINDOW', '5'))
    minDpEpochs   = int(os.getenv('MIN_DP_EPOCHS', '5'))
    slopeThresh   = float(os.getenv('DP_SLOPE_THRESHOLD', '0.015'))
    accPlatThresh = float(os.getenv('ACC_PLATEAU_THRESHOLD', '0.0005'))

    nodeId        = int(os.getenv('NODE_ID', '0'))
    numNodes      = int(os.getenv('NUM_NODES', '2'))
    partitionMode = os.getenv('PARTITION_MODE', 'iid')    # 'iid' | 'noniid_equal' | 'noniid_unequal'

    os.makedirs(scratchDir, exist_ok=True)

    print('***** Starting model =', modelName)
    print('-' * 64)

    # LOAD DATA
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

  
# ── PARTITION PER NODE ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)

    if partitionMode == 'noniid_equal':
        node_idx = []
        for c in range(10):
            idx   = np.where(y_train == c)[0]
            rng.shuffle(idx)
            split = int(len(idx) * 0.8)
            if c <= 4:
                node_idx.extend(idx[:split] if nodeId == 0 else idx[split:])
            else:
                node_idx.extend(idx[split:] if nodeId == 0 else idx[:split])
        node_idx = np.array(node_idx)
        rng.shuffle(node_idx)
        x_train = x_train[node_idx]
        y_train = y_train[node_idx]

    elif partitionMode == 'noniid_unequal':
        node_idx = []
        for c in range(10):
            idx   = np.where(y_train == c)[0]
            rng.shuffle(idx)
            split = int(len(idx) * 0.8)
            node_idx.extend(idx[:split] if nodeId == 0 else idx[split:])
        node_idx = np.array(node_idx)
        rng.shuffle(node_idx)
        x_train = x_train[node_idx]
        y_train = y_train[node_idx]

    else:
        # ── Dirichlet IID partition ──────────────────────────────────────────
        # DIRICHLET_ALPHA controls heterogeneity:
        #   'inf' or unset → true IID (global shuffle, equal split)
        #   1.0            → mild heterogeneity
        #   0.5            → moderate heterogeneity
        #   0.1            → strong heterogeneity (near non-IID)
        alpha_env = os.getenv('DIRICHLET_ALPHA', 'inf').lower()

        if alpha_env == 'inf':
            # True IID: global shuffle + contiguous equal split
            perm    = rng.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]

            split_size = len(x_train) // numNodes
            start      = nodeId * split_size
            end        = len(x_train) if nodeId == numNodes - 1 else start + split_size
            x_train    = x_train[start:end]
            y_train    = y_train[start:end]

            print(f'***** partition_mode=iid (true IID, alpha=inf) | node={nodeId}')

        else:
            alpha = float(alpha_env)

            node_idx = [[] for _ in range(numNodes)]
            for c in range(10):
                idx = np.where(y_train == c)[0]
                rng.shuffle(idx)

                proportions = rng.dirichlet(alpha=np.full(numNodes, alpha))
                splits      = (proportions * len(idx)).astype(int)
                splits[-1]  = len(idx) - splits[:-1].sum()   # fix rounding

                boundaries = np.concatenate([[0], np.cumsum(splits)])
                for n in range(numNodes):
                    node_idx[n].extend(idx[boundaries[n]:boundaries[n + 1]])

            # Trim to equal size across nodes
            min_size  = min(len(ix) for ix in node_idx)
            final_idx = np.array(node_idx[nodeId][:min_size])
            rng.shuffle(final_idx)
            x_train = x_train[final_idx]
            y_train = y_train[final_idx]

            print(f'***** partition_mode=iid (Dirichlet alpha={alpha}) | node={nodeId}')
    # NORMALIZE PIXELS
    x_train = x_train / 255.0
    x_test  = x_test / 255.0
    num_train_samples = len(x_train)

    # LOG CLASS DISTRIBUTION
    print(f'***** partition_mode={partitionMode} | node={nodeId} | samples={num_train_samples}')
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f'      Class {int(cls):2d}: {int(cnt):5d}')

    # ONE HOT ENCODE LABELS
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test, 10)

    # MODEL
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = get_optimizer(optimizerType, dpEnabled, learningRate or None, l2NormClip, noiseMultiplier, microbatches)
    loss = (
        tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        if dpEnabled
        else tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    )

    metrics = get_metrics()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # DATA PIPELINES
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(num_train_samples).batch(batchSize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize).prefetch(tf.data.AUTOTUNE)

    if partitionMode == "noniid_unequal":
        nodeWeightage = 80 if nodeId == 0 else 20
    else:
        nodeWeightage = 50
        
    largest_node_samples = int(60000 * 0.8)  # 48000
    steps_per_epoch = int(np.ceil(largest_node_samples / batchSize))  # 1500    

    # SWARM CALLBACK
    swarmCallback = SwarmCallback(
        syncFrequency=steps_per_epoch+1,
        minPeers=minPeers,
        adsValData=val_ds,
        adsValBatchSize=batchSize,
        mergeMethod='mean',
        nodeWeightage=nodeWeightage,
        totalEpochs=maxEpoch
    )

    callbacks = [swarmCallback]

    # ── Epoch barrier: keeps fast node locked to slow node's pace ──────────
    # Must be appended BEFORE CascadedDPCallback so the barrier fires first,
    # ensuring both nodes evaluate the same epoch's gradient stats together.
    if partitionMode == 'noniid_unequal':
        barrierCallback = EpochBarrierCallback(
            node_id=nodeId,
            num_nodes=numNodes,
            scratch_dir=scratchDir,
        )
        callbacks.append(barrierCallback)

    cascadedDpCallback = None
    if dpEnabled and cascadedDp:
        cascadedDpCallback = CascadedDPCallback(
            val_ds=val_ds,
            node_id=nodeId,
            num_nodes=numNodes,
            scratch_dir=scratchDir,
            optimizer_type=optimizerType,
            learning_rate=actual_lr,
            window_size=dpDropWindow,
            slope_threshold=slopeThresh,
            acc_plateau_threshold=accPlatThresh,
            min_dp_epochs=minDpEpochs
        )
        callbacks.append(cascadedDpCallback)

    print('Starting training ...')
    train_start = time.time()
    model.fit(train_ds, epochs=maxEpoch, validation_data=val_ds, callbacks=callbacks)
    train_end = time.time()
    training_time = round(train_end - train_start, 2)

    # PASSIVE DRIFT SNAPSHOT
    drift_snapshot = None
    trained_path = os.path.join(scratchDir, 'trained_weights.pkl')
    merged_path  = os.path.join(scratchDir, 'merged_weights.pkl')
    if os.path.exists(trained_path) and os.path.exists(merged_path):
        try:
            with open(trained_path, 'rb') as f:
                trained_w = pickle.load(f)
            with open(merged_path, 'rb') as f:
                merged_w  = pickle.load(f)
            if isinstance(trained_w, dict):
                trained_w = list(trained_w.values())
                merged_w  = list(merged_w.values())
            diff_sq = sum(
                np.sum((np.asarray(tw) - np.asarray(mw)) ** 2)
                for tw, mw in zip(trained_w, merged_w)
            )
            drift_snapshot = float(np.sqrt(diff_sq))
            print(f'***** Drift snapshot ||trained - merged||_2 = {drift_snapshot:.6f}')
        except Exception as e:
            print(f'***** Drift snapshot: could not compute ({e})')

    # ─────────────────────────────────────────────────────────────────────────
    # COMPREHENSIVE FINAL TEST EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    print('\nRunning final post-training evaluation on test dataset...')

    # 1. Extract exact test loss and test accuracy using direct evaluate sweep
    eval_results = model.evaluate(val_ds, verbose=0)
    final_test_loss     = float(eval_results[0])
    final_test_accuracy = float(eval_results[1])

    # 2. Extract final global validation/test F1-Score
    y_true_list = []
    y_pred_list = []
    for x_b, y_b in val_ds:
        preds = model.predict_on_batch(x_b)
        y_true_list.append(np.argmax(y_b.numpy(), axis=1))
        y_pred_list.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    final_test_f1 = float(f1_score(y_true, y_pred, average='macro'))

    # PRIVACY REPORT
    eps = None
    if dpEnabled and noiseMultiplier > 0:
        print('-' * 64)
        print('***** PRIVACY REPORT *****')
        delta     = 1.0 / num_train_samples
        dp_epochs = cascadedDpCallback.dp_drop_epoch if (cascadedDpCallback and cascadedDpCallback.dp_drop_epoch) else maxEpoch

        eps, _ = compute_dp_sgd_privacy(
            n=num_train_samples, batch_size=batchSize, noise_multiplier=noiseMultiplier, epochs=dp_epochs, delta=delta
        )
        print(f"Final Epsilon (ε): {eps:.4f} | Final Delta (δ): {delta:.2e}")
        print('-' * 64)

    # SAVE COMPREHENSIVE RESULTS
    results = {
        "config": {
            "model_name": modelName,
            "node_id": nodeId,
            "num_nodes": numNodes,
            "epochs": maxEpoch,
            "batch_size": batchSize,
            "optimizer": optimizerType,
            "learning_rate": actual_lr,
            "dp_enabled": dpEnabled,
            "cascaded_dp": cascadedDp,
            "l2_norm_clip": l2NormClip,
            "noise_multiplier": noiseMultiplier,
            "microbatches": microbatches,
            "partition_mode": partitionMode,
            "num_train_samples": num_train_samples
        },
        "performance": {
            "training_time_seconds": training_time,
            "final_test_loss": final_test_loss,
            "final_test_accuracy": final_test_accuracy,
            "final_test_f1_macro": final_test_f1
        },
        "privacy": {
            "epsilon": round(eps, 4) if eps is not None else None,
            "delta": 1.0 / num_train_samples if dpEnabled else None,
            "dp_drop_epoch": cascadedDpCallback.dp_drop_epoch if cascadedDpCallback else None,
            "dp_slope_threshold": slopeThresh,
            "accuracy_plateau_threshold": accPlatThresh,
            "dp_drop_reason": (
                cascadedDpCallback.dp_drop_reason
                if cascadedDpCallback
                else None
            )
        }
    }

    result_file  = os.getenv("RESULT_FILE", "results.json")
    results_path = os.path.join("/results", result_file)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(scratchDir, modelName))
    print('Saved the trained model and verified final test metrics JSON!')

if __name__ == '__main__':
    main()
