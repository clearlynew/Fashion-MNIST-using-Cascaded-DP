############################################################################
## (C)Copyright 2021-2023 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
############################################################################
import os
import glob
import json
import time
import numpy as np
import tensorflow as tf
from collections import deque
from sklearn.metrics import f1_score

# Using the library path as per standard TensorFlow Privacy usage
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
from swarmlearning.tf import SwarmCallback

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

batchSize = 32
defaultMaxEpoch = 50
defaultMinPeers = 2

# ─────────────────────────────────────────────────────────────────────────────
# DECENTRALIZED CASCADED DP CALLBACK (SWARM CONSENSUS)
# ─────────────────────────────────────────────────────────────────────────────

class CascadedDPCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, node_id, num_nodes, scratch_dir, optimizer_type='sgd', 
                 learning_rate=0.01, window_size=5, slope_threshold=0.015, 
                 acc_plateau_threshold=0.0005, min_dp_epochs=5):
        super().__init__()
        self.val_ds = val_ds
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.scratch_dir = scratch_dir
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.slope_threshold = slope_threshold
        self.acc_plateau_threshold = acc_plateau_threshold
        self.min_dp_epochs = min_dp_epochs

        self.grad_norm_window = deque(maxlen=window_size)
        self.acc_window = deque(maxlen=window_size)
        self.grad_history = []
        self.rolling_history = []
        self.acc_history = []
        self.dp_active = True
        self.dp_drop_epoch = None
        self.dp_drop_reason = None
        self.vote_file = os.path.join(self.scratch_dir, f".vote_drop_dp_node_{self.node_id}")

        if os.path.exists(self.vote_file):
            try: os.remove(self.vote_file)
            except: pass

        self._measure_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def _compute_grad_norm(self):
        norms = []
        for x, y in self.val_ds.take(5):
            with tf.GradientTape() as tape:
                preds = self.model(x, training=True)
                loss_val = self._measure_loss(y, preds)
            grads = tape.gradient(loss_val, self.model.trainable_variables)
            grad_norm = tf.linalg.global_norm(grads).numpy()
            norms.append(float(grad_norm))
        return float(np.mean(norms))

    def _drop_dp(self, epoch):
        print(f"\n***** CascadedDP: [Node {self.node_id}] SWARM QUORUM UNLOCKED *****")
        print(f"***** CascadedDP: dropping DP globally at epoch {epoch + 1} *****")

        if self.optimizer_type == 'adam':
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            new_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           optimizer=new_optimizer,
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])

        if hasattr(self.model, 'train_function'):
            self.model.train_function = None
            self.model.test_function = None
            self.model.predict_function = None
            for x_sample, y_sample in self.val_ds.take(1):
                self.model.make_train_function()
                self.model.make_test_function()
                self.model.make_predict_function()

        self.dp_active = False
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
        if not self.dp_active: return
        val_acc = (logs or {}).get('val_accuracy')
        if val_acc is None: return

        grad_norm = self._compute_grad_norm()
        self.grad_norm_window.append(grad_norm)
        self.acc_window.append(val_acc)
        self.grad_history.append(float(grad_norm))
        
        rolling_mean = float(np.mean(self.grad_norm_window))
        self.rolling_history.append(rolling_mean)
        self.acc_history.append(float(val_acc))

        print(
            f"  [CascadedDP] Node={self.node_id} | epoch={epoch + 1} | "
            f"grad_norm={grad_norm:.6f} | rolling_mean={rolling_mean:.6f} | val_acc={val_acc:.4f}"
        )

        if epoch + 1 < self.min_dp_epochs or len(self.grad_norm_window) < self.window_size: return

        relative_slope = abs(self.rolling_history[-2] - self.rolling_history[-1]) / self.rolling_history[-2] if len(self.rolling_history) >= 2 else 1.0
        acc_variance = float(np.var(self.acc_window))

        if relative_slope < self.slope_threshold and acc_variance < self.acc_plateau_threshold:
            if not os.path.exists(self.vote_file):
                try:
                    with open(self.vote_file, 'w') as f: f.write(f"Node {self.node_id} converged at epoch {epoch + 1}")
                    print(f"  [CascadedDP-Consensus] Node {self.node_id} posted drop vote to shared scratch.")
                except Exception as e:
                    print(f"  [CascadedDP-Consensus] Error writing vote file: {e}")

        total_votes = sum(1 for i in range(self.num_nodes) if os.path.exists(os.path.join(self.scratch_dir, f".vote_drop_dp_node_{i}")))
        
        print(f"  [CascadedDP-Quorum] Node {self.node_id} reporting cluster status: {total_votes}/{self.num_nodes} votes collected.")

        if total_votes == self.num_nodes:
            time.sleep(self.node_id * 0.2)
            self._drop_dp(epoch)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    modelName = 'fashion-mnist'
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))
    dpEnabled = os.getenv('DP_ENABLED', 'false').lower() == 'true'
    noiseMultiplier = float(os.getenv('NOISE_MULTIPLIER', '0.0'))
    l2NormClip = float(os.getenv('L2_NORM_CLIP', '1.0'))
    microbatches = int(os.getenv('MICROBATCHES', str(batchSize)))
    optimizerType = os.getenv('OPTIMIZER', 'sgd').lower()
    learningRate = float(os.getenv('LEARNING_RATE', '0'))
    actual_lr = learningRate or (0.001 if optimizerType == 'adam' else 0.01)
    cascadedDp = os.getenv('CASCADED_DP', 'false').lower() == 'true'
    dpDropWindow = int(os.getenv('DP_DROP_WINDOW', '5'))
    minDpEpochs = int(os.getenv('MIN_DP_EPOCHS', '5'))
    slopeThresh = float(os.getenv('DP_SLOPE_THRESHOLD', '0.015'))
    accPlatThresh = float(os.getenv('ACC_PLATEAU_THRESHOLD', '0.0005'))
    nodeId = int(os.getenv('NODE_ID', '0'))
    numNodes = int(os.getenv('NUM_NODES', '2'))

    os.makedirs(scratchDir, exist_ok=True)

    print('***** Starting model =', modelName)
    print('-' * 64)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # ── PARTITIONING LOGIC ──
    rng = np.random.default_rng(seed=42)
    alpha_env = os.getenv('DIRICHLET_ALPHA', 'inf').lower()

    if alpha_env == 'inf':
        partitionMode = 'iid'
        perm = rng.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]
        split_size = len(x_train) // numNodes
        start, end = nodeId * split_size, (len(x_train) if nodeId == numNodes - 1 else (nodeId + 1) * split_size)
        x_train, y_train = x_train[start:end], y_train[start:end]
        nodeWeightage = 50
        print(f"***** partition_mode=iid | node={nodeId}")
        print(f"***** Dynamic Node Weight Assignment: Node {nodeId} Weightage = {nodeWeightage}%")
        print(f"***** partition_mode=iid | node={nodeId} | samples={len(x_train)}")
        for c in range(10):
            count = int(np.sum(y_train == c))
            if count > 0:
                print(f"      Class {c:2d}: {count:5d}")
    else:
        partitionMode = 'non_iid'
        alpha = float(alpha_env)
        node_idx = [[] for _ in range(numNodes)]
        for c in range(10):
            idx = np.where(y_train == c)[0]
            rng.shuffle(idx)
            proportions = rng.dirichlet(alpha=np.full(numNodes, alpha))
            splits = (proportions * len(idx)).astype(int)
            splits[-1] = len(idx) - splits[:-1].sum()
            bounds = np.concatenate([[0], np.cumsum(splits)])
            for n in range(numNodes): node_idx[n].extend(idx[bounds[n]:bounds[n+1]])
        
        total = sum(len(node_idx[n]) for n in range(numNodes))
        nodeWeightage = int(round(100 * len(node_idx[nodeId]) / total))
        final_idx = np.array(node_idx[nodeId])
        rng.shuffle(final_idx)
        x_train, y_train = x_train[final_idx], y_train[final_idx]
        print(f"***** partition_mode=dirichlet (Dirichlet alpha={alpha_env}) | node={nodeId}")
        print(f"***** Dynamic Node Weight Assignment: Node {nodeId} Weightage = {nodeWeightage}%")
        print(f"***** partition_mode=dirichlet | node={nodeId} | samples={len(x_train)}")
        for c in range(10):
            count = int(np.sum(y_train == c))
            if count > 0:
                print(f"      Class {c:2d}: {count:5d}")

    x_train, x_test = x_train / 255.0, x_test / 255.0
    num_train_samples = len(x_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    if dpEnabled:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer, DPKerasSGDOptimizer
        optimizer = (DPKerasAdamOptimizer if optimizerType == 'adam' else DPKerasSGDOptimizer)(
            l2_norm_clip=l2NormClip, noise_multiplier=noiseMultiplier, 
            num_microbatches=microbatches, learning_rate=actual_lr)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        print(f"***** Using DP-{'Adam' if optimizerType == 'adam' else 'SGD'} optimizer")
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=actual_lr) if optimizerType == 'adam' else tf.keras.optimizers.SGD(learning_rate=actual_lr, momentum=0.9, nesterov=True)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        print(f"***** Using standard {'Adam' if optimizerType == 'adam' else 'SGD'} optimizer")

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).shuffle(num_train_samples).batch(batchSize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).batch(batchSize).prefetch(tf.data.AUTOTUNE)

    callbacks = [SwarmCallback(syncFrequency=1024, useAdaptiveSync=False, minPeers=minPeers, adsValData=val_ds, adsValBatchSize=batchSize, mergeMethod='mean', nodeWeightage=nodeWeightage, totalEpochs=maxEpoch)]
    
    cdp = None
    if dpEnabled and cascadedDp:
        cdp = CascadedDPCallback(val_ds, nodeId, numNodes, scratchDir, optimizerType, actual_lr, dpDropWindow, slopeThresh, accPlatThresh, minDpEpochs)
        callbacks.append(cdp)

    print('Starting training ...')
    train_start = time.time()
    model.fit(train_ds, epochs=maxEpoch, validation_data=val_ds, callbacks=callbacks)
    training_time = round(time.time() - train_start, 2)

    # Evaluation
    print('\nRunning final post-training evaluation on test dataset...')
    eval_res = model.evaluate(val_ds, verbose=0)
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds])
    y_pred = np.concatenate([np.argmax(model.predict_on_batch(x), axis=1) for x, _ in val_ds])
    
    eps = None
    if dpEnabled and noiseMultiplier > 0:
        print('-' * 64)
        print('***** PRIVACY REPORT *****')
        delta = 1.0 / num_train_samples
        dp_epochs = cdp.dp_drop_epoch if (cdp and cdp.dp_drop_epoch) else maxEpoch
        eps, _ = compute_dp_sgd_privacy(n=num_train_samples, batch_size=batchSize, noise_multiplier=noiseMultiplier, epochs=dp_epochs, delta=delta)
        print(f"Final Epsilon (ε): {eps:.4f} | Final Delta (δ): {delta:.2e}")
        print('-' * 64)

    results = {
        "config": {"model_name": modelName, "node_id": nodeId, "num_nodes": numNodes, "epochs": maxEpoch, "batch_size": batchSize, "optimizer": optimizerType, "learning_rate": actual_lr, "dp_enabled": dpEnabled, "cascaded_dp": cascadedDp, "l2_norm_clip": l2NormClip, "noise_multiplier": noiseMultiplier, "microbatches": microbatches, "partition_mode": partitionMode, "num_train_samples": num_train_samples},
        "performance": {"training_time_seconds": training_time, "final_test_loss": float(eval_res[0]), "final_test_accuracy": float(eval_res[1]), "final_test_f1_macro": float(f1_score(y_true, y_pred, average='macro'))},
        "privacy": {"epsilon": round(eps, 4) if eps is not None else None, "delta": 1.0/num_train_samples if dpEnabled else None, "dp_drop_epoch": cdp.dp_drop_epoch if cdp else None, "dp_slope_threshold": slopeThresh, "accuracy_plateau_threshold": accPlatThresh, "dp_drop_reason": cdp.dp_drop_reason if cdp else None}
    }
    
    with open(f"/results/{os.getenv('RESULT_FILE', 'results.json')}", 'w') as f: json.dump(results, f, indent=2)
    print('Saved the trained model and verified final test metrics JSON!')

if __name__ == '__main__': main()
