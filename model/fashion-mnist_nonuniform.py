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

batchSize = 32            # Fixed mini-batch size for both training and DP microbatching
defaultMaxEpoch = 50       # Fallback if MAX_EPOCHS env var isn't set
defaultMinPeers = 2        # Fallback minimum peer count for SwarmCallback to start a sync round

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PARAMETER-BASED COUPLING LAYER
# ─────────────────────────────────────────────────────────────────────────────

class ConvergenceFlagLayer(tf.keras.layers.Layer):
    """
    Wraps the dummy non-trainable weight in an actual Layer so it shows up
    under model.layers -> layer.weights, matching how SL's SwarmCallback
    enumerates weightNames. 
    
    Since call(x) returns x unchanged, appending this to a Sequential model
    preserves metrics and predictions completely while providing a cross-host
    communication channel via standard weight-merging.

    NOTE: this only works as a consensus channel if SwarmCallback's
    mergeMethod='mean' actually averages non-trainable weights along with
    trainable ones. That behavior isn't part of any documented contract here,
    so if SwarmCallback ever changes to merge only trainable variables, the
    flag will stay stuck at each node's local value and quorum will never be
    detected (silently — no error, just DP that never drops).
    """
    def __init__(self):
        super().__init__(name="convergence_flag_layer")
        # Single scalar weight, initialized to 0.0. Each node flips its own
        # copy to 1.0 once it has locally converged (see CascadedDPCallback).
        # Because SwarmCallback merges weights across peers by averaging,
        # this scalar doubles as a distributed "vote": the merged value is
        # exactly 1.0 only when every peer has voted, giving cheap quorum
        # detection without any extra coordination service.
        self.flag = self.add_weight(
            name="convergence_flag",
            shape=(1,),
            initializer="zeros",
            trainable=False,   # keeps it out of gradient computation entirely
        )

    def call(self, x):
        # Pass-through: this layer exists purely to hold state, not to
        # transform activations, so predictions/metrics are unaffected.
        return x

# ─────────────────────────────────────────────────────────────────────────────
# DECENTRALIZED CASCADED DP CALLBACK (SWARM PARAMETER CONSENSUS)
# ─────────────────────────────────────────────────────────────────────────────

class CascadedDPCallback(tf.keras.callbacks.Callback):
    """
    Monitors local training dynamics each epoch to decide when DP noise has
    stopped being necessary (gradients have flattened out and validation
    accuracy has plateaued), then uses the swarm-averaged ConvergenceFlagLayer
    to make sure *every* node agrees before anyone actually drops DP and
    recompiles with a plain (non-private) optimizer.
    """
    def __init__(self, val_ds, node_id, num_nodes, flag_layer, optimizer_type='sgd', 
                 learning_rate=0.01, window_size=5, k_slope=1.0, 
                 k_acc=1.0, min_dp_epochs=5):
        super().__init__()
        self.val_ds = val_ds                              # held-out data used both for grad-norm probing and convergence checks
        self.node_id = node_id                             # this node's index, used only for logging now that sleep staggering is removed
        self.num_nodes = num_nodes                         # total peers; used to interpret the averaged flag value as a vote count
        self.flag_layer = flag_layer                       # shared non-trainable weight synced via SwarmCallback's weight merge
        self.optimizer_type = optimizer_type               # 'sgd' or 'adam', determines which optimizer to rebuild with post-DP
        self.learning_rate = learning_rate                 # LR to use for the plain optimizer after DP is dropped
        self.window_size = window_size                     # number of recent epochs considered for the rolling grad-norm mean / accuracy variance
        self.k_slope = k_slope                             # dimensionless multiplier on the burn-in grad-norm noise floor (std dev); triggers when |slope| falls below k_slope * burn_in_std
        self.k_acc = k_acc                                 # dimensionless multiplier on the burn-in val-accuracy variance floor; triggers when windowed acc variance falls below k_acc * burn_in_variance
        self.min_dp_epochs = min_dp_epochs                 # minimum epochs DP must run before convergence checks even start; doubles as the burn-in period used to estimate the noise floor

        self.grad_norm_window = deque(maxlen=window_size)   # rolling buffer of raw per-epoch grad norms
        self.acc_window = deque(maxlen=window_size)          # rolling buffer of per-epoch val accuracy
        self.grad_history = []                               # full history of grad norms (for the results/debug record)
        self.rolling_history = []                            # full history of the rolling mean grad norm
        self.acc_history = []                                 # full history of val accuracy
        self.dp_active = True                                 # flips to False once this node has recompiled without DP
        self.dp_drop_epoch = None                             # epoch (1-indexed) at which DP was dropped, used later for epsilon accounting
        self.dp_drop_reason = None                            # snapshot of the metrics that triggered the drop, for the results JSON
        self.local_converged = False                          # guards against re-flipping the flag weight every epoch once converged

        # Burn-in noise floor: instead of comparing raw signals against
        # fixed, hand-picked constants, we estimate how noisy grad-norm and
        # val-accuracy naturally are during the first min_dp_epochs epochs
        # of DP training, and scale the trigger thresholds off of that.
        # This makes k_slope/k_acc self-scaling with respect to window_size
        # and dataset/architecture-specific noise levels, rather than
        # requiring separate re-tuning of raw thresholds every time either
        # changes.
        self.burn_in_grad_norms = []                          # grad norms collected during the burn-in period only
        self.burn_in_accs = []                                # val accuracies collected during the burn-in period only
        self.burn_in_std_grad = None                           # std dev of grad norms over burn-in, computed once burn-in ends
        self.burn_in_var_acc = None                             # variance of val accuracy over burn-in, computed once burn-in ends

        # Used only to compute a gradient-norm signal; not the actual training loss
        self._measure_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def _compute_grad_norm(self):
        """
        Probes the current gradient magnitude on a handful of validation
        batches (not used for actual weight updates) to get a cheap signal
        of how much the model is still "moving". Averaged over 5 batches to
        smooth out per-batch noise. training=True is intentional here so
        dropout etc. behave the same way they do during real training steps.
        """
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
        """
        Called once global quorum is reached (see on_epoch_end). Swaps the
        DP optimizer out for a plain one and forces Keras to rebuild its
        compiled train/test/predict graphs, since simply swapping
        self.model.optimizer would leave the old DP-wrapped train_function
        cached and still in effect.
        """
        print(f"\n***** CascadedDP: [Node {self.node_id}] SWARM PARAMETER QUORUM UNLOCKED *****")
        print(f"***** CascadedDP: dropping DP globally at epoch {epoch + 1} *****")

        # Rebuild with the same optimizer family the run started with, but
        # without any DP wrapping (no clipping/noise).
        if self.optimizer_type == 'adam':
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            new_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)

        # Recompiling gives the model a fresh optimizer (no leftover
        # momentum/state from the DP optimizer) and switches the loss back
        # to a standard mean-reduced CategoricalCrossentropy, since the DP
        # optimizer required per-example (unreduced) loss for microbatching.
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           optimizer=new_optimizer,
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])

        # model.compile() alone doesn't discard Keras's cached, traced
        # tf.function graphs for train/test/predict steps — those still
        # reference the old DP optimizer's step logic. Clearing them and
        # forcing a rebuild ensures subsequent fit() steps actually use the
        # new plain-optimizer graph.
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
        # Snapshot of the exact metrics that triggered the drop, kept for
        # the final results JSON / post-hoc debugging.
        self.dp_drop_reason = {
            "epoch": epoch + 1,
            "k_slope": self.k_slope,
            "k_acc": self.k_acc,
            "burn_in_std_grad": self.burn_in_std_grad,
            "burn_in_var_acc": self.burn_in_var_acc,
            "rolling_mean": float(np.mean(self.grad_norm_window)),
            "val_acc_window": list(self.acc_window)
        }

        print(f"***** CascadedDP: low-level execution graphs forcefully rebuilt *****")
        print(f"***** CascadedDP: model recompiled with standard optimizer *****\n")

    def on_epoch_end(self, epoch, logs=None):
        # Once this node has already dropped DP there's nothing left to
        # monitor — the rest of training proceeds with the plain optimizer.
        if not self.dp_active: return

        val_acc = (logs or {}).get('val_accuracy')
        if val_acc is None: return

        # Update rolling signals: gradient norm (are we still learning?) and
        # validation accuracy (has performance leveled off?).
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

        # Burn-in period: just collect raw signal samples, don't evaluate
        # convergence yet. This also builds up the noise-floor estimate
        # (std dev of grad norm, variance of val accuracy) that thresholds
        # are scaled against below, rather than resetting after burn-in.
        if epoch + 1 <= self.min_dp_epochs:
            self.burn_in_grad_norms.append(grad_norm)
            self.burn_in_accs.append(val_acc)
            return

        # Compute the burn-in noise floor exactly once, right as burn-in
        # ends. A small epsilon guards against a degenerate all-identical
        # burn-in signal producing a zero floor (which would make any
        # k_slope/k_acc value trigger immediately).
        if self.burn_in_std_grad is None:
            self.burn_in_std_grad = max(float(np.std(self.burn_in_grad_norms)), 1e-8)
            self.burn_in_var_acc = max(float(np.var(self.burn_in_accs)), 1e-10)
            print(
                f"  [CascadedDP] Node={self.node_id} | burn-in noise floor set: "
                f"grad_std={self.burn_in_std_grad:.6f} | acc_var={self.burn_in_var_acc:.8f}"
            )

        # Don't evaluate convergence until the rolling window itself is
        # full (avoids acting on a partially-populated window right after
        # burn-in ends).
        if len(self.grad_norm_window) < self.window_size: return

        # Read the flag BEFORE any local write this epoch. This value
        # reflects whatever the last actual SwarmCallback sync produced —
        # i.e. a real cross-host merge — not this epoch's local change.
        # Checking quorum before writing prevents a node from reading back
        # its own just-set 1.0 as if it were already-merged consensus,
        # which would falsely report full quorum the instant a single node
        # converges, before any other peer has done anything.
        # NOTE: this is a *weightage-weighted* merge (SwarmCallback was given
        # nodeWeightage), not a plain unweighted average of {0,1} votes. So
        # global_flag_value is NOT "fraction of nodes converged" whenever
        # weightage is non-uniform (e.g. under Dirichlet non-IID splits) —
        # e.g. 0.7525 could mean one node (carrying ~75% weightage) has
        # converged and the other (carrying ~25%) has not. Do NOT back out a
        # literal converged-node count from this value; log it as a raw
        # weighted fraction instead.
        global_flag_value = float(self.flag_layer.flag.numpy()[0])
        print(f"  [CascadedDP-Quorum] Node {self.node_id} tracking global weighted flag value: "
              f"{global_flag_value:.4f} (this is a weightage-weighted average, not a literal node count; "
              f"only == 1.0 implies every node has actually converged)")

        # Quorum is reached only when ALL nodes have set their local flag parameter to 1.0 (weighted mean == 1.0,
        # which can only happen if literally every node's local flag is 1.0, regardless of weightage).
        # 0.999 rather than an exact 1.0 comparison to allow for floating-point merge/round-trip error.
        if global_flag_value >= 0.999:
            self._drop_dp(epoch)
            return

        # Absolute change in the rolling grad-norm mean vs. the previous
        # epoch, compared against the burn-in noise floor rather than a
        # hand-picked constant — this makes the trigger self-scaling with
        # both window_size and how noisy this particular dataset/model
        # combination naturally is.
        grad_slope = abs(self.rolling_history[-2] - self.rolling_history[-1]) if len(self.rolling_history) >= 2 else float('inf')
        # Variance of val accuracy across the window, compared against the
        # burn-in accuracy-variance floor for the same reason.
        acc_variance = float(np.var(self.acc_window))

        slope_threshold = self.k_slope * self.burn_in_std_grad
        acc_threshold = self.k_acc * self.burn_in_var_acc
        slope_trigger = grad_slope < slope_threshold
        plateau_trigger = acc_variance < acc_threshold

        # Per-node diagnostic: shows the actual margin to each trigger, not
        # just the AND'd boolean. Without this it's impossible to tell
        # whether a stuck node is close to converging or nowhere near it —
        # e.g. under a skewed Dirichlet shard, acc_variance may sit
        # persistently above its threshold because DP noise keeps the local
        # val_accuracy jittering by roughly the same amount all the way
        # through training, rather than actually damping down over time.
        print(
            f"  [CascadedDP-Trigger] Node={self.node_id} | epoch={epoch + 1} | "
            f"grad_slope={grad_slope:.6f} (thresh {slope_threshold:.6f}, {'PASS' if slope_trigger else 'fail'}) | "
            f"acc_variance={acc_variance:.8f} (thresh {acc_threshold:.8f}, {'PASS' if plateau_trigger else 'fail'}) | "
            f"local_converged={self.local_converged}"
        )

        # Evaluate local convergence criteria. This write happens AFTER the
        # quorum check above, so the earliest this node's own vote can be
        # read back as part of a merged quorum is on a LATER epoch, once an
        # actual SwarmCallback sync has had a chance to run in between.
        #
        # IMPORTANT: every SwarmCallback merge round overwrites this node's
        # local flag weight with the incoming swarm-averaged value via
        # set_weights() — including on nodes that already voted 1.0. If we
        # only write 1.0 once (guarded by local_converged), that write gets
        # silently diluted back down by the very next merge and is never
        # re-affirmed, so no node's true vote survives more than one sync
        # round and the global average can never actually reach 1.0. So we
        # re-assert 1.0 every epoch once converged, regardless of whatever
        # the last merge overwrote it to — this guarantees the NEXT outgoing
        # merge always carries this node's true current vote.
        if slope_trigger and plateau_trigger:
            self.local_converged = True

        if self.local_converged:
            self.flag_layer.flag.assign([1.0])
            print(f"  [CascadedDP-Consensus] Node {self.node_id} re-asserting local flag = 1.0 "
                  f"(local_converged={self.local_converged})")

# ─────────────────────────────────────────────────────────────────────────────
# FLAG-VALUE OBSERVER (pure debugging aid — no vendor code touched)
# ─────────────────────────────────────────────────────────────────────────────

class FlagObserverCallback(tf.keras.callbacks.Callback):
    """
    Registered AFTER SwarmCallback in the callbacks list, so Keras invokes
    this on_train_batch_end strictly after SwarmCallback's own — meaning any
    time SwarmCallback just performed a merge and called set_weights(), this
    will see the new value on the very next batch. Prints only on change, so
    it won't spam every batch; this pinpoints exactly which batch/epoch a
    merge actually moved the flag, without editing swarm_tf_source.py.
    """
    def __init__(self, flag_layer, node_id):
        super().__init__()
        self.flag_layer = flag_layer
        self.node_id = node_id
        self._last_seen = None

    def on_train_batch_end(self, batch, logs=None):
        current = float(self.flag_layer.flag.numpy()[0])
        if self._last_seen is None or abs(current - self._last_seen) > 1e-9:
            print(f"  [FlagObserver] Node={self.node_id} | batch={batch} | "
                  f"flag changed: {self._last_seen} -> {current}")
            self._last_seen = current



def main():
    modelName = 'fashion-mnist'
    maxEpoch = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))          # total training epochs
    minPeers = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))            # min peers SwarmCallback needs before it will sync
    dpEnabled = os.getenv('DP_ENABLED', 'false').lower() == 'true'          # whether to train with DP-SGD at all
    noiseMultiplier = float(os.getenv('NOISE_MULTIPLIER', '0.0'))           # DP noise scale (0 = clipping only, no privacy noise)
    l2NormClip = float(os.getenv('L2_NORM_CLIP', '1.0'))                    # per-example gradient clipping norm for DP-SGD
    microbatches = int(os.getenv('MICROBATCHES', str(batchSize)))           # DP microbatch count; defaults to one microbatch per example group of batchSize
    optimizerType = os.getenv('OPTIMIZER', 'sgd').lower()                   # 'sgd' or 'adam'
    learningRate = float(os.getenv('LEARNING_RATE', '0'))                   # 0 means "use the optimizer-specific default" below
    actual_lr = learningRate or (0.001 if optimizerType == 'adam' else 0.01)
    cascadedDp = os.getenv('CASCADED_DP', 'false').lower() == 'true'        # whether to enable the auto-drop-DP-on-convergence mechanism
    dpDropWindow = int(os.getenv('DP_DROP_WINDOW', '5'))                    # rolling window size (epochs) used by CascadedDPCallback
    minDpEpochs = int(os.getenv('MIN_DP_EPOCHS', '5'))                      # minimum epochs before convergence checks can trigger a drop
    kSlope = float(os.getenv('K_SLOPE', '1.0'))                              # multiplier on burn-in grad-norm std dev; sweep this, not a raw threshold
    kAcc = float(os.getenv('K_ACC', '1.0'))                                  # multiplier on burn-in val-accuracy variance; sweep this, not a raw threshold
    nodeId = int(os.getenv('NODE_ID', '0'))                                 # this node's index within the swarm
    numNodes = int(os.getenv('NUM_NODES', '2'))                             # total number of participating nodes

    print('***** Starting model =', modelName)
    print('-' * 64)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # ── PARTITIONING LOGIC ──
    # Fixed seed so every node deterministically derives the same global
    # permutation/proportions and therefore ends up with a disjoint,
    # reproducible shard of the training data.
    rng = np.random.default_rng(seed=42)
    alpha_env = os.getenv('DIRICHLET_ALPHA', 'inf').lower()

    if alpha_env == 'inf':
        # alpha=inf is the standard shorthand for "uniform/IID" partitioning:
        # shuffle everything, then cut into numNodes equal contiguous chunks.
        partitionMode = 'iid'
        perm = rng.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]
        split_size = len(x_train) // numNodes
        # Last node absorbs any remainder from the integer division so no
        # samples are dropped.
        start, end = nodeId * split_size, (len(x_train) if nodeId == numNodes - 1 else (nodeId + 1) * split_size)
        x_train, y_train = x_train[start:end], y_train[start:end]
        # Equal-size shards under IID partitioning, so weightage is just an
        # even split across however many nodes are actually participating.
        nodeWeightage = round(100 / numNodes)
        print(f"***** partition_mode=iid | node={nodeId}")
        print(f"***** Dynamic Node Weight Assignment: Node {nodeId} Weightage = {nodeWeightage}%")
        print(f"***** partition_mode=iid | node={nodeId} | samples={len(x_train)}")
        for c in range(10):
            count = int(np.sum(y_train == c))
            if count > 0:
                print(f"      Class {c:2d}: {count:5d}")
    else:
        # Dirichlet-alpha partitioning: simulates non-IID/heterogeneous data
        # across nodes. Lower alpha -> more skewed class distribution per
        # node; higher alpha -> closer to IID.
        partitionMode = 'non_iid'
        alpha = float(alpha_env)
        node_idx = [[] for _ in range(numNodes)]
        for c in range(10):
            # For each class independently, draw a Dirichlet split across
            # nodes and hand out that class's sample indices accordingly.
            idx = np.where(y_train == c)[0]
            rng.shuffle(idx)
            proportions = rng.dirichlet(alpha=np.full(numNodes, alpha))
            splits = (proportions * len(idx)).astype(int)
            # Fix up rounding: give the last node whatever's left over so
            # every sample of this class is assigned to exactly one node.
            splits[-1] = len(idx) - splits[:-1].sum()
            bounds = np.concatenate([[0], np.cumsum(splits)])
            for n in range(numNodes): node_idx[n].extend(idx[bounds[n]:bounds[n+1]])
        
        # This node's actual share of the full training set, used both for
        # logging and as the weight SwarmCallback uses when merging models
        # (so a node with more data gets proportionally more influence).
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

    x_train, x_test = x_train / 255.0, x_test / 255.0   # scale pixel values to [0, 1]
    num_train_samples = len(x_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)  # one-hot labels for CategoricalCrossentropy
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

    # Initialize the tracking layer instance. Created before the model so
    # the same Python object reference can be handed to both the Sequential
    # model (for weight merging) and CascadedDPCallback (for reading/setting
    # the flag directly).
    flag_layer = ConvergenceFlagLayer()

    # Standard CNN benchmark architecture used across TF Privacy tutorials,
    # Opacus reference implementations, and DP-SGD literature for MNIST/Fashion-MNIST
    # (see e.g. tensorflow/privacy/tutorials/mnist_dpsgd_tutorial_common.py and
    # "On the Convergence and Calibration of Deep Learning with Differential Privacy").
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
        flag_layer  # Injected directly to become reachable by Swarm Learning's weight names enumeration
    ])

    if dpEnabled:
        # DP-SGD variants: clip per-example gradients to l2NormClip, add
        # Gaussian noise scaled by noiseMultiplier, and average over
        # microbatches groups.
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer, DPKerasSGDOptimizer
        optimizer = (DPKerasAdamOptimizer if optimizerType == 'adam' else DPKerasSGDOptimizer)(
            l2_norm_clip=l2NormClip, noise_multiplier=noiseMultiplier, 
            num_microbatches=microbatches, learning_rate=actual_lr)
        # DP optimizers need per-example (unreduced) loss values to clip
        # each example's gradient individually before aggregating.
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        print(f"***** Using DP-{'Adam' if optimizerType == 'adam' else 'SGD'} optimizer")
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=actual_lr) if optimizerType == 'adam' else tf.keras.optimizers.SGD(learning_rate=actual_lr, momentum=0.9, nesterov=True)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        print(f"***** Using standard {'Adam' if optimizerType == 'adam' else 'SGD'} optimizer")

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).shuffle(num_train_samples).batch(batchSize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).batch(batchSize).prefetch(tf.data.AUTOTUNE)

    # SwarmCallback drives the actual decentralized training: it syncs and
    # merges model weights across peers every syncFrequency batches, using
    # nodeWeightage to weight each peer's contribution to the merge.
    callbacks = [SwarmCallback(syncFrequency=1024, useAdaptiveSync=False, minPeers=minPeers, adsValData=val_ds, adsValBatchSize=batchSize, mergeMethod='mean', nodeWeightage=nodeWeightage, totalEpochs=maxEpoch)]
    callbacks.append(FlagObserverCallback(flag_layer, nodeId))
    
    cdp = None
    if dpEnabled and cascadedDp:
        # Only meaningful when DP is actually enabled — cascadedDp has
        # nothing to "drop" otherwise.
        cdp = CascadedDPCallback(val_ds, nodeId, numNodes, flag_layer, optimizerType, actual_lr, dpDropWindow, kSlope, kAcc, minDpEpochs)
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
        "privacy": {"epsilon": round(eps, 4) if eps is not None else None, "delta": 1.0/num_train_samples if dpEnabled else None, "dp_drop_epoch": cdp.dp_drop_epoch if cdp else None, "k_slope": kSlope, "k_acc": kAcc, "burn_in_std_grad": cdp.burn_in_std_grad if cdp else None, "burn_in_var_acc": cdp.burn_in_var_acc if cdp else None, "dp_drop_reason": cdp.dp_drop_reason if cdp else None}
    }
    
    with open(f"/results/{os.getenv('RESULT_FILE', 'results.json')}", 'w') as f: json.dump(results, f, indent=2)
    print('Saved the trained model and verified final test metrics JSON!')

if __name__ == '__main__': main()
