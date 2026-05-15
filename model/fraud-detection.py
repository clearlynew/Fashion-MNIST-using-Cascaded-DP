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
import numpy as np
import csv
import logging
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
# Correct import for tensorflow-privacy==0.7.3
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

from swarmlearning.tf import SwarmCallback


def getXY(dataSet):
    np.random.shuffle(dataSet)
    length = np.size(dataSet,0)
    X = dataSet[0:length, :-1]
    y = dataSet[0:length, -1:]
    return X , y


# Constants
testFileName = 'SB19_CCFDUBL_TEST.csv'
trainFileName = 'SB19_CCFDUBL_TRAIN.csv'

part = 0
batchSize = 32
defaultMaxEpoch = 100
defaultMinPeers = 2

def main():
  modelName = 'fraud-detection'
  dataDir = os.getenv('DATA_DIR', '/platform/data')
  scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
  maxEpoch = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
  minPeers = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))
  dpEnabled = os.getenv('DP_ENABLED', 'false').lower() == 'true'
  noiseMultiplier = float(
    os.getenv('NOISE_MULTIPLIER', '0.0')
  )
  l2NormClip = float(
    os.getenv('L2_NORM_CLIP', '1.0')
  )
  microbatches = int(
    os.getenv('MICROBATCHES', str(batchSize))
  )
  os.makedirs(scratchDir, exist_ok=True)
  print('***** Starting model =', modelName)
  # ================== load test and train Data =========================
  print('-' * 64)

  trainFile = dataDir + '/' + trainFileName
  print("loading train dataset %s .." % trainFile)
  with open(trainFile, 'r') as f:
    # first line is the header row so remove it
    trainData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
    num_train_samples = np.size(trainData, 0)
    print('size of training Data set : %s' % num_train_samples)

  print('-' * 64)
  testFile = dataDir + '/' + testFileName
  print("loading test dataset %s .." % testFile)
  with open(testFile, 'r') as f:
    # first line is the header row so remove it
    testData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
    print('size of test Data set : %s' % np.size(testData,0))

  print('-' * 64)
  # ================== Model to train and evaluate =========================
  # logistic regression Model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=(30,), activation='sigmoid',
    kernel_initializer='random_uniform', bias_initializer='zeros'))
  if dpEnabled:
    print("***** Using DP-SGD optimizer")

    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2NormClip,
        noise_multiplier=noiseMultiplier,
        num_microbatches=microbatches,
        learning_rate=0.01
    )

  else:
    print("***** Using standard SGD optimizer")

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.01,
        decay=1e-6,
        momentum=0.9,
        nesterov=True
    )
    
  loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.NONE
  )

  model.compile(loss = loss,
                optimizer=optimizer,
                metrics=[tf.keras.metrics.AUC()])
  print(model.summary())

  print('Starting training ...')
  x_train, y_train = getXY(trainData)
  x_test, y_test = getXY(testData)

  # Professional Data Pipeline Fix for DP-SGD
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_ds = train_ds.shuffle(len(x_train)).batch(batchSize, drop_remainder=True)
  train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

  val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize)
  val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

  # Adding swarm callback
  swarmCallback = SwarmCallback(syncFrequency=128,
                                minPeers=minPeers,
                                adsValData=val_ds,
                                adsValBatchSize=batchSize,
                                mergeMethod='mean',
                                totalEpochs=maxEpoch)

  # Model training
  model.fit(
      train_ds
    , epochs=maxEpoch
    , validation_data=val_ds
    , callbacks=[swarmCallback]
  )

  print('Training done!')

  # Calculate Epsilon and Delta if DP is enabled
  if dpEnabled and noiseMultiplier > 0:
      print('-' * 64)
      print('***** PRIVACY REPORT *****')
      # Calculate delta based on dataset size
      delta = 1.0 / num_train_samples
      
      # compute_dp_sgd_privacy returns (epsilon, optimal_order)
      eps, _ = compute_dp_sgd_privacy(
          n=num_train_samples, 
          batch_size=batchSize, 
          noise_multiplier=noiseMultiplier, 
          epochs=maxEpoch, 
          delta=delta
      )
      print(f"Final Epsilon (ε): {eps:.2f}")
      print(f"Final Delta (δ):   {delta:.2e}")
      print('**************************')
      print('-' * 64)
  elif dpEnabled and noiseMultiplier <= 0:
      print("***** WARNING: noise_multiplier is 0.0. Privacy budget is infinite.")

  # Evaluate
  scores = model.evaluate(val_ds, verbose=1)
  print('***** Test loss:', scores[0])
  print('***** Test auc:', scores[1])

  # Save
  model_path = os.path.join(scratchDir, modelName)
  model.save(model_path)
  print('Saved the trained model!')


if __name__ == '__main__':
    main()
