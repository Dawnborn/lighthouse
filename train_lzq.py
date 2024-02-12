# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train multiscale lighting volume prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import scipy.io as sio
import tensorflow as tf

import lighthouse.data_loader as loader
from lighthouse.mlv import MLV

flags.DEFINE_string("vgg_model_file", default="/storage/user/huju/transferred/ws_lighthouse/lighthouse/model/imagenet-vgg-verydeep-19.mat", help="VGG weights filename")
flags.DEFINE_string(
    "load_dir",
    default="./lighthouse/model/",
    help="Directory for loading checkpoint to continue training")
flags.DEFINE_string(
    "data_dir",
    default="/storage/local/lhao/junpeng/interiornet_dataset/",
    help="InteriorNet dataset directory")
flags.DEFINE_string(
    "experiment_dir",
    default="/storage/user/huju/transferred/ws_lighthouse/lighthouse/experiment_dir",
    help="Directory to store experiment summaries and checkpoints")

FLAGS = flags.FLAGS

# Model parameters
batch_size = 1  # implementation only works for batch size 1 currently
height = 480  # px
width = 640  # px
env_height = 512  # px
env_width = 1024  # px
cube_res = 64  # px
theta_res = 1024  # px
phi_res = 512  # px
r_res = 128  # px
scale_factors = [2, 4, 8, 16]  # try omitting 16 if you have GPU memory issues
num_planes = 32
depth_clip = 20.0  # change depending on your dataset

# Training parameters
random_seed = 0
learning_rate = 1e-3
summary_freq = 20
checkpoint_freq = 500
max_steps = 720000

tf.compat.v1.set_random_seed(random_seed)


def main(argv):

  del argv  # unused

  if FLAGS.vgg_model_file is None:
    raise ValueError("`vgg_model_file` must be defined")

  # Load VGG model
  vgg_rawnet = sio.loadmat(FLAGS.vgg_model_file)
  vgg_layers = vgg_rawnet["layers"][0]

  checkpoint_dir = os.path.join(FLAGS.experiment_dir, "checkpoints")
  summary_dir = os.path.join(FLAGS.experiment_dir, "summaries")

  if not FLAGS.load_dir:
    load_dir = checkpoint_dir
  else:
    load_dir = FLAGS.load_dir

  if not os.path.exists(FLAGS.experiment_dir):
    os.mkdir(FLAGS.experiment_dir)
  if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
  if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

  # Create datasets and iterators
  # ds = loader.data_loader(parent_dir=FLAGS.data_dir,dataset_list=['HD1'])
  # train_iterator = ds.training.make_one_shot_iterator()

  root_dir = "/storage/user/lhao/hjp/ws_interiornet_stable/output/setup_ptc_ours"
  start_end = (0, 0.8)
  mydataloader = loader.MyDataloader_lzq(root_dir=root_dir, start_end=start_end)
  train_iterator = mydataloader.tf_data_generator()

  # Set up input pipeline
  # s = train_iterator.get_next()
  s = next(train_iterator)
  # batch = loader.format_inputs(s, height, width, env_height, env_width)
  batch = s
  min_depth = tf.reduce_min(batch["ref_depth"])
  max_depth = tf.reduce_max(batch["ref_depth"])

  # Set up training operation
  model = MLV()
  global_step = tf.placeholder(tf.int32, name="global_step")
  tf.summary.scalar("global step", global_step)
  train_op = model.build_train_graph(
      batch,
      min_depth,
      max_depth,
      cube_res,
      theta_res,
      phi_res,
      r_res,
      scale_factors,
      num_planes,
      learning_rate=learning_rate,
      vgg_model_weights=vgg_layers,
      global_step=global_step,
      depth_clip=depth_clip)
  print("finished setting up training graph")

  # Run training iterations
  model.train(train_op, load_dir, checkpoint_dir, summary_dir, summary_freq,
              checkpoint_freq, max_steps, global_step)


if __name__ == "__main__":
  app.run(main)
