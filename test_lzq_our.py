# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# 打印当前工作目录
print("当前工作目录：", os.getcwd())
# 改变工作目录
new_directory = "/storage/user/lhao/hjp/ws_lightshouse"
os.chdir(new_directory)
# 再次打印当前工作目录，确认改变
print("改变后的工作目录：", os.getcwd())

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import imageio
from tqdm import tqdm

import geometry.projector as pj
from mlv import MLV
import nets as nets

from scipy.spatial.transform import Rotation as R

from data_loader import MyDataloader_lzq

from tqdm import tqdm

# %%
exp_dir = "/storage/user/lhao/hjp/ws_lightshouse/lighthouse/experiment_finetune"
ckpt = "75000"

exp_dir = "/storage/user/lhao/hjp/ws_lightshouse/lighthouse/experiment_finetune_depth8"
ckpt = "100000"

checkpoint_dir = os.path.join(exp_dir,"checkpoints")
# output_dir = "/home/wiss/lhao/junpeng/ws_lighthouse/output_lzq2frame_our"
output_dir = os.path.join(exp_dir,"outputs/{}".format(ckpt))
output_dir = os.path.join(exp_dir,"outputs_200test/{}".format(ckpt))
print("output directory: ", output_dir)
import pdb; pdb.set_trace()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

batch_size = 1  # implementation only works for batch size 1 currently.
height = 480  # px
width = 640  # px
env_height = 256  # px
env_width = 512  # px
cube_res = 64  # px
theta_res = 512  # px
phi_res = 256  # px
r_res = 128  # px
scale_factors = [2, 4, 8, 16]
num_planes = 32
depth_clip = 10.0

root_dir = "/storage/user/lhao/hjp/ws_interiornet_stable/output/setup_ptc_ours"
start_end=(0.8,1)

root_dir = "/storage/user/lhao/hjp/ws_interiornet_stable/output/setup_ptc_ours_200test"
start_end=(0.,1)

mydataloader = MyDataloader_lzq(root_dir=root_dir,start_end=start_end)

ourbatch = mydataloader[5]
ourbatch.keys()

# %%
# train_iterator = mydataloader.tf_data_generator()
# train_iterator

# %%
output_signature = {
    "ref_image":tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3]),
    "ref_depth":tf.placeholder(dtype=tf.float32, shape=[None, height, width]),
    "intrinsics":tf.placeholder(dtype=tf.float32, shape=[None, 3, 3]),
    "env_pose":tf.placeholder(dtype=tf.float32, shape=[None, 4, 4])
}
# tf.data.Dataset.from_generator(mydataloader,output_signature=output_signature)

# %%
tf.__version__

# %%
print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available())
print(tf.__version__)
print(tf.config.experimental.list_physical_devices(device_type="GPU"))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) 

# %%
# Set up placeholders
ref_image = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
ref_depth = tf.placeholder(dtype=tf.float32, shape=[None, height, width])
intrinsics = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3])
ref_pose = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4])
src_images = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
src_poses = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 1])
env_pose = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4])

# Set up model
model = MLV()

# We use the true depth bounds for testing
# Adjust to estimated bounds for your dataset
min_depth = tf.reduce_min(ref_depth)
max_depth = tf.reduce_max(ref_depth)

mpi_planes = pj.inv_depths(min_depth, max_depth, num_planes)

mpi_gt = model.img2mpi(ref_image,ref_depth,mpi_planes)

lightvols, lightvol_centers, \
lightvol_side_lengths, \
cube_rel_shapes, \
cube_nest_inds = model.predict_lighting_vol(mpi_gt, mpi_planes,
                                            intrinsics, cube_res,
                                            scale_factors,
                                            depth_clip=depth_clip)
lightvols_out = nets.cube_net_multires(lightvols, cube_rel_shapes,
                                        cube_nest_inds)
output_envmap, all_shells_list = model.render_envmap(lightvols_out, lightvol_centers,
                                        lightvol_side_lengths, cube_rel_shapes,
                                        cube_nest_inds, ref_pose, env_pose,
                                        theta_res, phi_res, r_res)
#%%
# for batch in mydataloader:
#     break

# %%

# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# sess1 = tf.Session(config=config)
sess1 = tf.Session()

with sess1 as sess:
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(checkpoint_dir, "model.ckpt-{}".format(ckpt)))

    # i = 0
    # for i in range(0, len(input_files)):
    # print("running example:", i)

    # Load inputs
    # batch = np.load(data_dir + input_files[i])
    for id_ref, batch in tqdm(enumerate(mydataloader)): # 
        # batch = get_mbatch(id_ref,id_src,id_env)
    
        output_lightvols_out_eval, output_lightvols_eval, output_envmap_eval, output_lightvol_centers_eval,all_shells_list_eval = sess.run(
            [lightvols_out, lightvols, output_envmap, lightvol_centers,all_shells_list],
            feed_dict={
                ref_image: batch["ref_image"][np.newaxis, ...],
                ref_depth: batch["ref_depth"][np.newaxis, ...],
                intrinsics: batch["intrinsics"][np.newaxis, ...],
                ref_pose: batch["ref_pose"][np.newaxis, ...],
                env_pose: batch["env_pose"][np.newaxis, ...]
            })
            
        rgb = output_envmap_eval.squeeze()[:,:,[2,1,0]]
        pred_dir = os.path.join(output_dir,"pred")
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        output_path = os.path.join(pred_dir,"env_ref{}.png".format(str(id_ref)))
        imageio.imwrite(output_path, rgb, format='PNG')

        ref = batch["ref_image"][:,:,[2,1,0]]
        ref_dir = os.path.join(output_dir,"ref")
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        ref_path = os.path.join(ref_dir,"env_ref{}.png".format(str(id_ref)))
        imageio.imwrite(ref_path, ref, format='PNG')

        real = batch["env_image"][:,:,[2,1,0]]
        real_dir = os.path.join(output_dir,"real")
        if not os.path.exists(real_dir):
            os.makedirs(real_dir)
        real_path = os.path.join(real_dir,"env_ref{}.png".format(str(id_ref)))
        imageio.imwrite(real_path, real, format='PNG')
    

# %%
print(output_lightvols_out_eval[0].shape) # (1, 64, 64, 64, 4)
print(output_lightvols_eval[0].shape) # (1, 64, 64, 64, 4)
print(output_lightvol_centers_eval[4])
print(all_shells_list_eval[0].shape)



# %%
