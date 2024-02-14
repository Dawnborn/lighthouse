# %% [markdown]
# # Main

# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

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

# %%
mykey = lambda x:int(x.split(".")[0].split("_")[-1])

def pose2H(pose):
    t = pose[-3:]
    q = pose[:4]
    q = q[[1,2,3,0]]
    Rot = R.from_quat(q).as_matrix()
    H = np.eye(4)
    H[:3,:3] = Rot
    H[:3,3] = t
    return H

def euclidean_depth(intrinsics, z_depth_image):
    fx = intrinsics[0, 0]  # focal length in the x-axis
    fy = intrinsics[1, 1]  # focal length in the y-axis
    cx = intrinsics[0, 2]  # optical center x-coordinate
    cy = intrinsics[1, 2]  # optical center y-coordinate

    # Generate pixel grid
    (height, width) = z_depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate Euclidean depth
    x = (u - cx) * z_depth_image / fx
    y = (v - cy) * z_depth_image / fy
    euclidean_depth = np.sqrt(x**2 + y**2 + z_depth_image**2)

    return euclidean_depth

# %%
def get_mbatch_inter(id_ref=1,id_src=3,id_env=99,format=None,seq_name="3FO4K13F24WL",traj="3",data_root = "/storage/local/lhao/junpeng/data/HD1"):
    # this function is for interiornet dataset
    def mykey2(element):
        return float(element.split(".")[0])

    mbatch = {}

    # mimage_root = "/home/wiss/lhao/junpeng/ws_lighthouse/data/setup_ptc/scene0370_02/pfm"
    # mimage_root = "/storage/local/lhao/junpeng/data/HD1/3FO4K13F24WL/original_1_1/cam0/data/"
    mimage_root = "{}/{}/original_{}_{}/cam0/data/".format(data_root,seq_name,traj,traj)
    mimage_dirs = sorted(os.listdir(mimage_root),key=mykey2)

    # id_ref = 1
    mref_image_path = os.path.join(mimage_root,mimage_dirs[id_ref-1])
    # print(mref_image_path)

    # mdepth_root = mimage_root.replace("pfm","depth")
    # mdepth_path = "/home/wiss/lhao/junpeng/ws_lighthouse/data/setup_ptc/scene0370_02/depth/bgrdepth_2.png"
    # mdepth_root = "/storage/local/lhao/junpeng/data/HD1/3FO4K13F24WL/original_1_1/depth0/data/"
    mdepth_root = "{}/{}/original_{}_{}/depth0/data/".format(data_root,seq_name,traj,traj)
    mdepth_path = os.path.join(mdepth_root,sorted(os.listdir(mdepth_root),key=mykey2)[id_ref-1])
    
    mintrinsic = np.array([[600,0,320],[0,600,240],[0, 0, 1]])
    
    # H_root = "/storage/user/lhao/hjp/ws_interiornet/output/img_and_pose/3FO4K13F24WL/pose_H"

    H_root = "/storage/user/lhao/hjp/ws_interiornet/output/img_and_pose/{}/original_{}_{}/pose_H".format(seq_name,traj,traj)
    H_dirs = sorted(os.listdir(H_root),key=mykey2)

    # ref_pose
    mH_ref = np.load(os.path.join(H_root,H_dirs[id_ref-1]))

    # src_images
    msrc_image_path = os.path.join(mimage_root,mimage_dirs[id_src-1])

    # src_pose
    mH_src = np.load(os.path.join(H_root,H_dirs[id_src-1]))
    
    # env_pose
    mH_env = np.load(os.path.join(H_root,H_dirs[id_env-1]))[np.newaxis,...]

    mbatch["ref_image"] = plt.imread(mref_image_path)[:,:,:3][np.newaxis, ...]
    mbatch["ref_depth"] = cv2.imread(mdepth_path,cv2.CV_16UC1)[np.newaxis, ...].astype(np.float32)/1000.0
    mbatch["intrinsics"] = mintrinsic[np.newaxis, ...]
    mbatch["ref_pose"] = mH_ref[np.newaxis, ...]
    mbatch["src_images"] = plt.imread(msrc_image_path)[:,:,:3][np.newaxis, ...]
    mbatch["src_poses"] = mH_src[np.newaxis,...,np.newaxis]
    mbatch["env_pose"] = mH_env.astype(np.float32)
    
    mbatch["ref_img_path"] = mref_image_path
    mbatch["ref_depth_path"] = mdepth_path
    mbatch["src_img_path"] = msrc_image_path

    return mbatch

# mbatch = get_mbatch_inter()
# print(mbatch["ref_image"].shape) # = 6 input image
# print(mbatch["ref_depth"].shape) # = 6 input depth
# print(mbatch["intrinsics"].shape)
# print(mbatch["src_images"].shape)
# print(mbatch["ref_pose"].shape)
# print(mbatch["src_poses"].shape)
# print(mbatch["env_pose"].shape)

# %%
# data_root = "/storage/local/lhao/junpeng/data/HD1"
# data_root = "/home/wiss/lhao/storage/user/hjp/interiornet_da"
# data_root = "/storage/local/lhao/junpeng/datanew/HD1"
data_root = "/storage/remote/atcremers95/lhao/junpeng/dataval/HD1"
# data_root = "/storage/remote/atcremers95/lhao/junpeng/dataval/HD6"

seq_root = "/storage/user/lhao/hjp/ws_interiornet/output/img_and_pose"

# seq_name = "3FO4K13F24WL"
# seq_name = "3FO4K6QIABEF"
# seq_name = "3FO4JY4U3OY0"
# seq_name = "3FO4K6QVKNFL"
# seq_name = "3FO4K6RLJS21"
# seq_name = "3FO4K72VETLU"
# seq_name = "3FO4JY378Y5G"
# seq_name = "3FO4K70L7E38"


seq_names = ['3FO4K72VETLU','3FO4JVJDJAX0']
seq_name = '3FO4K72VETLU'
# seq_name = '3FO4JVJDJAX0'
seq_name = "3FO4JVLBAWMA"
seq_name = "3FO4JVNWUHSO"
seq_name = "3FO4JY378Y5G"

# seq_name = "3FO4JY378Y5G"
# seq_name = "3FO4JVJDJAX0"
# seq_name = "3FO4JVLBAWMA"
seq_name = "3FO4JVNWUHSO"

seq_name = "3FO4K6YXM8JW"

traj = "3"

# data_dir = "./testset/test_set_with_coords_fixed/"
checkpoint_dir = "./lighthouse/model/"
# output_dir = "./lighthouse_output/"
# output_dir = "/home/wiss/lhao/junpeng/ws_lighthouse/output/output_lzq2frame_fixid2_inferedmpi/"
# output_dir = "/home/wiss/lhao/junpeng/ws_lighthouse/output/output_icl/"
# output_dir = "/home/wiss/lhao/junpeng/ws_lighthouse/output/output_lzq2frame_ldr_inferedmpi3_eudepth/"
# output_root = "/home/wiss/lhao/junpeng/ws_lighthouse/output/output_icl_gtmpi"
# output_root = "/home/wiss/lhao/junpeng/ws_lighthouse/output/output_icl_gtmpi"
# output_root = "/home/wiss/lhao/junpeng/ws_lighthouse/output/output_icl_gtmpi_eval"
output_root = "/home/wiss/lhao/junpeng/ws_lighthouse/output/output_icl_gtmpi_eval2"

output_dir = os.path.join(output_root,"{}/original_{}_{}".format(seq_name,traj,traj))

batch_size = 1  # implementation only works for batch size 1 currently.
height = 480  # px
width = 640  # px
env_height = 512  # px
env_width = 1024  # px
cube_res = 64  # px
# theta_res = 240  # px
# phi_res = 120  # px
theta_res = 512
phi_res = 256

r_res = 128  # px
scale_factors = [2, 4, 8, 16]
num_planes = 32
depth_clip = 20.0

# %%
# scene_list = ['3FO4K70L7E38', '3FO4K13F24WL', '3FO4K6QIABEF']

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

# pred = model.infer_mpi(src_images, ref_image, ref_pose, src_poses, intrinsics, mpi_planes) # the result of 3D conv encoder-decoder
# rgba_layers = pred["rgba_layers"]

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

# %%
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# id_ref = 21
# id_ref = 111
# id_ref = 211
# id_ref = 311
# id_ref= 411
# id_ref = 511
# id_ref = 611
# id_ref = 711
# id_ref = 811

# id_env = 291
# id_env = 271
# id_env = 376
# id_env = 151
# id_env = 141
# id_env = 286
# id_env = 931
# id_env = 701
# id_env = 261

# id_env = 271
# id_env = 291
# id_env = 431
# id_env= 961
# id_env = 71
# id_env = 101

id_src=6

ids_ref = list(range(0,1000,10))

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

sess1 = tf.Session(config=config)

with sess1 as sess:
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(checkpoint_dir, "model.ckpt"))

    # i = 0
    # for i in range(0, len(input_files)):
    # print("running example:", i)

    # Load inputs
    # batch = np.load(data_dir + input_files[i])
    for id_ref in tqdm(ids_ref):
        # if id_ref<309:
        #     continue
        id_ref = 581
        id_env = id_ref
        print(id_env)
        # batch = get_mbatch2(id_ref,id_src,id_env)
        batch = get_mbatch_inter(id_ref,id_src,id_env,format="ldr",seq_name=seq_name,data_root=data_root)
    
        output_lightvols_out_eval, output_lightvols_eval, output_envmap_eval, output_lightvol_centers_eval,all_shells_list_eval = sess.run(
            [lightvols_out, lightvols, output_envmap, lightvol_centers,all_shells_list],
            feed_dict={
                ref_image: batch["ref_image"],
                ref_depth: batch["ref_depth"],    
                intrinsics: batch["intrinsics"],
                ref_pose: batch["ref_pose"],
                src_images: batch["src_images"],
                src_poses: batch["src_poses"],
                env_pose: batch["env_pose"]
            })
            
        rgb = output_envmap_eval[0, :, :, :3]
        output_path = os.path.join( output_dir,"env{}_ref{}_src{}.png".format(str(id_env), str(id_ref), str(id_src)))
        plt.imsave(output_path, rgb)

        # rgb = output_envmap_eval.squeeze()[:,:,[2,1,0]]
        # output_path = os.path.join(output_dir,"env_ref{}_{}.pfm".format(str(id_ref),str(id_env)))
        # imageio.imwrite(output_path, rgb, format='PFM')
        
        break

# %%
# (1, 480, 640, 3)
# (1, 480, 640)
# (1, 3, 3)
# (1, 480, 640, 3)
# (1, 4, 4)
# (1, 4, 4, 1)
# (1, 4, 4)
