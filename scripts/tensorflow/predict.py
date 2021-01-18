"""
Example script to predict a displacement field from paired PE FOD images with DrC-Net models.

python predict.py --datalist whole_data.txt --gpu 0 --model ../model/1500.h5 --num_feature 6 --exp_name DrC_net
"""

import os
import argparse
import numpy as np
import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))))

import DrC_Net as drc
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--datalist', help='data list text file')
parser.add_argument('--exp_name', help='experiment name for output dir')
parser.add_argument('--num_feature', type=int, default=1, help='number of FOD components used for registration, specify that data has multiple channel features')
parser.add_argument('--diff', help='difference image output filename')
parser.add_argument('--model', help='run nonlinear registration - must specify keras model file')
parser.add_argument('--warp', help='output warp filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
args = parser.parse_args()

# sanity check on the input
assert (args.model or args.affine_model), 'must provide at least a warp or affine model'

# device handling
if args.gpu and (args.gpu != '-1'):
    device = '/gpu:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.keras.backend.set_session(tf.Session(config=config))
else:
    device = '/cpu:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# unet architecture
enc_nf = [16, 32, 32, 32]
# padding step to fit the network
sampling_num = 2**len(enc_nf)
train_vol_names = np.loadtxt(args.datalist,dtype='str',ndmin=2)
# fixed = drc.py.utils.load_volfile(train_vol_names[0,0], add_feat_axis=add_feat_axis)
num_subject = len(train_vol_names)
for j in range(num_subject):

    base_name = train_vol_names[j,0].split('Diffusion')[0]
    output_dir = os.path.join(base_name,'Diffusion','eddy_unwarp',args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    warp_name = os.path.join(output_dir,'deformation.nii.gz')
    diff_name = os.path.join(output_dir,'diff_image.nii.gz')

    if not os.path.exists(warp_name):
        a = datetime.datetime.now()
        fixed = drc.py.utils.load_volfile(train_vol_names[j,1], num_feature=args.num_feature)

        vol_size = fixed.shape
        padding_num = []
        pad_dim = len(vol_size)-1
        if args.num_feature > 1:
            # 4D volume
            new_vol_size = list(vol_size)
        else:
            new_vol_size = list(vol_size[:-1])
            print('new_vol_size',new_vol_size)
        for i in range(pad_dim):
            divid_val = int(vol_size[i])/sampling_num
            tmp = int((np.ceil(divid_val) - divid_val)*sampling_num)
            padding_num.append(tmp)
            new_vol_size[i] = vol_size[i] + tmp
        print('new_vol_size',new_vol_size)
        print('vol_size',vol_size)
        print('padding_num',padding_num)

        moving = drc.py.utils.load_volfile(train_vol_names[j,0], add_batch_axis=True, num_feature=args.num_feature, pad_shape=tuple(new_vol_size))
        fixed,fixed_affine  = drc.py.utils.load_volfile(train_vol_names[j,1], add_batch_axis=True, num_feature=args.num_feature, pad_shape=tuple(new_vol_size),ret_affine=True)

        offsets = [int((p - v) / 2) for p, v in zip(new_vol_size, vol_size)]
        slices  = tuple([slice(offset, l + offset) for offset, l in zip(offsets, vol_size)])
        if args.num_feature > 1:
            # take first 3 dimensions
            warp_slices = tuple(list(slices)[0:3])
        else:
            warp_slices = slices + tuple([slice(0,3)])
        print('offsets', offsets)
        print('slices', slices)
        print('warp_slices', warp_slices)

        if args.model:

            with tf.device(device):
                # load model and predict
                diff,warp = drc.networks.DrC_net.load(args.model).predict([moving, fixed])
                warp = warp.squeeze()
                diff  = diff.squeeze()
                print(warp.shape)
                print(diff.shape)
                warp_unpad = warp[warp_slices]
                diff_unpad  = diff[slices]
                print(warp_unpad.shape)

            # save warp
            drc.py.utils.save_volfile(warp_unpad, warp_name, fixed_affine)
            drc.py.utils.save_volfile(diff_unpad, diff_name, fixed_affine)
            b = datetime.datetime.now()
            c= b-a
            print('The time elapsed for prediction is',int(c.total_seconds()),'seconds.')
