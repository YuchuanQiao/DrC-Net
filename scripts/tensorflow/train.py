"""
Example script to train a DrC-Net model.

python train.py --datalist train_data.txt --gpu 1 --model-dir ../model/ --epochs 1500 --phase_encoding RL --num_feature 6 --lr 0.0001 --image-loss mse --lambda 0.005

"""

import os
import random
import argparse
import glob
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import keras.backend as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))))
import DrC_Net as drc


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--datalist', help='data list text file')
parser.add_argument('--num_feature', type=int, default=1, help='number of FOD components used for registration, specify that data has multiple channel features')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
parser.add_argument("--phase_encoding", type=str, default='RL', help="phase encoding direction")
# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.00001)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image difference loss (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01, help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--kl-lambda', type=float, default=10, help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                    help='image noise parameter (recommended value is 0.02 when --use-probs is enabled)')
args = parser.parse_args()

# load and prepare training data
train_vol_names = np.loadtxt(args.datalist,dtype='str',ndmin=2)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# padding step to fit the network
sampling_num = 2**len(enc_nf)
fixed = drc.py.utils.load_volfile(train_vol_names[0,0], num_feature=args.num_feature)
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

generator = drc.generators.scan_to_scan_FOD(train_vol_names, pad_shape=tuple(new_vol_size),batch_size=args.batch_size, bidir=args.bidir, num_feature=args.num_feature)

# extract shape and number of features from sampled input
sample_shape = next(generator)[0][0].shape
print('sample shape',sample_shape)
inshape = sample_shape[1:-1]
print('inshape',inshape)
nfeats = sample_shape[-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow gpu handling
device = '/gpu:' + args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tf.keras.backend.set_session(tf.Session(config=config))

# ensure valid batch size given gpu count
nb_gpus = len(args.gpu.split(','))
assert np.mod(args.batch_size, nb_gpus) == 0, 'Batch size (%d) should be a multiple of the number of gpus (%d)' % (args.batch_size, nb_gpus)


# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

print('args.phase_encoding',args.phase_encoding)

with tf.device(device):

    # build the model
    model = drc.networks.DrC_net(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        use_probs=args.use_probs,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats,
        phase_encoding=args.phase_encoding
    )

    # load initial weights (if provided)
    if args.load_weights:
        model.load_weights(args.load_weights)

    # prepare image loss

    if args.image_loss == 'mse':
        image_loss_func = drc.losses.MSE(args.image_sigma).loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    print('args.image_loss',args.image_loss)

    # image_loss_func = lambda _, y_p: K.mean(K.square(y_p))
    # need two image loss functions if bidirectional
    if args.bidir:
        losses  = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses  = [image_loss_func]
        weights = [1]

    # prepare deformation loss
    if args.use_probs:
        flow_shape = model.outputs[-1].shape[1:-1]
        losses += [drc.losses.KL(args.kl_lambda, flow_shape).loss]
    else:
        losses += [drc.losses.Grad('l2').loss]

    weights += [args.lambda_weight]
    #
    # multi-gpu support
    if nb_gpus > 1:
        save_callback = drc.networks.ModelCheckpointParallel(save_filename)
        model = tf.keras.utils.multi_gpu_model(model, gpus=nb_gpus)
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))
    model.summary()
    model.fit_generator(generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[save_callback],
        verbose=2
    )

    # save final model weights
    model.save(save_filename.format(epoch=args.epochs))
