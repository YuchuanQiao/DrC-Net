import numpy as np
from collections.abc import Iterable

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

from .. import default_unet_features
from . import layers
from . import neuron as ne
from .modelio import LoadableModel, store_config_args
from .utils import gaussian_blur, value_at_location, point_spatial_transformer


# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

class DrC_net(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=7,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            trg_feats=1,
            input_model=None,
            phase_encoding='RL'):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
            phase_encoding: phase encoding direction, default is RL.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        print('source shape',source.shape)
        # build core unet model and grab inputs
        unet_model = Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='flow')(unet_model.output)
        pos_flow = flow_mean
        neg_flow = ne.layers.Negate(name='neg_flow')(flow_mean)
        # ---------------------------------------------------------------------------------------------------
        # this may be diffeomorphic transformation
        # # optionally include probabilities
        # if use_probs:
        #     # initialize the velocity variance very low, to start stable
        #     flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
        #                     kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
        #                     bias_initializer=KI.Constant(value=-10),
        #                     name='log_sigma')(unet_model.output)
        #     flow_params = KL.concatenate([flow_mean, flow_logsigma], name='prob_concat')
        #     flow = ne.layers.SampleNormalLogVar(name="z_sample")([flow_mean, flow_logsigma])
        # else:
        #     flow_params = flow_mean
        #     flow = flow_mean
        #
        # # optionally resize for integration
        # if int_steps > 0 and int_downsize > 1:
        #     flow = layers.RescaleTransform(1 / int_downsize, name='flow_resize')(flow)
        #
        # # optionally negate flow for bidirectional model
        # pos_flow = flow
        # neg_flow = ne.layers.Negate(name='neg_flow')(flow)
        #
        # # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        # if int_steps > 0:
        #     pos_flow = ne.layers.VecInt(method='ss', name='flow_int', int_steps=int_steps)(pos_flow)
        #     neg_flow = ne.layers.VecInt(method='ss', name='neg_flow_int', int_steps=int_steps)(neg_flow)
        #
        #     # resize to final resolution
        #     if int_downsize > 1:
        #         pos_flow = layers.RescaleTransform(int_downsize, name='diffflow')(pos_flow)
        #         neg_flow = layers.RescaleTransform(int_downsize, name='neg_diffflow')(neg_flow)
        # ---------------------------------------------------------------------------------------------------

        # warp image with flow field
        y_source = layers.SpatialTransformer(interp_method='linear', indexing='ij', phase_encoding=phase_encoding,name='transformer')([source, pos_flow])
        y_target = layers.SpatialTransformer(interp_method='linear', indexing='ij', phase_encoding=phase_encoding,name='neg_transformer')([target, neg_flow])

        y_diff  = ne.layers.SpatialDiff(name='y_diff')([y_source, y_target])

        # initialize the keras model
        # outputs = [y_source, y_target, flow_params] if bidir else [y_source, flow_params]
        outputs=[y_diff, flow_mean]
        super().__init__(name='DrC_net', inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow
        self.references.y_diff   = y_diff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])



class Transform(tf.keras.Model):
    """
    Simple transform model to apply dense or affine transforms.
    """

    def __init__(self, inshape, affine=False, interp_method='linear', nb_feats=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            affine: Enable affine transform. Default is False.
            interp_method: Interpolation method. Can be 'linear' or 'nearest'. Default is 'linear'.
            nb_feats: Number of source image features. Default is 1.
        """

        # configure inputs
        ndims = len(inshape)
        scan_input = tf.keras.Input((*inshape, nb_feats), name='scan_input')

        if affine:
            trf_input = tf.keras.Input((ndims * (ndims + 1),), name='trf_input')
        else:
            trf_input = tf.keras.Input((*inshape, ndims), name='trf_input')

        # transform and initialize the keras model
        y_source = layers.SpatialTransformer(interp_method=interp_method, name='transformer')([scan_input, trf_input])
        super().__init__(inputs=[scan_input, trf_input], outputs=y_source)


def conv_block(x, nfeat, strides=1, name=None):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    Conv = getattr(KL, 'Conv%dD' % ndims)

    convolved = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(x)
    # print('Conv%dD' % ndims,'convolved x shape',convolved.shape)
    name = name + '_activation' if name else None
    return KL.LeakyReLU(0.2, name=name)(convolved)


def upsample_block(x, connection, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    upsampled = UpSampling(name=name)(x)
    # print('Conv%dD' % ndims,'upsampled x shape',upsampled.shape)
    name = name + '_concat' if name else None
    return KL.concatenate([upsampled, connection], name=name)


class Unet(tf.keras.Model):
    """
    A unet architecture that builds off of an input keras model. Layer features can be specified directly
    as a list of encoder and decoder features or as a single integer along with a number of unet levels.
    The default network features per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self, input_model, nb_features=None, nb_levels=None, feat_mult=1, nb_conv_per_level=1):
        """
        Parameters:
            input_model: Input model that feeds directly into the unet before concatenation.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
        """

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        # configure encoder (down-sampling path)
        enc_layers = [KL.concatenate(input_model.outputs, name='unet_input_concat')]
        last = enc_layers[0]
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                strides = 2 if conv == (nb_conv_per_level - 1) else 1
                name = 'unet_enc_conv_%d_%d' % (level, conv)
                last = conv_block(last, nf, strides=strides, name=name)
            enc_layers.append(last)

        # configure decoder (up-sampling path)
        last = enc_layers.pop()
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                name = 'unet_dec_conv_%d_%d' % (real_level, conv)
                last = conv_block(last, nf, name=name)
            name = 'unet_dec_upsample_' + str(real_level)
            last = upsample_block(last, enc_layers.pop(), name=name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            name = 'unet_dec_final_conv_' + str(num)
            last = conv_block(last, nf, name=name)

        return super().__init__(inputs=input_model.inputs, outputs=last)
