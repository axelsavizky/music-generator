@ex.capture()
def build_model(fragment_length, nb_filters, nb_output_bins, dilation_depth, nb_stacks, use_skip_connections,
                learn_all_outputs, _log, desired_sample_rate, use_bias, res_l2, final_l2):
    def residual_block(x):
        original_x = x
        # TODO: initalization, regularization?
        # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
        tanh_out = CausalAtrousConvolution1D(nb_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True,
                                             use_bias=use_bias,
                                             name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
                                             kernel_regularizer=l2(res_l2))(x)
        sigm_out = CausalAtrousConvolution1D(nb_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True,
                                             use_bias=use_bias,
                                             name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid',
                                             kernel_regularizer=l2(res_l2))(x)
        x = layers.Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

        res_x = layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
                                     kernel_regularizer=l2(res_l2))(x)
        skip_x = layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
                                      kernel_regularizer=l2(res_l2))(x)
        res_x = layers.Add()([original_x, res_x])
        return res_x, skip_x

    input = Input(shape=(fragment_length, nb_output_bins), name='input_part')
    out = input
    skip_connections = []
    out = CausalAtrousConvolution1D(nb_filters, 2,
                                    dilation_rate=1,
                                    padding='valid',
                                    causal=True,
                                    name='initial_causal_conv'
                                    )(out)
    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out, skip_out = residual_block(out)
            skip_connections.append(skip_out)

    if use_skip_connections:
        out = layers.Add()(skip_connections)
    out = layers.Activation('relu')(out)
    out = layers.Convolution1D(nb_output_bins, 1, padding='same',
                               kernel_regularizer=l2(final_l2))(out)
    out = layers.Activation('relu')(out)
    out = layers.Convolution1D(nb_output_bins, 1, padding='same')(out)

    if not learn_all_outputs:
        raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
        out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(
            out)  # Based on gif in deepmind blog: take last output?

    out = layers.Activation('softmax', name="output_softmax")(out)
    model = Model(input, out)

    receptive_field, receptive_field_ms = compute_receptive_field()

    _log.info('Receptive Field: %d (%dms)' % (receptive_field, int(receptive_field_ms)))
    return model
