import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def get_gradients(model, pulse_input, top_pred_idx=None, hidden_cam=False):
    """Computes the gradients of outputs w.r.t input signals.

    Args:
        pulse_input: 3D array 
        top_pred_idx: Predicted label for the input pulse
        hidden_cam: gradient w.r.t hidden layer or input layer

    Returns:
        Gradients of the predictions w.r.t pulse input
    """
    pulse_input = tf.cast(pulse_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(pulse_input)
        if hidden_cam == False:
            preds = model(pulse_input, training=False)
        else:
            hidden_out, preds = model(pulse_input, training=False)
        if top_pred_idx is None:
            top_pred_idx= tf.argmax(preds[0])
        top_class = preds[:, top_pred_idx]
    if hidden_cam == False:
        grads = tape.gradient(top_class, pulse_input)
        return grads
    else:
        grads = tape.gradient(top_class, hidden_out)
        return grads, hidden_out
    

def get_integrated_gradients(model, pulse_input, top_pred_idx=None, 
                             baseline=None, num_steps=50, pulse_size=200, hidden_cam=False):
    """Computes Integrated Gradients for a predicted label.

    Args:
        pulse_input (ndarray): Original pulse input
        top_pred_idx: Predicted label for the input
        baseline (ndarray): The baseline pulse to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        hidden_cam: whether or not to produce the hidden outputs, at which features 
                    are to be explained 

    Returns:
        Integrated gradients w.r.t input pulse
    """
    # If baseline is not provided, start with a black pulse
    # having same size as the input pulse.
    if baseline is None:
        baseline = np.zeros((pulse_size, 2)).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    pulse_input = pulse_input.astype(np.float32)
    interpolated_pulse = [
        baseline + (step / num_steps) * (pulse_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_pulse = np.array(interpolated_pulse).astype(np.float32)

    # 2. Get the gradients
    grads = []
    hidden_outs= []
    for i, pit in enumerate(interpolated_pulse):
        pit = tf.expand_dims(pit, axis=0)
        if hidden_cam == False:
            grad = get_gradients(model, pit, top_pred_idx=top_pred_idx)
            grads.append(grad[0])
        else:
            grad, hidden_out= get_gradients(model, pit, top_pred_idx=top_pred_idx, hidden_cam=True)
            grads.append(grad[0])
            hidden_outs.append(hidden_out[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)
    if hidden_cam == True:
        hidden_outs= tf.convert_to_tensor(hidden_outs, dtype=tf.float32)

    # 3. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    if hidden_cam == False:
        avg_grads = tf.reduce_mean(grads, axis=0)
    else:
        avg_grads = tf.reduce_mean(grads, axis=(0,1))

    # 4. Calculate integrated gradients and return
    if hidden_cam == False:
        igs = (pulse_input - baseline) * avg_grads
    else:
        igs = (hidden_outs[-1]-hidden_outs[0]) * avg_grads
        
    return igs


def runs_of_integrated_gradients(
    model, pulse_input, num_steps=50, num_runs=2, hidden_cam=False
):
    """Generates a number of random baseline pulses.

    Args:
        pulse_input (ndarray): 2D pulse input
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        num_runs: number of baseline pulses to generate

    Returns:
        Averaged integrated gradients for `num_runs` baseline pulses
    """
    # 1. List to keep track of Integrated Gradients (IG) for all the pulses
    integrated_grads = []

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
        igrads = get_integrated_gradients(
            model=model,
            pulse_input=pulse_input,
            num_steps=num_steps,
            hidden_cam=hidden_cam
        )
        integrated_grads.append(igrads)

    # 3. Return the average integrated gradients for the pulse
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    if hidden_cam == False:
        # not reduce the last axis, means keep gradients for both voltage and current
        integrated_grads= tf.reduce_mean(integrated_grads, axis=(0))
    else:
        integrated_grads= tf.reduce_mean(integrated_grads, axis=(0,1))
        integrated_grads= integrated_grads / tf.math.reduce_max(tf.math.abs(integrated_grads))
    return integrated_grads

def CAM_integrated_gradients(model, pulse_input, 
                             hidden_layer_name, last_layer_name):
    """Hidden CAM with integrated gradients

    Args:
        model: pre-trained model
        pulse_input (ndarray): 2D pulse input
        hidden_layer_name (string): intermediate network layer name
        last_layer_name (string): last network layer name

    Returns:
        CAM outputs
    """
    # remove last layer's softmax
    grad_model= model
    grad_model.layers[-1].activation= None

    cam_model= keras.Model(
        [model.inputs], [grad_model.get_layer(hidden_layer_name).output, 
                        grad_model.get_layer(last_layer_name).output]
    )
    cam_input= pulse_input[tf.newaxis,:]
    hidden_out, _= cam_model(cam_input, training=False)
    igrads= runs_of_integrated_gradients(cam_model, pulse_input, hidden_cam=True)
    igrads= igrads[..., tf.newaxis]

    hidden_out = hidden_out[0]
    heatmap = hidden_out @ igrads
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap


def Input_integrated_gradients(model, pulse_input, last_layer_name):
    """Input feature importance with integrated gradients

    Args:
        model: pre-trained model
        pulse_input (ndarray): 2D pulse input
        last_layer_name (string): last network layer name

    Returns:
        Input integrated gradients
    """
    # remove last layer's softmax
    grad_model= model
    grad_model.layers[-1].activation= None

    ig_model= keras.Model(
        [model.inputs],  [grad_model.get_layer(last_layer_name).output]
    )
    igrads= runs_of_integrated_gradients(ig_model, pulse_input, hidden_cam=False)

    return igrads


def plot_ig_input_heatmp(model, input_data,
                         vol_max_tmp=160, vol_min=-10, cur_max_tmp=30, cur_min=0):
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.rcParams.update({'font.size': 7})
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams.update({'axes.linewidth': 1})
    linewidth=0.5

    # !!! those three parameter should be tuned w.r.t different cases
    plt_vol_sig= input_data[:, 0]*(vol_max_tmp - vol_min)+vol_min
    plt_cur_sig= input_data[:, 1]*(cur_max_tmp - cur_min)+cur_min

    fig, axs= plt.subplots(3, 1, gridspec_kw={'height_ratios':[1,1,4],}, figsize=(2,2), dpi=300)
    p1, =axs[-1].plot(plt_vol_sig, 'tab:orange', linewidth=linewidth, label='voltage')
    axs[-1].set_ylim([vol_min, vol_max_tmp*1.2])
    axs[-1].set_ylabel("Voltage/ V")
    axs[-1].set_xlabel("Time instance")
    axs2= axs[-1].twinx()
    p2, =axs2.plot(plt_cur_sig, 'tab:green', linewidth=linewidth, label='current')
    axs2.set_ylim([cur_min-1, cur_max_tmp*1.2])
    axs2.set_ylabel("Current/ A")
    axs2.set_xlim([0, len(plt_cur_sig)])
    # set color for edge and ticks
    axs[-1].yaxis.label.set_color(p1.get_color())
    axs[-1].spines["left"].set_edgecolor(p1.get_color())
    axs[-1].tick_params(axis='y', colors=p1.get_color())
    axs2.yaxis.label.set_color(p2.get_color())
    axs2.spines['left'].set_visible(False) # trick!: need to invisual of the ax2' left spine 
    axs2.spines["right"].set_edgecolor(p2.get_color())
    axs2.tick_params(axis='y', colors=p2.get_color())

    # Input feature importance
    last_layer_name= 'tcn_sfm'
    grads= Input_integrated_gradients(model, input_data, last_layer_name)
    grads_vol= grads[:,0]
    grads_vol /= np.max(np.abs(grads_vol), axis=0)
    axs[0].imshow(np.atleast_2d(grads_vol), cmap='Spectral', aspect='auto', vmin=-1, vmax=1)
    axs[0].set_xticklabels("")
    axs[0].set_yticklabels("")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    grads_cur= grads[:,1]
    grads_cur /= np.max(np.abs(grads_cur), axis=0)
    shw_in= axs[1].imshow(np.atleast_2d(grads_cur), cmap='Spectral', aspect='auto', vmin=-1, vmax=1)
    axs[1].set_xticklabels("")
    axs[1].set_yticklabels("")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # add space for colour bar
    fig.subplots_adjust(right=0.65)
    cbar_ax = fig.add_axes([0.85, 0.65, 0.04, 0.3]) # left,bottom,width,height
    fig.colorbar(shw_in, cax=cbar_ax)

    fig.tight_layout()
    plt.show()


def plot_ig_cam_heatmap_tcn(model, input_data,
                        vol_max_tmp=160, vol_min=-10, cur_max_tmp=30, cur_min=0, num_heatmaps=4):
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    # from plot_gradcam import make_gradcam_heatmap
    matplotlib.rcParams.update({'font.size': 7})
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams.update({'axes.linewidth': 1})
    linewidth=1.5

    plt_vol_sig= input_data[:, 0]*(vol_max_tmp - vol_min)+vol_min
    plt_cur_sig= input_data[:, 1]*(cur_max_tmp - cur_min)+cur_min

    height_ratio=[1]*num_heatmaps
    height_ratio.append(6)
    fig, axs= plt.subplots(num_heatmaps+1, 1, 
                           gridspec_kw={'height_ratios':height_ratio,
                                        'hspace':0.2,},
                           figsize=(2,2), dpi=300)
    p1, =axs[-1].plot(plt_vol_sig, 'tab:orange', linewidth=linewidth, label='voltage')
    axs[-1].set_ylim([vol_min, vol_max_tmp*2.2])
    axs[-1].set_ylabel("Voltage/ V")
    axs[-1].set_xlabel("Time instance")
    axs2= axs[-1].twinx()
    p2, =axs2.plot(plt_cur_sig, 'tab:green', linewidth=linewidth, label='current')
    axs2.set_ylim([cur_min-1, cur_max_tmp*1.2])
    axs2.set_ylabel("Current/ A")
    axs2.set_xlim([0, len(plt_cur_sig)])
    # set color for edge and ticks
    axs[-1].yaxis.label.set_color(p1.get_color())
    axs[-1].spines["left"].set_edgecolor(p1.get_color())
    axs[-1].tick_params(axis='y', colors=p1.get_color())
    axs2.yaxis.label.set_color(p2.get_color())
    axs2.spines['left'].set_visible(False) # trick!: need to invisual of the ax2' left spine 
    axs2.spines["right"].set_edgecolor(p2.get_color())
    axs2.tick_params(axis='y', colors=p2.get_color())

    #  resize the heatmap w.r.t the original temproal signal
    last_layer_name= 'tcn_sfm'
    for i in range(num_heatmaps):
        hidden_layer_name='tcn2_conv_0_'+str(i)
        heatmap= CAM_integrated_gradients(model, input_data, hidden_layer_name, last_layer_name)

        shw= axs[i].imshow(np.atleast_2d(heatmap), aspect='auto', vmin=0, vmax=1)
        axs[i].set_xticklabels("")
        axs[i].set_yticklabels("")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # add space for colour bar
    fig.subplots_adjust(right=0.65)
    cbar_ax = fig.add_axes([0.75, 0.55, 0.04, 0.3]) # left,bottom,width,height
    fig.colorbar(shw, cax=cbar_ax)

    plt.show()


def plot_ig_cam_heatmap_lstm(model, input_data,
                        vol_max_tmp=160, vol_min=-10, cur_max_tmp=30, cur_min=0, num_heatmaps=2):
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    # from plot_gradcam import make_gradcam_heatmap
    matplotlib.rcParams.update({'font.size': 7})
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams.update({'axes.linewidth': 1})
    linewidth=1.5

    plt_vol_sig= input_data[:, 0]*(vol_max_tmp - vol_min)+vol_min
    plt_cur_sig= input_data[:, 1]*(cur_max_tmp - cur_min)+cur_min

    height_ratio=[1]*num_heatmaps
    height_ratio.append(6)
    fig, axs= plt.subplots(num_heatmaps+1, 1, 
                           gridspec_kw={'height_ratios':height_ratio,
                                        'hspace':0.2,},
                           figsize=(2,1.5), dpi=300)
    p1, =axs[-1].plot(plt_vol_sig, 'tab:orange', linewidth=linewidth, label='voltage')
    axs[-1].set_ylim([vol_min, vol_max_tmp*2.2])
    axs[-1].set_ylabel("Voltage/ V")
    axs[-1].set_xlabel("Time instance")
    axs2= axs[-1].twinx()
    p2, =axs2.plot(plt_cur_sig, 'tab:green', linewidth=linewidth, label='current')
    axs2.set_ylim([cur_min-1, cur_max_tmp*1.2])
    axs2.set_ylabel("Current/ A")
    axs2.set_xlim([0, len(plt_cur_sig)])
    # set color for edge and ticks
    axs[-1].yaxis.label.set_color(p1.get_color())
    axs[-1].spines["left"].set_edgecolor(p1.get_color())
    axs[-1].tick_params(axis='y', colors=p1.get_color())
    axs2.yaxis.label.set_color(p2.get_color())
    axs2.spines['left'].set_visible(False) # trick!: need to invisual of the ax2' left spine 
    axs2.spines["right"].set_edgecolor(p2.get_color())
    axs2.tick_params(axis='y', colors=p2.get_color())

    #  resize the heatmap w.r.t the original temproal signal
    last_layer_name= 'residual_lstm__sfm'
    for i in range(num_heatmaps):
        if i==0:
            hidden_layer_name='bidirectional'
        elif i==1:
            hidden_layer_name='bidirectional_1'
        heatmap= CAM_integrated_gradients(model, input_data, hidden_layer_name, last_layer_name)

        shw= axs[i].imshow(np.atleast_2d(heatmap), aspect='auto', vmin=0, vmax=1)
        axs[i].set_xticklabels("")
        axs[i].set_yticklabels("")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # add space for colour bar
    fig.subplots_adjust(right=0.65)
    cbar_ax = fig.add_axes([0.75, 0.55, 0.04, 0.3]) # left,bottom,width,height
    fig.colorbar(shw, cax=cbar_ax)

    plt.show()


def plot_ig_cam_heatmap(model_tcn, model_lstm, input_data,
                        vol_max_tmp=160, vol_min=-10, cur_max_tmp=30, cur_min=0, 
                        num_heatmaps_tcn=4, num_heatmaps_lstm=2):
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    # from plot_gradcam import make_gradcam_heatmap
    matplotlib.rcParams.update({'font.size': 7})
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams.update({'axes.linewidth': 1})
    linewidth=1.5

    plt_vol_sig= input_data[:, 0]*(vol_max_tmp - vol_min)+vol_min
    plt_cur_sig= input_data[:, 1]*(cur_max_tmp - cur_min)+cur_min

    height_ratio=[1]*(num_heatmaps_tcn+num_heatmaps_lstm)
    height_ratio.append(6)
    fig, axs= plt.subplots(num_heatmaps_tcn+num_heatmaps_lstm+1, 1, 
                           gridspec_kw={'height_ratios':height_ratio,
                                        'hspace':0.2,},
                           figsize=(2,1.5), dpi=300)
    p1, =axs[-1].plot(plt_vol_sig, 'tab:orange', linewidth=linewidth, label='voltage')
    axs[-1].set_ylim([vol_min, vol_max_tmp*2.2])
    axs[-1].set_ylabel("Voltage/ V")
    axs[-1].set_xlabel("Time instance")
    axs2= axs[-1].twinx()
    p2, =axs2.plot(plt_cur_sig, 'tab:green', linewidth=linewidth, label='current')
    axs2.set_ylim([cur_min-1, cur_max_tmp*1.2])
    axs2.set_ylabel("Current/ A")
    axs2.set_xlim([0, len(plt_cur_sig)])
    # set color for edge and ticks
    axs[-1].yaxis.label.set_color(p1.get_color())
    axs[-1].spines["left"].set_edgecolor(p1.get_color())
    axs[-1].tick_params(axis='y', colors=p1.get_color())
    axs2.yaxis.label.set_color(p2.get_color())
    axs2.spines['left'].set_visible(False) # trick!: need to invisual of the ax2' left spine 
    axs2.spines["right"].set_edgecolor(p2.get_color())
    axs2.tick_params(axis='y', colors=p2.get_color())

    #  resize the heatmap w.r.t the original temproal signal
    last_layer_name= 'tcn_sfm'
    for i in range(num_heatmaps_tcn):
        hidden_layer_name='tcn2_conv_0_'+str(i)
        heatmap= CAM_integrated_gradients(model_tcn, input_data, hidden_layer_name, last_layer_name)
        shw= axs[i].imshow(np.atleast_2d(heatmap), aspect='auto', vmin=0, vmax=1)
        axs[i].set_xticklabels("")
        axs[i].set_yticklabels("")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    #  resize the heatmap w.r.t the original temproal signal
    last_layer_name= 'residual_lstm__sfm'
    for i in range(num_heatmaps_tcn, num_heatmaps_tcn+num_heatmaps_lstm):
        # hidden_layer_name='re_lu_'+str(i+1)
        if i==0:
            hidden_layer_name='bidirectional'
        elif i==1:
            hidden_layer_name='bidirectional_1'
        heatmap= CAM_integrated_gradients(model_lstm, input_data, hidden_layer_name, last_layer_name)

        shw= axs[i].imshow(np.atleast_2d(heatmap), aspect='auto', vmin=0, vmax=1)
        axs[i].set_xticklabels("")
        axs[i].set_yticklabels("")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # add space for colour bar
    fig.subplots_adjust(right=0.65)
    cbar_ax = fig.add_axes([0.75, 0.55, 0.04, 0.3]) # left,bottom,width,height
    fig.colorbar(shw, cax=cbar_ax)

    # fig.tight_layout()
    plt.show()
