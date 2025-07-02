import json
import query as q
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import argparse
import warnings



def detect_approx_kink(loss_curve, window_length=11, polyorder=2, 
                       percentile_threshold=20, lookahead=5, plot=True, block=True, start_epoch=0):
    """
    Approximate kink detector for training loss curve.
    
    Args:
        loss_curve: Array or list of training loss values (per epoch).
        window_length: Smoothing window length for Savitzky-Golay filter (must be odd).
        polyorder: Polynomial order for Savitzky-Golay filter.
        percentile_threshold: Percentile (of smoothed derivative) to define approximate strong negative slope.
        lookahead: Number of epochs to confirm sustained trend.
        plot: Whether to plot curves and suggested stopping region.
        
    Returns:
        Suggested approximate stopping epoch index (int).
    """
    loss_curve = np.array(loss_curve)
    deriv = np.diff(loss_curve)

    inverted_deriv = -deriv
    peaks, _ = find_peaks(inverted_deriv[start_epoch:], prominence=0.005) 

    # Smooth derivative
    smooth_deriv = savgol_filter(deriv, window_length=window_length, polyorder=polyorder)
    smooth_2deriv = np.diff(smooth_deriv)

    # Define approximate threshold using percentile
    neg_thresh = np.percentile(smooth_deriv[start_epoch:], percentile_threshold)

    # Find first point after early plateau where derivative drops below approximate threshold
    kink_epoch = None
    for i in range(start_epoch, len(smooth_deriv) - lookahead):
        if np.all((smooth_deriv[i+2:i+2+lookahead] < neg_thresh) & (smooth_2deriv[i+1:i+1+lookahead] < 0)):
            kink_epoch = int(i + 1 + (lookahead / 2))
            break
    """for i in range(len(smooth_deriv) - lookahead):
        if np.all(smooth_deriv[i+1:i+1+lookahead] < neg_thresh):
            kink_epoch = int(i + 1 + (lookahead / 2))
            break"""

    if kink_epoch is None:
        warnings.warn("kink_epoch is None! Consider changing the parameters.")
        return kink_epoch
        
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    epochs = np.array(list(range(len(loss_curve))))

    # Mark local minima
    #ax1.plot(epochs[peaks], deriv[peaks], "x", color='blue', label='Local minima (peaks in -deriv)')
    
    ax1.plot(epochs, loss_curve, label='Training Loss', color='tab:blue')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # Enable grid on ax1 (primary axis)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs[1:], smooth_deriv, label='Smoothed dLoss/dEpoch', color='tab:orange', linewidth=2)
    ax2.set_ylabel("dLoss/dEpoch", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    ax1.minorticks_on()
    ax2.minorticks_on()
    
    if kink_epoch:
        ax1.axvline(kink_epoch, color='red', linestyle='--', label=f'Approx stop ~ {kink_epoch}')
        ax2.axvline(kink_epoch, color='red', linestyle='--')

    fig.suptitle("Approximate Kink Detection")
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
    plt.show(block=block)
        
    return kink_epoch

def main(args):
    start_epoch = args.start_epoch
    list_of_dicts = []
    fn = args.records_file
    with open(fn, "r") as f:
        for line in f:
            d = json.loads(line[:-1])
            list_of_dicts.append(d)

    #print(list_of_dicts[0].keys())

    records = q.Q(list_of_dicts[:-1])

    v_acc = records.select('val_acc1')
    te_acc = records.select("test_acc1")
    epochs = records.select("epoch")
    tr_loss = records.select('train_loss')
    v_loss = records.select('val_loss')
    te_loss = records.select("test_loss")

    #---------------------------------------------------------------------------------

    # Example training loss curve (replace with your actual loss list)
    loss_curve = tr_loss
    approx_stop_epoch = None
    approx_stop_epoch = detect_approx_kink(loss_curve, block=False, percentile_threshold=args.percentile_threshold, lookahead=args.lookahead, start_epoch=start_epoch)
    print(f"Approximate suggested stopping epoch: {approx_stop_epoch}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, v_acc, label='val')
    plt.plot(epochs, te_acc, label='test')
    if approx_stop_epoch is not None:
        plt.axvline(approx_stop_epoch, color='red', linestyle='--', label=f'Approx stop ~ {approx_stop_epoch}')
        plt.axvline(approx_stop_epoch, color='red', linestyle='--')
    # Enable grid on ax1 (primary axis)
    ax1 = plt.gca()  # or your existing axis
    ax1.minorticks_on()
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.3)

    # Add labels and title
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Val & Test Accuracies')
    plt.legend()

    # Show the plot
    plt.show(block=False)

    #---------------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, tr_loss, label='train')
    plt.plot(epochs, v_loss, label='val')
    plt.plot(epochs, te_loss, label='test')
    if approx_stop_epoch is not None:
        plt.axvline(approx_stop_epoch, color='red', linestyle='--', label=f'Approx stop ~ {approx_stop_epoch}')
        plt.axvline(approx_stop_epoch, color='red', linestyle='--')
    # Enable grid on ax1 (primary axis)
    ax1 = plt.gca()  # or your existing axis
    ax1.minorticks_on()
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.3)

    # Add labels and title
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train, Val & Test Losses')
    plt.legend()

    # Show the plot
    plt.show(block=False)

    #-------------------------------------------------------------------------

    n = 1
    d_tr_loss = np.diff(np.array(tr_loss), n=n)
    d_val_loss = np.diff(np.array(v_loss), n=n)
    if n==1:
        n_str = ''
    elif n==2:
        n_str = '^2'

    # Smooth derivative
    d_tr_loss_lpf = savgol_filter(d_tr_loss, window_length=9, polyorder=2)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs[n:], d_tr_loss, alpha=0.5, label='Raw')
    plt.plot(epochs[n:], d_tr_loss_lpf, label='LPF', linewidth=2)
    if approx_stop_epoch is not None:
        plt.axvline(approx_stop_epoch, color='red', linestyle='--', label=f'Approx stop ~ {approx_stop_epoch}')
        plt.axvline(approx_stop_epoch, color='red', linestyle='--')
    # Enable grid on ax1 (primary axis)
    ax1 = plt.gca()  # or your existing axis
    ax1.minorticks_on()
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.3)

    # Add labels and title
    plt.xlabel('epoch')
    plt.ylabel(f"d{n_str} Loss")
    plt.title(f"d{n_str} Loss/dEpoch{n_str} Training")
    plt.legend()

    # Show the plot
    plt.show(block=True)
    #-------------------------------------------------------------------------
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DG overfitting.")
    parser.add_argument("--records_file", type=str)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--percentile_threshold", type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=5)

    args = parser.parse_args()

    main(args)
