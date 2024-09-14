import torch
import numpy as np

def digitize(probabilities, bin_boundaries):
    # Map probabilities to intervals
    digitized = torch.zeros_like(probabilities, dtype=torch.long)
    for i in range(1, len(bin_boundaries)):
        digitized += probabilities >= bin_boundaries[i]
    return digitized


def emce(probabilities, true_labels, num_bins=10):
    # Divide the prediction probability into multiple probability intervals
    probabilities = torch.tensor(probabilities)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_indices = digitize(probabilities, bin_boundaries)
   
    ece = 0.0
    mce = 0.0
    for bin_idx in range(1, num_bins + 1):
        # Calculate the average probability and the proportion of actual observations within each interval
        bin_mask = bin_indices == bin_idx
        bin_indices = torch.nonzero(bin_mask).squeeze()
        if bin_mask.any():
            bin_true_labels = true_labels[bin_indices]
            bin_probabilities = probabilities[bin_indices]
            bin_accuracy = torch.mean(bin_true_labels.float() == (bin_probabilities > 0.5).float())
            bin_confidence = torch.abs(bin_probabilities - bin_accuracy)
            bin_weight = len(bin_true_labels) / len(true_labels)
            ece += bin_weight * bin_confidence.item()

            bin_calibration_error = torch.max(torch.abs(bin_probabilities - bin_accuracy)).item()
            mce = max(mce, bin_calibration_error)

    return ece, mce



def compute_mce(probabilities, true_labels, num_bins=10):
    # Divide the prediction probability into multiple probability intervals
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_indices = torch.digitize(probabilities, bin_boundaries)

    max_calibration_error = 0.0
    for bin_idx in range(1, num_bins + 1):
        # Calculate the maximum difference within each interval
        bin_mask = bin_indices == bin_idx
        if bin_mask.any():
            bin_true_labels = true_labels[bin_mask]
            bin_probabilities = probabilities[bin_mask]
            bin_accuracy = torch.mean(bin_true_labels.float() == (bin_probabilities > 0.5).float())
            bin_calibration_error = torch.max(torch.abs(bin_probabilities - bin_accuracy)).item()
            max_calibration_error = max(max_calibration_error, bin_calibration_error)

    return max_calibration_error


def calc_bins(preds, labels_oneh):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned==bin]).sum() / (bin_sizes[bin] + 1e-10)
            bin_confs[bin] = (preds[binned==bin]).sum() / (bin_sizes[bin] + 1e-10)

    return bins, binned, bin_accs, bin_confs, bin_sizes



def get_CE_metrics(preds, labels_oneh):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / (sum(bin_sizes) + 1e-10)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE
