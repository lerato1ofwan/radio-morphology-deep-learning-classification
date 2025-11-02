import numpy as np

EPS = 1e-12

def kapur_entropy_threshold(image: np.ndarray) -> int:
    """Compute Kapur's entropy threshold for a uint8 grayscale image.

    Finds the threshold t in [0, 255] that maximizes the sum of entropies
    of the background class [0..t] and the foreground class [t+1..255].

    Returns:
        int: threshold in [0, 255]
    """
    if image.dtype != np.uint8:
        raise ValueError("kapur_entropy_threshold expects uint8 image")

    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    pdf = histogram / histogram.sum()

    # Cumulative distribution function (class weights/masses)
    cdf = np.cumsum(pdf)

    # Safe copy to avoid log(0); 0·log(0) is defined as 0, but we compute with EPS
    pdf_safe = pdf.copy()
    pdf_safe[pdf_safe == 0] = EPS

    # Cumulative entropy H(k) = -sum_{i=0..k} p(i) log p(i)
    cumulative_entropy = -np.cumsum(pdf_safe * np.log(pdf_safe))

    # Entropy sums for background [0..t] and foreground [t+1..255]
    total_entropy = cumulative_entropy[-1]
    background_entropy = cumulative_entropy                # H_b(t)
    foreground_entropy = total_entropy - cumulative_entropy  # H_f(t)

    # Average entropies per class; guard zero weights by replacing with 1 (no effect on argmax)
    cdf_safe = cdf.copy()
    cdf_safe[cdf_safe == 0] = 1.0
    foreground_mass = 1.0 - cdf
    foreground_mass[foreground_mass == 0] = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        background_mean_entropy = background_entropy / cdf_safe
        foreground_mean_entropy = foreground_entropy / foreground_mass
        score = background_mean_entropy + foreground_mean_entropy
        score[np.isnan(score)] = -np.inf

    threshold = int(np.argmax(score))
    return threshold
