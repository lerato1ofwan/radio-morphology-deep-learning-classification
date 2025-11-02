import numpy as np
import cv2

def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def edge_density(gray: np.ndarray, low: int = 50, high: int = 150) -> float:
    edges = cv2.Canny(gray, low, high)
    return float(np.count_nonzero(edges)) / float(gray.size)
