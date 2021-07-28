from skimage import metrics


class Loss:
    def __init__(self, img_reference: np.ndarray, img_query: np.ndarray):
        self.reference_img = img_reference
        self.query_img = img_query 
        
    def calc_PSNR(self):
        """
        calculate Peak Signal Noise Ratio for both images
        """
        return metrics.peak_signal_noise_ratio(self.reference_img, self.query_img)
        
    def calc_SSIM(self):
        """
        calculate Structural Similarity between two images
        """
        mean_ssim, ssim_gradient, ssim_img =  metrics.structural_similarity(self.reference_img, self.query_img)
        return mean_ssim