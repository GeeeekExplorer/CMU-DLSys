import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = img[:, ::-1]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        pad = self.padding
        shift_x, shift_y = np.random.randint(low=0, high=2*self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        pad_width = ((pad, pad), (pad, pad), (0, 0)) if img.ndim == 3 else ((pad, pad), (pad, pad))
        h, w = img.shape[:2]
        img = np.pad(img, pad_width)
        return img[shift_x:h+shift_x, shift_y:w+shift_y]
        ### END YOUR SOLUTION
