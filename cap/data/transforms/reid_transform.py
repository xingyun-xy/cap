import numpy as np

from cap.registry import OBJECT_REGISTRY

__all__ = ["RandomErasing", "CenterJitter", "ReIDTransform"]


@OBJECT_REGISTRY.register
class ReIDTransform:
    """
    Doing transform for the dataset.

    This class applies the augmentation process.

    Parameters
    ----------
    input_size : tuple
                size of network input
    input_mean : tuple
                 the mean of the dataset
    is_norm : boolean
                whether to norm data
    random_erase : boolean
                 whether or not use random erasing
    random_mirror : boolean
                 whether or not use random mirror
    erase_mode : string
                 mode to use int random erasing methods.
                 Candidates are 'mean', 'white', and 'noise'.
    erase_prob : float
                 probability to perform random erasing
    erase_ratio_range : tuple
                 erase ratio range to the input image
    erase_aspect_ratio_range : tuple
                 erase region aspect ratio range
    """

    def __init__(
        self,
        input_size=(128, 128),
        input_mean=(128, 128, 128),
        input_scale=0.0078125,
        is_norm=True,
        random_erase=False,
        random_mirror=False,
        jitter_center=False,
        erase_mode="mean",
        erase_prob=0.5,
        erase_ratio_range=(0.02, 0.4),
        erase_aspect_ratio_range=(0.3, 3.33),
        jitter_center_scale=0.25,
    ):

        self._input_size = input_size
        self._is_norm = is_norm
        self._input_mean = np.array(input_mean, dtype=np.float32)
        self._input_mean = np.reshape(self._input_mean, (1, 1, 3))
        self._input_scale = input_scale
        if random_erase:
            self._eraser = RandomErasing(
                erase_mode=erase_mode,
                mean_value=np.array(input_mean),
                prob=erase_prob,
                erase_ratio_range=erase_ratio_range,
                aspect_ratio_range=erase_aspect_ratio_range,
            )
        else:
            self._eraser = None
        if jitter_center:
            self._center_jitter = CenterJitter(
                input_size=input_size,
                shift_scale=jitter_center_scale,
                padding_value=(0, 0, 0),
            )
        else:
            self._center_jitter = None

        self._random_mirror = random_mirror

    def __call__(self, data):
        img_yuvbytes = data["img"]
        label_dict = data["anno"]
        gid = label_dict["id"]
        img = np.frombuffer(img_yuvbytes, dtype=np.uint8)
        img = np.reshape(img, (self._input_size[0], self._input_size[1], 3))
        img = img.copy()
        if self._eraser is not None:
            img = self._eraser(img)
        if self._center_jitter is not None:
            img = self._center_jitter(img)
        if self._random_mirror:
            if_mirror = np.random.choice([True, False], p=[0.5, 0.5])
            if if_mirror:
                img = np.fliplr(img)
        img = img.astype(np.float32)
        if self._is_norm:
            img = (img - self._input_mean) * self._input_scale
        img = np.transpose(img, (2, 0, 1))
        return {"labels": gid, "img": img}


class CenterJitter:
    """
    Randomly shift images.

    Parameters
    -----------
    input_size : tuple
        size of the input image
    padding_value : tuple
        when shift out of the edge, values to pad
    shift_scale : float
        the ratio to input_size, used to control maximum shift values
    """

    def __init__(
        self, input_size=(128, 128), padding_value=(0, 0, 0), shift_scale=0.25
    ):
        self._input_size = input_size
        self._shift_scale = shift_scale
        self._prealloc = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.float32
        ) * np.array(padding_value)

    def __call__(self, img):
        shift_x = int(
            np.random.normal(
                0, self._input_size[0] * self._shift_scale / 6.0, 1
            )
        )
        shift_y = int(
            np.random.normal(
                0, self._input_size[1] * self._shift_scale / 6.0, 1
            )
        )
        x1 = max(0, shift_x)
        y1 = max(0, shift_y)
        x2 = min(self._input_size[0], self._input_size[0] + shift_x)
        y2 = min(self._input_size[1], self._input_size[1] + shift_y)

        new_img = self._prealloc.copy()
        new_img[y1 : y2 + 1, x1 : x2 + 1] = img[y1 : y2 + 1, x1 : x2 + 1]

        return new_img


class RandomErasing:
    """
    Apply random erasing data augmentation.

    Randomly erase the image pixels.

    Parameters
    ----------
    erase_mode : string
                 Mode to perform erasing. Candidate are 'mean', 'white' and
                 'noise'. 'mean' means using ``mean_value``, 'white' means
                 using white color, and 'noise' means using noise to fill
                 erased the regions.
    mean_value : tuple
                 mean of the image or the dataset
    prob : float
           probability to perform erasing
    erase_ratio_range : tuple
           erase ratio range to the input image
    aspect_ratio_range : tuple
           erase region aspect ratio range
    """

    def __init__(
        self,
        erase_mode="mean",
        mean_value=(128, 128, 128),
        prob=0.5,
        erase_ratio_range=(0.02, 0.4),
        aspect_ratio_range=(0.3, 3.33),
    ):
        self._erase_mode = erase_mode
        self._prob = prob
        self._mean = mean_value
        self._erase_ratio_list = np.logspace(
            np.log10(erase_ratio_range[0]),
            np.log10(erase_ratio_range[1]),
            num=100,
        )
        self._aspect_ratio_list = np.logspace(
            np.log10(aspect_ratio_range[0]),
            np.log10(aspect_ratio_range[1]),
            num=100,
        )

    def __call__(self, img):
        is_erase = np.random.choice(
            [True, False], 1, p=[self._prob, 1 - self._prob]
        )
        if is_erase:
            h, w = img.shape[:2]
            area = w * h
            erase_ratio = np.random.choice(self._erase_ratio_list)
            area_e = erase_ratio * area

            aspect_ratio = np.random.choice(self._aspect_ratio_list)
            h_e = max(1.0, np.sqrt(area_e * aspect_ratio))
            w_e = min(int(area_e / h_e), w - 1)
            h_e = min(int(h_e), h - 1)

            x_e = self._choice_loc(w_e, w)
            y_e = self._choice_loc(h_e, h)
            erase_data = np.array([0, 0, 0], dtype=img.dtype)
            if self._erase_mode == "mean":
                erase_data = np.array(self._mean, dtype=img.dtype)
            elif self._erase_mode == "white":
                erase_data = np.array([255, 255, 255], dtype=img.dtype)
            elif self._erase_mode == "noise":
                erase_data = np.random.choice(256, 3)
                erase_data = erase_data.astype(img.dtype)
            img[y_e : y_e + h_e, x_e : x_e + w_e] = erase_data
        return img

    def _choice_loc(self, value, max_value):
        loc = np.random.choice(max_value - value)
        return loc
