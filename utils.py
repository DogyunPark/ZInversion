import torch
import numpy as np
import torch.nn as nn
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional
import os
from PIL import Image
import json
from zipfile import ZipFile
import io

from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

import torchvision.transforms.functional as F
#from .functional import InterpolationMode, _interpolation_modes_from_int


def get_cw_VR_loss(activation, batch_label, num_labels, is_normalized = True):
    
    label_variances = []
    for label in range(num_labels):
        lists = [i for i,c in enumerate(batch_label) if c == label]
        label_act = activation[lists]
        label_mean = label_act.mean(dim = 0)
        label_var = torch.var(label_act, dim = 0, unbiased = False)
        label_variances.append(torch.sum(label_var))
    '''    
    lists = [i for i,c in enumerate(batch_label) if c == 1]
    label_act = activation[lists]
    label_mean = label_act.mean(dim = 0)
    label_var = torch.var(label_act, dim = 0, unbiased = False)
    label_variances.append(torch.sum(label_var))
    '''

    if is_normalized:
        _, channel, width, height = activation.shape
        num_units = channel * width * height 
        normalization_factor = num_units * num_labels

        return torch.mul(1. / normalization_factor, sum(label_variances))

    else:
        return sum(label_variances)
            


class InversionActivationHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.featuremap_hook)

    def featuremap_hook(self, module, input, output):
        self.temp = output

    def close(self):
        self.hook.remove()

   


def rs_maker(model, shape):
    rs_forward_hook(model)

    cache_input = torch.randn(*shape).cuda()
    with torch.no_grad():
        __ = model(cache_input)

    random_shift_factor = compute_rs(model)

    torch.cuda.empty_cache()
    model.apply(remove_hook_function)
    return random_shift_factor



def rs_forward_hook(model):
    for module in model.modules():
        module.register_forward_hook(rs_maker_hook)


def compute_rs(model):
    rs_list = []
    rs_list2 = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            rs_list.append(module.rs)
            rs_list2.append(module.rs2)
    return rs_list, rs_list2


def rs_maker_hook(module, inputs, outputs):
    if isinstance(module, nn.BatchNorm2d):
        nch = outputs.shape[1]
        choices = [0.05, 0, -0.05]
        choices2 = [0.001, 0, -0.001]
        np_rand = np.random.choice(choices, nch, p =[0.3,0.4,0.3])
        np_rand2 = np.random.choice(choices2, nch, p = [0.3,0.4,0.3])
        rs = torch.from_numpy(np_rand).cuda()
        rs2 = torch.from_numpy(np_rand2).cuda()
        module.rs = rs
        module.rs2 = rs2
    
def remove_hook_function(module):
    if hasattr(module, 'rs'):
        delattr(module, 'rs')



def denormalize(image_tensor, dataset = 'cifar10'):
    if dataset == 'cifar10':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    elif dataset == 'imagenet':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:,c] * s + m, 0, 1)

    return image_tensor



def matrix_loss(m1, num_classes = 10):
    batch, dims, h, w = m1.shape

    matrix = m1.view(batch, dims, -1).permute(0,2,1).contiguous()
    #matrix = m1.view(batch, dims, -1)
    matrix = matrix.view(-1, dims)
    #matrix = matrix.view(-1, h*w)

    matrix_score = torch.mm(matrix, matrix.t()) # (batch_size * H * W, batch_size * H * W)
    #loss = torch.norm(matrix_score, 1)
    
    mask = torch.eye(10)[:,:,None, None].expand(-1,-1, h*w*10, h*w*10)
    mask = mask.permute(0,2,1,3).contiguous().reshape(h*w*100, h*w*100)
    mask = mask.to(matrix_score.device)
    matrix_score = matrix_score * mask
    
    mask2 = torch.eye(10)[:,:,None, None].expand(-1,-1,h*w,h*w)
    mask2 = mask2.permute(0,2,1,3).contiguous().reshape(h*w*10, h*w*10)
    mask2 = torch.cat([mask2] * 10, dim = 1)
    mask2 = torch.cat([mask2] * 10, dim = 0)
    mask2 = 1 - mask2
    mask2 = mask2.to(matrix_score.device)
    matrix_score = matrix_score * mask2

    mask3 = torch.eye(h*w)
    mask3 = torch.cat([mask3] * 100, dim = 1)
    mask3 = torch.cat([mask3] * 100, dim = 0)
    mask3 = mask3.to(matrix_score.device)
    matrix_score = matrix_score * mask3

    #return loss
    #matrix_score = matrix_score.view(batch, dims, batch, dims)#.permute(0,2,1,3).contiguous()
    loss = torch.norm(matrix_score, 1)
    '''
    for i in range(num_classes):
        if i != 9:
            B = matrix_score[6*i : 6*i+6,:, 6*i : 6*i + 6]
            for j in range(6):
                for k in range(6):
                    if j != k:
                        loss += torch.norm(torch.diag(B[j,:,k]), 1)
        else:
            B = matrix_score[60:, :, 60:]
            for j in range(4):
                for k in range(4):
                    if j != k:
                        loss += torch.norm(torch.diag(B[j,:,k]), 1)
    '''

    return loss



def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch +1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e /es)) * base_lr
        return lr
    return lr_policy(_lr_fn)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomCrop(torch.nn.Module):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a sequence of length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.shape[2], img.shape[3]
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = Pad(img, self.padding, self.fill, self.padding_mode)

        width, height = img.shape[2], img.shape[3]
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = Pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = Pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return Crop(img, i, j, h, w)


def Crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:

    return img[..., top:top + height, left:left + width]


def Pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if isinstance(padding, int):
        if torch.jit.is_scripting():
            # This maybe unreachable
            raise ValueError("padding can't be an int while torchscripting, set it as a list [value, ]")
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == "edge":
        # remap padding_mode str
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        # route to another implementation
        return _pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (torch.float32, torch.float64):
        # Here we temporary cast input tensor to float
        # until pytorch issue is resolved :
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        img = img.to(torch.float32)

    img = torch_pad(img, p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        img = img.to(out_dtype)

    return img



'''
def _get_image_size(img: Tensor) -> List[int]:
    """Returns image size as [w, h]
    """
    if isinstance(img, torch.Tensor):
        return F_t._get_image_size(img)
'''


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, gpu):
        self.n_holes = n_holes
        self.length = length
        self.gpu = gpu

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        b = img.size(0)
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        mask = mask.to('cuda:0')
        img = img * mask

        return img



class ZipCreator(object):
    def __init__(self, name='', precision=32, overlay=None):
        #self.file = os.path.abspath(file)
        #self.out_file = './cifar10_{}_feature{}.zip'.format(precision, feature_layer)
        self.name = os.path.basename(file) if name == '' else name
        self.precision = precision
        self.overlay = overlay

    def create8_(self, data, zip_file, metadata):
        uint8_max = np.iinfo(np.uint8).max
        global_max = data.reshape(-1).max()
        global_min = data.reshape(-1).min()
        data = np.round((data - global_min) / (global_max - global_min) * uint8_max).astype(np.uint8)

        # Add missing feature channels to make shape[2] divisible by 4.
        if data.shape[2] % 4 != 0:
            zeros = np.zeros((data.shape[0], data.shape[1], 4 - (data.shape[2] % 4)), dtype=np.uint8)
            data = np.concatenate((data, zeros), axis=2)

        splits = np.split(data, data.shape[2] // 4, axis=2)
        for i, split in enumerate(splits):
            filename = '{}.png'.format(i)
            bytes_io = io.BytesIO()
            Image.fromarray(split).save(bytes_io, format='png')
            zip_file.writestr(filename, bytes_io.getvalue())

    def create16_(self, data, zip_file, metadata):
        uint16_max = np.iinfo(np.uint16).max
        global_max = data.reshape(-1).max()
        global_min = data.reshape(-1).min()
        data = np.round((data - global_min) / (global_max - global_min) * uint16_max).astype(np.uint16)

        # Add missing feature channels to make shape[2] divisible by 2.
        if data.shape[2] % 2 != 0:
            zeros = np.zeros((data.shape[0], data.shape[1], 2 - (data.shape[2] % 2)), dtype=np.uint16)
            data = np.concatenate((data, zeros), axis=2)

        splits = np.split(data, data.shape[2] // 2, axis=2)
        for i, split in enumerate(splits):
            # Convert 2x uint16 to 4x uint8
            buf = np.array(split, dtype=np.uint16)
            split = np.frombuffer(buf.tobytes(), dtype=np.uint8).reshape(data.shape[0], data.shape[1], 4)
            filename = '{}.png'.format(i)
            bytes_io = io.BytesIO()
            Image.fromarray(split).save(bytes_io, format='png')
            zip_file.writestr(filename, bytes_io.getvalue())

    def create32_(self, data, zip_file, metadata):
        uint32_max = np.iinfo(np.uint32).max
        global_max = data.reshape(-1).max()
        global_min = data.reshape(-1).min()
        data = np.round((data - global_min) / (global_max - global_min) * uint32_max).astype(np.uint32)

        for i in range(data.shape[2]):
            # Convert 1x uint32 to 4x uint8
            split = np.frombuffer(data[:, :, i].tobytes(), dtype=np.uint8).reshape(data.shape[0], data.shape[1], 4)
            filename = '{}.png'.format(i)
            bytes_io = io.BytesIO()
            Image.fromarray(split).save(bytes_io, format='png')
            zip_file.writestr(filename, bytes_io.getvalue())

    def create(self, image, feature_layer):
        #if self.file.endswith('.npy'):
        #    data = np.load(self.file)
        #else:
        #    with np.load(self.file) as npz_file:
        #        data = npz_file[npz_file.files[0]]

        data = image
        if data.ndim != 3:
            raise ValueError('The input data has an unexpected number of dimensions ({} given, 3 expected).'.format(data.ndim))
        out_file = './cifar10_{}_feature{}.zip'.format(self.precision, feature_layer)
        zip_file = ZipFile(out_file, 'w')
        has_overlay = self.overlay != None
        metadata = {
            'name': self.name,
            'width': data.shape[1],
            'height': data.shape[0],
            'features': data.shape[2],
            'precision': self.precision,
            'overlay': has_overlay,
        }

        if has_overlay:
            bytes_io = io.BytesIO()
            Image.open(self.overlay).save(bytes_io, format='jpeg', quality=85)
            zip_file.writestr('overlay.jpg', bytes_io.getvalue())

        if self.precision == 32:
            self.create32_(data, zip_file, metadata)
        elif self.precision == 16:
            self.create16_(data, zip_file, metadata)
        else:
            self.create8_(data, zip_file, metadata)

        zip_file.writestr('metadata.json', json.dumps(metadata))
        zip_file.close()
