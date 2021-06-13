from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_img_size, check_requirements, set_logging

import mmcv
from mmcv.runner import load_checkpoint
import torch
import torch.nn.functional as F

class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose and move to GPU
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True, 
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = False # ignores input, assuming already in RGB
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True, device='cuda:0'):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device).unsqueeze(0)

        return img, img_shape, pad_shape, scale_factor

class ImageTransformGPU(object):
    """Preprocess an image.
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.std_inv = 1/self.std
        # self.to_rgb = to_rgb, assuming already in RGB
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True, device='cuda:0'):
        h, w = img.shape[:2]
        if keep_ratio:
            if isinstance(scale, (float, int)):
                if scale <= 0:
                    raise ValueError(
                         'Invalid scale {}, must be positive.'.format(scale))
                scale_factor = scale
            elif isinstance(scale, tuple):
                max_long_edge = max(scale)
                max_short_edge = min(scale)
                scale_factor = min(max_long_edge / max(h, w),
                                max_short_edge / min(h, w))
            else:
                raise TypeError(
                    'Scale must be a number or tuple of int, but got {}'.format(
                        type(scale)))
            
            new_size = (round(h*scale_factor), round(w*scale_factor))
        else:
            new_size = scale
            w_scale = new_size[1] / w
            h_scale = new_size[0] / h
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = (*new_size, 3)

        img = torch.from_numpy(img).to(device).float()
        # to BxCxHxW
        img = img.permute(2, 0, 1).unsqueeze_(0)

        if new_size[0] != img.shape[1] or new_size[1] != img.shape[2]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore the align_corner warnings
                img = F.interpolate(img, new_size, mode='bilinear')
        if flip:
            img = torch.flip(img, 3)

        for c in range(3):
            img[:, c, :, :].sub_(self.mean[c]).mul_(self.std_inv[c])

        if self.size_divisor is not None:
            pad_h = int(np.ceil(new_size[0] / self.size_divisor)) * self.size_divisor - new_size[0]
            pad_w = int(np.ceil(new_size[1] / self.size_divisor)) * self.size_divisor - new_size[1]
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            pad_shape = (img.shape[2], img.shape[3], 3)
        else:
            pad_shape = img_shape
        return img, img_shape, pad_shape, scale_factor

def _prepare_data(img, img_transform, cfg, device):
    zc_cfg = cfg.data.test.zoom_crop
    if zc_cfg is not None:      
        img = img[zc_cfg['y']: zc_cfg['y'] + zc_cfg['h']]
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True),
        device=device,
    )
    # for update in bbox_head.py
    if type(scale_factor) is int:
        scale_factor = float(scale_factor)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_metas=[img_meta])

def init_detector(opts, device='cuda:0', imgsz = 1920, half = False):
    check_requirements(exclude=('tensorboard', 'thop'))

    weights = opts.weights 
    
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA


    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    
    ##model.to(device)
    model.eval()
    return model

def inference_detector(model, img, gpu_pre=True, numpy_res=True, decode_mask=True):
    # assume img has RGB channel ordering instead of BGR
    cfg = model.cfg
    if gpu_pre:
        img_transform = ImageTransformGPU()
            #size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg) #TODO
    else:
        img_transform = ImageTransform()
            #size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    with torch.no_grad():
        data = _prepare_data(img, img_transform, cfg, device)# TODO
        result = model(return_loss=False, rescale=True, numpy_res=numpy_res, decode_mask=decode_mask, **data) 
        zc_cfg = cfg.data.test.zoom_crop # TODO
        if zc_cfg is not None and len(result[0]):
            result[0][:, [1, 3]] += zc_cfg['y']
    return result


def parse_det_result(result, class_mapping=None, n_class=None, separate_scores=True, return_sel=False):
    if len(result) > 2:
        bboxes_scores, labels, masks = result
    else:
        bboxes_scores, labels = result
        masks = None

    if class_mapping is not None:
        labels = class_mapping[labels]
        sel = labels < n_class
        bboxes_scores = bboxes_scores[sel]
        labels = labels[sel]
        if masks is not None:
            masks = masks[sel]
    else:
        sel = None
    if separate_scores:
        if len(labels):
            bboxes = bboxes_scores[:, :4]
            scores = bboxes_scores[:, 4]
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        outs = [bboxes, scores, labels, masks]
    else:
        outs = [bboxes_scores, labels, masks]
    if return_sel:
        outs.append(sel)
    return tuple(outs)