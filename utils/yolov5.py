from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_img_size, check_requirements, set_logging

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
        img_transform = ImageTransformGPU(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    else:
        img_transform = ImageTransform(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    with torch.no_grad():
        data = _prepare_data(img, img_transform, cfg, device)
        result = model(return_loss=False, rescale=True, numpy_res=numpy_res, decode_mask=decode_mask, **data) 
        zc_cfg = cfg.data.test.zoom_crop
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