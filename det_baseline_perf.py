import numpy as np
from time import perf_counter
import torch
import argparse, json
from tqdm import tqdm

# vision specific
from pycocotools.coco import COCO

# benchmark toolkit API
from sap_toolkit.client import EvalClient

# utility functions for mmdetection
from utils import print_stats
from utils.mmdet import init_detector, inference_detector, parse_det_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--eval-config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--dynamic-schedule', action='store_true', default=False)
    opts = parser.parse_args()
    return opts

def get_model(opts):
    model = init_detector(opts)
    model.eval()

    # warm up the GPU
    _ = inference_detector(model, np.zeros((1920, 1200, 3), np.uint8))
    torch.cuda.synchronize()
    return model

def detector_process():
    # initialize model and mapping
    opts = parse_args()
    config = json.load(open(opts.eval_config, 'r'))

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    
    print("Loading model")
    model = get_model(opts)

    eval_client = EvalClient(config, verbose=False)

    runtime_all = []

    # dynamic scheduling
    if opts.dynamic_schedule:
        # initialize runtime mean to 0.0
        mean_rtf = 0

    for seq in tqdm(seqs):

        # Request stream for current sequence from benchmark toolkit
        eval_client.request_stream(seq)

        fid = 0
        fid_latest = None

        # get the time when stream's first frame was received
        t_start = eval_client.get_stream_start_time()

        runtime = []

        count_detections = 1
        while fid is not None:
 
            t1 = perf_counter()
            t_elapsed = t1 - t_start

            # identify latest available frame
            fid, frame = eval_client.get_frame()

            # for dynamic scheduling
            fid_continous = t_elapsed*opts.fps

            if fid == fid_latest:
                # algorithm is fast and has some idle time
                wait_for_next = True
            else:
                wait_for_next = False
                if opts.dynamic_schedule:
                    if mean_rtf >= 1: # when runtime < 1, it should always process every frame
                        fid_remainder = fid_continous - fid
                        if mean_rtf < np.floor(fid_remainder + mean_rtf):
                            # wait till next frame
                            wait_for_next = True

            if wait_for_next:
                continue

            # run model on current frame 
            t_det_start = perf_counter()
            result = inference_detector(model, frame, gpu_pre=not opts.cpu_pre)
            t_det = perf_counter() - t_det_start

            # calculate new runtime mean for dynamic scheduling
            if opts.dynamic_schedule:
                sum_rtf = mean_rtf*count_detections + t_det*opts.fps
                count_detections += 1
                mean_rtf = sum_rtf/count_detections

            fid_latest = fid

            # bboxes, bbox_scores, labels are numpy arrays of dimensions Nx4, Nx1, Nx1   
            bboxes, bbox_scores, labels, _ = parse_det_result(result, coco_mapping, n_class)
            torch.cuda.synchronize()

            # send result to benchmark toolkit
            # The toolkit requires three inputs to send results
            # which are numpy arrays of dimensions Nx4, Nx1, Nx1 representing
            # bounding box coordinates, bounding box scores, and labels respectively
            # where N is the number of detections (100 detections per frame maximum)
            eval_client.send_result_to_server(bboxes, bbox_scores, labels)

            # get next frame
            fid, frame = eval_client.get_frame()
            runtime.append(perf_counter() - t1)
        
        eval_client.stop_stream()
        print("Finished seq ", seq)

        runtime_all += runtime

        runtime = np.asarray(runtime)
        s2ms = lambda x: 1e3*x
        print_stats(runtime, 'Runtime per frame for seq ' + str(seq) + '(ms)', cvt=s2ms)

    runtime_all_np = np.asarray(runtime_all)
    # convert to ms for display
    s2ms = lambda x: 1e3*x
    print_stats(runtime_all_np, 'Runtime (ms)', cvt=s2ms)
    eval_client.close()


if __name__ == '__main__':
    detector_process()

