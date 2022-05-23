import os
import pickle
import argparse
import json
import pdb

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np

coco_classes = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def eval_detection_metrics(det_csv_file, gt_file, mapping_needed = False, disc_results = None):
    if mapping_needed:
        assert disc_results != None, "Discovery results is needed"
        gt = json.load(open(gt_file,'rb'))
        class_mapping = {idx: ele['id'] for idx, ele in enumerate(gt['categories'])}
    pred_df = pd.read_csv(det_csv_file)
    all_results = []
    for idx, row in pred_df.iterrows():
        res = {}

        res['image_id'] = row['image_id']
        if mapping_needed:
            disc_cluster_assign = disc_results['class_name'][row['category_id']]
            coco_class_assign = coco_classes.index(disc_cluster_assign)-1
            coco_category_id = class_mapping[coco_class_assign]
        else:
            coco_category_id = row['category_id']
        res['category_id'] = coco_category_id
        res['bbox'] = [row['x'], row['y'], row['w'], row['h']]
        res['score'] = row['score']

        all_results.append(res)
        
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(all_results)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0], cocoEval.stats[1]