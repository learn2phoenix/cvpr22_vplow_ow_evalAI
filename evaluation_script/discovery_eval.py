from collections import Counter, defaultdict
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

pascal_indices = [2, 5, 15, 9, 40, 6, 3, 16, 57, 20, 61, 17, 18, 4, 1, 59, 19, 58, 7, 63]
pascal_classes = [coco_classes[i] for i in pascal_indices]


def get_auc(purity, coverage):
    coverage_append = [0] + coverage
    coverage_diff = np.diff(coverage_append)
    return np.sum(purity * coverage_diff)


def get_knwn_sorted_purity_coverage(dump, marker):
    pur = []
    cov = []
    for i in range(marker):
        if dump['res']['class_name'][i] != '__background__' and dump['res']['class_name'][i] in pascal_classes:
            pur.append(dump['res']['purity'][i])
            cov.append(dump['res']['coverage'][i][1][np.in1d(np.arange(81), pascal_indices)])
    pur_sort = np.sort(pur)[::-1]
    pur_sort_ord = np.argsort(pur)[::-1]
    cov_sort = [cov[i] for i in pur_sort_ord]
    pur_cum = [np.mean(pur_sort[:i]) for i in range(1, len(pur_sort) + 1)]
    cov_cum = [np.mean(np.sum(np.stack(cov_sort[:i], axis=0), axis=0), axis=0)
               for i in range(1, len(cov_sort) + 1)]
    return pur_cum, cov_cum


def get_purity_coverage(dump, marker):
    pur = []
    cov = []
    objs = []
    for i in range(marker):
        if dump['res']['class_name'][i] != '__background__':
            pur.append(dump['res']['purity'][i])
            cov.append(dump['res']['coverage'][i][1])
            objs.append(dump['res']['class_name'][i])
    pur_sort = np.sort(pur)[::-1]
    pur_sort_ord = np.argsort(pur)[::-1]
    cov_sort = [cov[i] for i in pur_sort_ord]
    obj_sort = [objs[i] for i in pur_sort_ord]
    pur_cum = [np.mean(pur_sort[:i]) for i in range(1, len(pur_sort) + 1)]
    cov_cum = [np.mean(np.sum(np.stack(cov_sort[:i], axis=0), axis=0), axis=0)
               for i in range(1, len(cov_sort) + 1)]
    obj_cum = [len(list(set(obj_sort[:i]))) for i in range(1, len(obj_sort) + 1)]
    return pur_cum, cov_cum, obj_cum,


def get_novel_sorted_purity_coverage(dump, marker):
    pur = []
    cov = []
    objs = []
    for i in range(marker):
        if dump['res']['class_name'][i] != '__background__' and dump['res']['class_name'][i] not in pascal_classes:
            pur.append(dump['res']['purity'][i])
            cov.append(dump['res']['coverage'][i][1][~np.in1d(np.arange(81), pascal_indices)])
            objs.append(dump['res']['class_name'][i])
    pur_sort = np.sort(pur)[::-1]
    pur_sort_ord = np.argsort(pur)[::-1]
    cov_sort = [cov[i] for i in pur_sort_ord]
    obj_sort = [objs[i] for i in pur_sort_ord]
    pur_cum = [np.mean(pur_sort[:i]) for i in range(1, len(pur_sort) + 1)]
    cov_cum = [np.mean(np.sum(np.stack(cov_sort[:i], axis=0), axis=0), axis=0)
               for i in range(1, len(cov_sort) + 1)]
    obj_cum = [len(list(set(obj_sort[:i]))) for i in range(1, len(obj_sort) + 1)]
    return pur_cum, cov_cum, obj_cum,


def find_iou(box1, box2):
    """ Find iou """
    # box1 is of shape Nx4 [x1,y1,x2,y2]
    # box2 is of shape Mx4 [x1,y1,x2,y2]
    # returns a NxM IoU matrix
    box1 = np.expand_dims(box1, axis=1)  # shape Nx1x4
    box2 = np.expand_dims(box2, axis=0)  # shape 1xMx4
    I_x = np.minimum(box1[:, :, 2], box2[:, :, 2]) - np.maximum(box1[:, :, 0], box2[:, :, 0])
    I_y = np.minimum(box1[:, :, 3], box2[:, :, 3]) - np.maximum(box1[:, :, 1],
                                                                box2[:, :, 1])
    I_x[I_x < 0] = 0
    I_y[I_y < 0] = 0
    I_A = I_x * I_y

    A_1 = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1])
    A_2 = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1])
    U_A = A_1 + A_2 - I_A
    return I_A / U_A


def eval_discovery_metrics(roi_dump, gt_dump, cache_meta, all_labels,
                           iou_thresh=0.5, marker=80, verbose=False):
    # for coco... Hard coded for now
    box_counts = np.zeros(81)
    # initialize flags
    cluster_flag_dict = {}
    cluster_pos_label_dict = {}
    coverage_cluster_dict = {}
    dump_data = {}
    coverage_dict = {}
    # Get variables for eval
    for key, value in cache_meta.items():
        dump_data[key] = -1 * np.ones(len(value))
        cluster_flag_dict[key] = -1 * np.ones(len(value))
        cluster_pos_label_dict[key] = np.zeros(len(value))
        coverage_cluster_dict[key] = {}
        coverage_dict[key] = []
        for sample_num, sample in enumerate(value):
            coverage_cluster_dict[key][sample_num] = {}

    start = 0.0
    end = 0.0
    avg = 0.0
    elapsed = 0.0
    im_names = list(gt_dump.keys())
    corloc_count = 0
    covered_count = 0
    for idx, val in enumerate(im_names):
        gt_boxes = gt_dump[val]
        rem = []
        for id_box, box in enumerate(gt_boxes):
            if box[4] <= 80:
                box_counts[int(box[4])] += 1
            else:
                rem.append(id_box)
        # proposal_ious = []  # Unused variable
        proposal_meta = []
        if val not in all_labels:
            continue
        cluster_assignments = all_labels[val]
        grp_roi = []
        for lab_id, lab in enumerate(cluster_assignments):
            if not np.isnan(lab):
                if lab >= 0:
                    try:
                        sample_id = cache_meta[lab].index((val, lab_id))
                    except:
                        raise Exception("Possible user file processing inconsistency. Cache for a cluster does not"
                                        "have expected image")
                    cluster_num = lab
                    feat_idx = lab_id
                    grp_roi.append(roi_dump[val][feat_idx])
                    proposal_meta.append([sample_id, cluster_num, idx,
                                          val, feat_idx])
        if not grp_roi:
            # if no predictions in current image
            continue
        grp_roi = np.stack(grp_roi, axis=0)
        proposal_ious = find_iou(grp_roi, gt_boxes[:, :4])
        if (proposal_ious >= 0.5).any():
            corloc_count += 1

        iou = proposal_ious
        visited_proposal = np.zeros(iou.shape[0], dtype=bool)
        gts_covered = np.zeros((iou.shape[1]))
        for gt_box_id in range(iou.shape[1]):
            cur_ious = iou[:, gt_box_id]
            # Sorted for no apparent reason
            sorted_ious = np.sort(cur_ious)[::-1]
            sorted_ious_ids = np.argsort(cur_ious)[::-1]
            found = False
            for ii, (iou_, corr_id) in enumerate(zip(sorted_ious,
                                                     sorted_ious_ids)):
                cluster_num_sample = proposal_meta[corr_id][1]

                cur_sample_id = proposal_meta[corr_id][0]

                if iou_ < iou_thresh:
                    # If iou is less than threshold, then don't bother processing all other sorted proposals
                    if dump_data[cluster_num_sample][cur_sample_id] == -1:
                        dump_data[cluster_num_sample][cur_sample_id] = iou_
                    break
                if not found and not visited_proposal[corr_id]:
                    cluster_pos_label_dict[cluster_num_sample][cur_sample_id] = gt_boxes[gt_box_id, 4]
                    cluster_flag_dict[cluster_num_sample][cur_sample_id] = 1
                    dump_data[cluster_num_sample][cur_sample_id] = iou_
                    coverage_cluster_dict[cluster_num_sample][cur_sample_id] = (proposal_meta[corr_id][2], gt_box_id)
                    if (idx, gt_box_id, gt_boxes[gt_box_id, 4]) not in coverage_dict[cluster_num_sample]:
                        coverage_dict[cluster_num_sample].append((idx, gt_box_id, gt_boxes[gt_box_id, 4]))
                    found = True
                    visited_proposal[corr_id] = True
                    gts_covered[ii] = 1
                else:
                    cluster_pos_label_dict[cluster_num_sample][cur_sample_id] = gt_boxes[gt_box_id, 4]
                    cluster_flag_dict[cluster_num_sample][cur_sample_id] = 0
                    dump_data[cluster_num_sample][cur_sample_id] = iou_
                    coverage_cluster_dict[cluster_num_sample][cur_sample_id] = (proposal_meta[corr_id][2], gt_box_id)
        covered_count += gts_covered.sum()
        elapsed += end
        avg = ((idx * avg) + end) / (idx + 1)
    results = defaultdict(list)
    results['corloc'] = corloc_count / len(gt_dump)
    results['recall'] = covered_count / box_counts.sum()
    results['gt_box_cnts'] = box_counts
    results['coverage_dict'] = coverage_dict
    coverage_list = []
    for cluster in cache_meta:
        cur_pos_list = cluster_pos_label_dict[cluster]
        cur_flag = cluster_flag_dict[cluster]
        # Values in flag > 0 have IoUs > thresh with some groound truth
        num_positives = float(np.sum(cur_flag >= 0))
        coverage_meta = coverage_cluster_dict[cluster]
        new_coverage_meta = coverage_dict[cluster]
        coverage_list += new_coverage_meta
        if len(cache_meta[cluster]) != len(coverage_meta):
            raise Exception("Cache_meta and coverage_meta do not agree for a cluster")

        if num_positives > 0:

            if np.all(cur_pos_list <= 0):
                results['purity'].append(0)
                results['coverage'].append([np.zeros(81), 0])
                results['class_name'].append(coco_classes[0])
                results['pos_id'].append(cur_flag)
                results['pos_label'].append(cur_pos_list)
                results['cov_assign'].append(new_coverage_meta)
                results['cag_coverage'].append(0)

            else:
                # put = to consider duplicates as well
                class_dist_counter = Counter(cur_pos_list[cur_flag >= 0])

                class_dist = class_dist_counter.most_common(1)[0]
                # Coverage is like recall, so don't consider duplicates
                # consider only positives hence cur_flag > 0
                # Coverage is number of gt boxes covered by crrent cluster
                # class_dist_counter_coverage = Counter(cur_pos_list[cur_flag > 0])
                class_agnostic_coverage = len(new_coverage_meta) / np.sum(box_counts[1:])

                coverage_arr = np.zeros(81)
                for ele in new_coverage_meta:
                    coverage_arr[int(ele[2])] += 1
                if np.sum(coverage_arr) != len(new_coverage_meta):
                    raise Exception("coverage_Arr and new_coverage_meta do not agree")
                coverage_arr = coverage_arr.astype(float)
                coverage_arr = [0] + list((coverage_arr[1:]) / (box_counts[1:] + 1e-08))
                coverage_arr = np.asarray(coverage_arr)

                coverage_arr[(box_counts == 0).astype(bool)] = 0
                if not np.all(coverage_arr <= 1.):
                    assert np.all(coverage_arr <= 1.)
                try:
                    _ = np.where(cur_pos_list == class_dist[0])[0]
                except:
                    raise Exception("cannot find class_dist in curr_pos_list")
                if np.isnan(class_dist[1] / num_positives):
                    raise Exception("Possible nan in class_dist")
                if class_dist[1] / float(cur_pos_list.shape[0]) > 1:
                    raise Exception("class_dist > 1")
                results['purity'].append(class_dist[1] / float(cur_pos_list.shape[0]))
                results['class_name'].append(coco_classes[int(class_dist[0])])
                results['pos_id'].append(cur_flag)
                results['pos_label'].append(cur_pos_list)
                results['cov_assign'].append(new_coverage_meta)
                results['cag_coverage'].append(len(new_coverage_meta) / box_counts.sum())
                cvrg = np.zeros(81)
                for ele in new_coverage_meta:
                    cvrg[int(ele[2])] += 1
                final_coverage = np.asarray([0] + (cvrg[1:] / (box_counts[1:] + 1e-08)).tolist())
                if np.any(final_coverage > 1):
                    raise Exception("final_coverage > 1 for some or all")
                results['coverage'].append([0, final_coverage, 0, 0])

        else:
            results['purity'].append(0)
            results['coverage'].append([0, np.zeros(81), 0, 0])
            results['class_name'].append(coco_classes[0])
            results['pos_id'].append(cur_flag)
            results['pos_label'].append(cur_pos_list)
            results['cov_assign'].append(new_coverage_meta)
            results['cag_coverage'].append(0)

        cvrg = np.zeros(81)
        for ele in coverage_list:
            cvrg[int(ele[2])] += 1
        final_coverage = np.asarray([0] + (cvrg[1:] / (box_counts[1:] + 1e-08)).tolist())

        results['final_coverage'].append(final_coverage)

    knwn_cum_pur, knwn_cum_cov = get_knwn_sorted_purity_coverage({'res': results}, marker)
    knwn_auc = get_auc(knwn_cum_pur, knwn_cum_cov)
    unknwn_cum_pur, unknwn_cum_cov, unknwn_num_obj = get_novel_sorted_purity_coverage({'res': results}, marker)
    unknwn_auc = get_auc(unknwn_cum_pur, unknwn_cum_cov)
    results['unknown_auc'] = unknwn_auc
    results['unknown_cum_pur'] = unknwn_cum_pur
    results['unknown_cum_cov'] = unknwn_cum_cov
    results['unknown_num_obj'] = unknwn_num_obj[-1]
    results['number_of_clusters'] = marker

    results['known_auc'] = knwn_auc
    results['known_cum_pur'] = knwn_cum_pur
    results['known_cum_cov'] = knwn_cum_cov

    results['iou'] = [iou_thresh]
    return results, dump_data
