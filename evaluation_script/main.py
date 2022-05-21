import json
import os
import random
import uuid
import zipfile

from collections import defaultdict
import numpy as np
import pandas as pd

from .discovery_eval import eval_discovery_metrics


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")

        # TODO: Most of this code will be shared with test phase. Do not copy paste, modularize
        # Assuming that /tmp is accessible. Creating a unique folder for this run
        user_id = uuid.uuid4()
        user_dir = f'/tmp/{user_id}'
        os.makedirs(user_dir)  # This should never exist (ok, ideally :))

        # Extracting the user submission to this folder
        with zipfile.ZipFile(user_submission_file) as zip_ref:
            zip_ref.extractall(user_dir)

        # Checking the integrity of submission
        # TODO: Communicate this to users
        # TODO: Figure out how these error messages go to EvalAI front-end
        assert os.path.isfile(f'{user_dir}/discovery.csv')
        assert os.path.isfile(f'{user_dir}/detection.csv')

        # PART1: Doing discovery evaluation
        # I am assuming that the check if file is a csv will be performed at the accepting end
        discovery_data = pd.read_csv(f'{user_dir}/discovery.csv', header=0)
        # Doing sanity check on the submitted file.
        disc_exp_cols = {'image_id', 'x1', 'y1', 'x2', 'y2', 'cluster_id'}
        assert disc_exp_cols == set(discovery_data.columns)
        # Proceeding with discovery evaluation
        # Reading the gt file. TODO: Figure out if we can supply this precomputed. No point every worker redoing this.
        #                            My hunch is that we should be able to supply a pkl file as GT annotation file
        gt_dump = defaultdict(list)
        with open(test_annotation_file, 'rb') as f:
            gt = json.load(f)

        gt_class_mapping = {elem['id']: idx + 1 for idx, elem in enumerate(gt['categories'])}
        for anno in gt['annotations']:
            cur_box = anno['bbox']
            # box is in xywh format
            cur_box[2] += cur_box[0]
            cur_box[3] += cur_box[1]
            gt_dump[anno['image_id']].append(np.asarray(cur_box + [gt_class_mapping[anno['category_id']]]))
        for k, v in gt_dump.items():
            gt_dump[k] = np.stack(v)
            assert (gt_dump[k][:, 4] > 0).all()

        # Reading the user file
        cache_meta = defaultdict(list)
        all_labels = {}
        roi_dump = {}
        results_by_image = discovery_data.groupby('image_id')
        for img_id in results_by_image.groups:
            res = results_by_image.get_group(img_id)
            boxes = np.stack([res['x1'], res['y1'], res['x2'], res['y2']], axis=-1)
            roi_dump[img_id] = boxes
            if img_id not in all_labels:
                all_labels[img_id] = -np.inf * np.ones((res.shape[0]))
            for idx, (_, r) in enumerate(res.iterrows()):
                cache_meta[r['cluster_id']].append((img_id, idx))
                all_labels[img_id][idx] = r['cluster_id']

        discovery_result = eval_discovery_metrics(roi_dump, gt_dump, cache_meta,
                                                  all_labels,
                                                  marker=int(len(cache_meta)),
                                                  verbose=True)

        # TODO: Do detection eval

        output["result"] = [
            {
                "train_split": {
                    "disc_purity": discovery_result['purity'],
                    "disc_coverage": discovery_result['coverage'],
                    "disc_class_name": discovery_result['class_name'],
                    "disc_pos_id": discovery_result['pos_id'],
                    "disc_pos_label": discovery_result['pos_label'],
                    "disc_cov_assign": discovery_result['cov_assign'],
                    "disc_cag_coverage": discovery_result['cag_coverage'],
                    "disc_final_coverage": discovery_result['final_coverage'],
                    "disc_unknown_auc": discovery_result['unknown_auc'],
                    "disc_unknown_cum_pur": discovery_result['unknown_cum_pur'],
                    "disc_unknown_cum_cov": discovery_result['unknown_cum_cov'],
                    "disc_unknown_num_obj": discovery_result['unknown_num_obj'],
                    "disc_number_of_clusters": discovery_result['number_of_clusters'],
                    "disc_known_auc": discovery_result['known_auc'],
                    "disc_known_cum_pur": discovery_result['known_cum_pur'],
                    "disc_known_cum_cov": discovery_result['known_cum_cov'],
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "train_split": {
                    "Metric1": random.randint(0, 99),
                    "Metric2": random.randint(0, 99),
                    "Metric3": random.randint(0, 99),
                    "Total": random.randint(0, 99),
                }
            },
            {
                "test_split": {
                    "Metric1": random.randint(0, 99),
                    "Metric2": random.randint(0, 99),
                    "Metric3": random.randint(0, 99),
                    "Total": random.randint(0, 99),
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
