#!/usr/bin/env python3
import tempfile
import json
import zipfile
import os
from collections import namedtuple
import math

import click
import numpy as np
import tensorflow

from rastervision.utils import chain_workflow
from rastervision.utils.files import download_if_needed

CLASSIFICATION = 'classification'
OBJECT_DETECTION = 'object-detection'
all_tests = [CLASSIFICATION, OBJECT_DETECTION]

test_package_uri = '/opt/data/lf-dev/tiny-test.zip'

np.random.seed(1234)
tensorflow.set_random_seed(5678)


def get_workflow_path(temp_dir, test):
    return os.path.join(
        temp_dir, 'tiny-test/{}/configs/workflow.json'.format(test))


def get_expected_eval_path(temp_dir, test):
    return os.path.join(
        temp_dir, 'tiny-test/{}/expected-output/eval.json'.format(test))


def get_actual_eval_path(temp_dir, test):
    return os.path.join(
        temp_dir, ('rv-output/raw-datasets/tiny-test/datasets/' +
        '{}/models/default/predictions/default/evals/default/' +
        'output/eval.json').format(test))


def download_test_package(temp_dir):
    test_package_path = download_if_needed(test_package_uri, temp_dir)

    zip = zipfile.ZipFile(test_package_path, 'r')
    zip.extractall(temp_dir)
    zip.close()

    for test in all_tests:
        # Set rv_root to be temp_dir in workflow config file.
        workflow_path = get_workflow_path(temp_dir, test)
        with open(workflow_path, 'r') as workflow_file:
            workflow_dict = json.load(workflow_file)
            workflow_dict['local_uri_map']['rv_root'] = temp_dir
        with open(workflow_path, 'w') as workflow_file:
            json.dump(workflow_dict, workflow_file)


TestError = namedtuple('TestError', ['test', 'message', 'details'])


def is_eval_close(expected_eval, actual_eval):
    return True


def get_average_eval_item(eval):
    for item in eval:
        if item['class_name'] == 'average':
            return item
    return None


def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)


def check_eval_item(test, expected_item, actual_item):
    errors = []
    f1_threshold = 0.01
    class_name = expected_item['class_name']

    if math.fabs(expected_item['f1'] - actual_item['f1']) > f1_threshold:
        errors.append(TestError(
            test, 'F1 scores are not close enough',
            'for class_name: {} expected f1: {}, actual f1: {}'.format(
                class_name, expected_item['f1'], actual_item['f1'])))

    if expected_item['gt_count'] != actual_item['gt_count']:
        errors.append(TestError(
            test, 'gt_counts are not the same',
            'for class_name: {} expected gt_count: {}, actual gt_count: {}'.format(  # noqa
                class_name, expected_item['gt_count'], actual_item['gt_count'])))  # noqa

    return errors


def check_eval(test, temp_dir):
    errors = []

    actual_eval_path = get_actual_eval_path(temp_dir, test)
    expected_eval_path = get_expected_eval_path(temp_dir, test)

    if os.path.isfile(actual_eval_path):
        expected_eval = open_json(expected_eval_path)
        actual_eval = open_json(actual_eval_path)

        for expected_item in expected_eval:
            class_name = expected_item['class_name']
            actual_item = \
                next(filter(
                    lambda x: x['class_name'] == class_name, actual_eval))
            errors.extend(check_eval_item(test, expected_item, actual_item))
    else:
        errors.append(TestError(
            test, 'actual eval file does not exist', actual_eval_path))

    return errors


def run_test(test, temp_dir):
    # Run full workflow.
    errors = []
    workflow_path = get_workflow_path(temp_dir, test)
    tasks = []
    try:
        chain_workflow._main(workflow_path, tasks, run=True)
    except Exception as exc:
        errors.append(TestError(
            test, 'raised an exception while running', exc))
        return errors

    errors.extend(check_eval(test, temp_dir))

    return errors


@click.command()
@click.argument('tests', nargs=-1)
def main(tests):
    if len(tests) == 0:
        tests = all_tests

    with tempfile.TemporaryDirectory() as temp_dir:
        download_test_package(temp_dir)

        errors = []
        for test in tests:
            if test not in all_tests:
                print('{} is not valid.'.format(test))
                return

            errors.extend(run_test(test, temp_dir))

        if errors:
            for error in errors:
                print(error)
            exit(1)


if __name__ == '__main__':
    main()
