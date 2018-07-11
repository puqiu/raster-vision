import unittest

import numpy as np
from object_detection.utils.np_box_list import BoxList

from rastervision.core.crs_transformer import CRSTransformer
from rastervision.core.box import Box
from rastervision.labels.object_detection_labels import ObjectDetectionLabels


class DoubleCRSTransformer(CRSTransformer):
    """Mock CRSTransformer used for testing.

    Assumes map coords are 2x pixels coords.
    """
    def web_to_pixel(self, web_point):
        return (web_point[0] * 2, web_point[1] * 2)

    def pixel_to_web(self, pixel_point):
        return (pixel_point[0] / 2, pixel_point[1] / 2)


class ObjectDetectionLabelsTest(unittest.TestCase):
    def setUp(self):
        self.npboxes = np.array([
            [0., 0., 2., 2.],
            [2., 2., 4., 4.],
        ])
        self.class_ids = np.array([1, 2])
        self.scores = np.array([0.9, 0.9])
        self.labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids, scores=self.scores)

        self.geojson = {
            'features': [
                {
                    'geometry': {
                        'coordinates': [
                            [
                                [0, 0],
                                [0, 1],
                                [1, 1],
                                [1, 0],
                                [0, 0]
                            ]
                        ]
                    },
                    'properties': {
                        'class_id': 1,
                        'score': 0.9
                    }
                },
                {
                    'geometry': {
                        'coordinates': [
                            [
                                [1, 1],
                                [1, 2],
                                [2, 2],
                                [2, 1],
                                [1, 1]
                            ]
                        ]
                    },
                    'properties': {
                        'class_id': 2,
                        'score': 0.9
                    }
                }
            ]
        }
        self.crs_transformer = DoubleCRSTransformer()

    def check_labels(self, labels, expected_npboxes, expected_class_ids,
                     expected_scores):
        np.testing.assert_array_equal(labels.get_npboxes(), expected_npboxes)
        np.testing.assert_array_equal(
            labels.get_class_ids(), expected_class_ids)
        np.testing.assert_array_equal(labels.get_scores(), expected_scores)

    def test_from_boxlist(self):
        boxlist = BoxList(self.npboxes)
        boxlist.add_field('classes', self.class_ids)
        boxlist.add_field('scores', self.scores)
        labels = ObjectDetectionLabels.from_boxlist(boxlist)
        self.check_labels(labels, self.npboxes, self.class_ids, self.scores)

    def test_make_empty(self):
        npboxes = np.empty((0, 4))
        class_ids = np.empty((0,))
        scores = np.empty((0,))

        labels = ObjectDetectionLabels.make_empty()
        self.check_labels(labels, npboxes, class_ids, scores)

    def test_constructor(self):
        labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids, scores=self.scores)
        self.check_labels(labels, self.npboxes, self.class_ids, self.scores)

        labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids)
        zero_scores = np.zeros(self.class_ids.shape)
        self.check_labels(labels, self.npboxes, self.class_ids, zero_scores)

    def test_geojson_to_labels(self):
        labels = ObjectDetectionLabels.from_geojson(
            self.geojson, self.crs_transformer)
        self.check_labels(labels, self.npboxes, self.class_ids, self.scores)

    


if __name__ == '__main__':
    unittest.main()
