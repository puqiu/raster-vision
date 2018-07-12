import unittest

import numpy as np
from object_detection.utils.np_box_list import BoxList

from rastervision.core.crs_transformer import CRSTransformer
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.labels.object_detection_labels import (
    ObjectDetectionLabels, geojson_to_labels)


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
        self.class_map = ClassMap([
            ClassItem(1, 'car'),
            ClassItem(2, 'house')
        ])

        self.npboxes = np.array([
            [0., 0., 2., 2.],
            [2., 2., 4., 4.],
        ])
        self.class_ids = np.array([1, 2])
        self.scores = np.array([0.9, 0.9])
        self.labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids, scores=self.scores)

        self.crs_transformer = DoubleCRSTransformer()
        self.geojson = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [
                                [0., 0.],
                                [0., 1.],
                                [1., 1.],
                                [1., 0.],
                                [0., 0.]
                            ]
                        ]
                    },
                    'properties': {
                        'class_id': 1,
                        'class_name': 'car',
                        'score': 0.9
                    }
                },
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [
                                [1., 1.],
                                [1., 2.],
                                [2., 2.],
                                [2., 1.],
                                [1., 1.]
                            ]
                        ]
                    },
                    'properties': {
                        'class_id': 2,
                        'score': 0.9,
                        'class_name': 'house'
                    }
                }
            ]
        }

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
        labels = geojson_to_labels(self.geojson, self.crs_transformer)
        self.check_labels(labels, self.npboxes, self.class_ids, self.scores)

    def test_from_geojson(self):
        labels = ObjectDetectionLabels.from_geojson(
            self.geojson, self.crs_transformer)
        self.check_labels(labels, self.npboxes, self.class_ids, self.scores)

        # Should filter out second box using extent.
        extent = Box.make_square(0, 0, 2)
        labels = ObjectDetectionLabels.from_geojson(
            self.geojson, self.crs_transformer, extent=extent)
        self.check_labels(
            labels, self.npboxes[0:1, :], self.class_ids[0:1],
            self.scores[0:1])

    def test_to_geojson(self):
        geojson = self.labels.to_geojson(self.crs_transformer, self.class_map)
        self.assertDictEqual(geojson, self.geojson)

    def test_get_boxes(self):
        boxes = self.labels.get_boxes()
        self.assertEqual(len(boxes), 2)
        np.testing.assert_array_equal(
            boxes[0].npbox_format(), self.npboxes[0, :])
        np.testing.assert_array_equal(
            boxes[1].npbox_format(), self.npboxes[1, :])

    def test_get_coordinates(self):
        ymins, xmins, ymaxs, xmaxs = self.labels.get_coordinates()
        np.testing.assert_array_equal(ymins, self.npboxes[:, 0])
        np.testing.assert_array_equal(xmins, self.npboxes[:, 1])
        np.testing.assert_array_equal(ymaxs, self.npboxes[:, 2])
        np.testing.assert_array_equal(xmaxs, self.npboxes[:, 3])

    def test_len(self):
        nb_labels = len(self.labels)
        self.assertEqual(self.npboxes.shape[0], nb_labels)

    def test_prune_duplicates(self):
        # This first box has a score below score_thresh so it should get
        # pruned. The third box overlaps with the second, but has higher score,
        # so the second one should get pruned. The fourth box overlaps with
        # the second less than merge_thresh, so it should not get pruned.
        npboxes = np.array([
            [0., 0., 2., 2.],
            [2., 2., 4., 4.],
            [2.1, 2.1, 4.1, 4.1],
            [3.5, 3.5, 5.5, 5.5]
        ])
        class_ids = np.array([1, 2, 1, 2])
        scores = np.array([0.2, 0.9, 0.9, 1.0])
        labels = ObjectDetectionLabels(npboxes, class_ids, scores=scores)
        score_thresh = 0.5
        merge_thresh = 0.5
        pruned_labels = labels.prune_duplicates(score_thresh, merge_thresh)

        self.assertEqual(len(pruned_labels), 2)

        expected_npboxes = np.array([
            [2.1, 2.1, 4.1, 4.1],
            [3.5, 3.5, 5.5, 5.5]
        ])
        expected_class_ids = np.array([1, 2])
        expected_scores = np.array([0.9, 1.0])

        # prune_duplicates does not maintain ordering of boxes, so find match
        # between pruned boxes and expected_npboxes.
        pruned_npboxes = pruned_labels.get_npboxes()
        pruned_inds = [None, None]
        for box_ind, box in enumerate(expected_npboxes):
            for pruned_box_ind, pruned_box in enumerate(pruned_npboxes):
                if np.array_equal(pruned_box, box):
                    pruned_inds[box_ind] = pruned_box_ind
        self.assertTrue(np.all(pruned_inds != None))

        self.check_labels(pruned_labels, expected_npboxes[pruned_inds],
                          expected_class_ids[pruned_inds],
                          expected_scores[pruned_inds])


if __name__ == '__main__':
    unittest.main()
