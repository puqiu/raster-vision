import numpy as np

from object_detection.utils.np_box_list import BoxList
from object_detection.utils.np_box_list_ops import (
    prune_non_overlapping_boxes, clip_to_window, change_coordinate_frame,
    concatenate, scale, non_max_suppression, _copy_extra_fields)

from rastervision.core.box import Box
from rastervision.core.labels import Labels
from rastervision.labels.utils import boxes_to_geojson


def geojson_to_labels(geojson, crs_transformer):
    """Convert GeoJSON to ObjectDetectionLabels object.

    Args:
        geojson: dict in GeoJSON format
        crs_transformer: used to convert map coords in geojson to pixel coords
            in labels object
    """
    features = geojson['features']
    boxes = []
    class_ids = []
    scores = []

    for feature in features:
        # Convert polygon to pixel coords and then convert to bounding box.
        polygon = feature['geometry']['coordinates'][0]
        polygon = [crs_transformer.web_to_pixel(p) for p in polygon]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        boxes.append(Box(ymin, xmin, ymax, xmax))

        properties = feature['properties']
        class_ids.append(properties['class_id'])
        scores.append(properties.get('score', 1.0))

    boxes = np.array([box.npbox_format() for box in boxes], dtype=float)
    class_ids = np.array(class_ids)
    scores = np.array(scores)
    return ObjectDetectionLabels(boxes, class_ids, scores=scores)


def inverse_change_coordinate_frame(boxlist, window):
    scaled_boxlist = scale(boxlist, window.get_height(), window.get_width())
    npboxes = np.round(scaled_boxlist.get())
    npboxes += [window.ymin, window.xmin, window.ymin, window.xmin]
    boxlist_new = BoxList(npboxes)
    _copy_extra_fields(boxlist_new, boxlist)
    return boxlist_new


class ObjectDetectionLabels(Labels):
    def __init__(self, npboxes, class_ids, scores=None):
        """Construct a set of object detection labels.

        Args:
            npboxes: float numpy array of size nx4 with cols
                ymin, xmin, ymax, xmax. Should be in pixel coordinates.
            class_ids: int numpy array of size n with class ids starting at 1
            scores: float numpy array of size n
        """
        self.boxlist = BoxList(npboxes)
        # This field name actually needs to be 'classes' to be able to use
        # certain utility functions in the TF Object Detection API.
        self.boxlist.add_field('classes', class_ids)
        # We need to ensure that there is always a scores field so that the
        # concatenate method will work with empty labels objects.
        if scores is None:
            scores = np.zeros(class_ids.shape)
        self.boxlist.add_field('scores', scores)

    @staticmethod
    def from_boxlist(boxlist):
        """Make ObjectDetectionLabels from BoxList object."""
        scores = boxlist.get_field('scores') \
                 if boxlist.has_field('scores') else None
        return ObjectDetectionLabels(
            boxlist.get(), boxlist.get_field('classes'), scores=scores)

    @staticmethod
    def from_geojson(geojson, crs_transformer, extent=None):
        """Return ObjectDetectionLabels from GeoJSON.

        If extent is provided, filter out the boxes that lie more than a
        bit outside the extent.

        Args:
            geojson: dict in GeoJSON format
            crs_transformer: used to convert map coords in geojson to pixel coords
                in labels object
            extent: Box in pixel coords
        """
        labels = geojson_to_labels(geojson, crs_transformer)
        if extent is not None:
            labels = labels.get_overlapping(extent, min_ioa=0.8)
        return labels

    def to_geojson(self, crs_transformer, class_map):
        """Convert to GeoJSON dict.

        Args:
            crs_transformer: (CRSTransformer) used to convert pixel coords
                back to map
            class_map: (ClassMap) used to infer class_names from class_ids
        """
        boxes = self.get_boxes()
        class_ids = self.get_class_ids().tolist()
        scores = self.get_scores().tolist()

        return boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                                scores=scores)

    @staticmethod
    def make_empty():
        npboxes = np.empty((0, 4))
        class_ids = np.empty((0,))
        scores = np.empty((0,))
        return ObjectDetectionLabels(npboxes, class_ids, scores)

    def get_boxes(self):
        """Return list of Boxes."""
        return [Box.from_npbox(npbox) for npbox in self.boxlist.get()]

    def get_coordinates(self):
        """Return (ymins, xmins, ymaxs, xmaxs) tuple."""
        return self.boxlist.get_coordinates()

    def get_npboxes(self):
        return self.boxlist.get()

    def get_scores(self):
        if self.boxlist.has_field('scores'):
            return self.boxlist.get_field('scores')
        return None

    def get_class_ids(self):
        return self.boxlist.get_field('classes')

    def __len__(self):
        return self.boxlist.get().shape[0]

    def __str__(self):
        return str(self.boxlist.get())

    def get_subwindow(self, window, ioa_thresh=1.0):
        """Returns boxes relative to window.

        This returns the boxes that overlap enough with window, clipped to
        the window and in relative coordinates that lie between 0 and 1.
        A box overlaps "enough" if the IOA (box over window) exceeds
        ioa_thresh.

        Args:
            window: (Box)
            ioa_thresh: (float 0-1) intersection over area threshold
        """
        window_npbox = window.npbox_format()
        window_boxlist = BoxList(np.expand_dims(window_npbox, axis=0))
        boxlist = prune_non_overlapping_boxes(
            self.boxlist, window_boxlist, minoverlap=ioa_thresh)
        boxlist = clip_to_window(boxlist, window_npbox)
        boxlist = change_coordinate_frame(boxlist, window_npbox)
        return ObjectDetectionLabels.from_boxlist(boxlist)

    def get_overlapping(self, window, min_ioa=0.000001):
        """Returns list of boxes that overlap with window.

        Does not clip or perform coordinate transform.

        Args:
            min_ioa: the minimum ioa for a Box to be considered as overlapping
        """
        window_npbox = window.npbox_format()
        window_boxlist = BoxList(np.expand_dims(window_npbox, axis=0))
        boxlist = prune_non_overlapping_boxes(
            self.boxlist, window_boxlist, minoverlap=min_ioa)
        return ObjectDetectionLabels.from_boxlist(boxlist)

    def concatenate(self, window, labels):
        boxlist_new = concatenate([
            self.boxlist,
            inverse_change_coordinate_frame(labels.boxlist, window)])
        return ObjectDetectionLabels.from_boxlist(boxlist_new)

    def prune_duplicates(self, score_thresh, merge_thresh):
        max_output_size = 1000000

        pruned_boxlist = non_max_suppression(
            self.boxlist, max_output_size=max_output_size,
            iou_threshold=merge_thresh, score_threshold=score_thresh)

        return ObjectDetectionLabels.from_boxlist(pruned_boxlist)
