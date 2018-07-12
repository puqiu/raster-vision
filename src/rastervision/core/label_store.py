from abc import ABC, abstractmethod


class LabelStore(ABC):
    """The place where labels are stored for a scene.

    An LabelStore is a read/write source of labels for a scene
    that could be backed by a file, a database, an API, etc. The difference
    between LabelStores and Labels can be understood by analogy to the
    difference between a database and result sets queried from a database.
    """
    @abstractmethod
    def get_labels(self, window):
        """Get labels for a window.

        Args:
            window: Box covering area to retrieve Labels from

        Returns:
            Labels object with labels lying inside the window
        """
        pass

    @abstractmethod
    def get_all_labels(self):
        """Get all labels.

        Returns:
            Labels object with all the labels.
        """
        pass

    @abstractmethod
    def extend(self, window, labels):
        """Add labels to the LabelStore.

        Args:
            window: Box covering area where labels are from
            labels: Labels
        """
        pass

    @abstractmethod
    def post_process(self):
        """Perform some preprocessing operation before saving."""
        pass

    @abstractmethod
    def save(self):
        """Save."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all labels."""
        pass
