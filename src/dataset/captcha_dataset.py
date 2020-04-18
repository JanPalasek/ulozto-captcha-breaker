import cv2


class CaptchaDataset:
    def __init__(self, annotations_path: str):
        self._annotations_path = annotations_path

        self._data = self._get_items()

    def get_image_shape(self):
        return self._data[0][0].shape

    def get_classes(self):
        # TODO
        return 10

    def _get_items(self):
        result = []
        with open(self._annotations_path, "r") as file:
            for line in file:
                image_path, image_label = line.split()
                image_label = list(image_label)

                image = cv2.imread(image_path, 0)
                result.append((image, image_label))

        return result

    def get_data(self):
        return list(zip(*self._data))
