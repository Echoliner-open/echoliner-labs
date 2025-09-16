import numpy as np
from echoliner.vision.edge_detection import sobel_edges


def test_sobel_edges_detects_vertical_line():
    image = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    edges = sobel_edges(image)
    assert edges.shape == image.shape
    # The edge should be strongest along the horizontal gradient
    assert edges[1, 1] > 0
