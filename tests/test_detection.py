import numpy as np
import pytest


@pytest.mark.ml
def test_pnet_filters_unusable_scales_and_preserves_valid_stage_results():
    from rostos.deteccao import SafeStagePNet, valid_scale_groups
    from mtcnn.stages import StagePNet
    from mtcnn.utils.images import build_scale_pyramid
    import tensorflow as tf

    for shape in ([1600, 900, 3], [1600, 1200, 3], [768, 1024, 3], [80, 80, 3]):
        original = build_scale_pyramid(shape[1], shape[0], 24, .709)
        filtered = valid_scale_groups([shape], 24, .709)[0]
        np.testing.assert_array_equal(filtered, original[np.floor(min(shape[:2]) * original) >= 12])
    stage = SafeStagePNet()
    # Esse formato disparava Conv2D com saída vazia, mesmo sem nenhum rosto.
    result = stage(tf.zeros((1, 1600, 900, 3)), np.array([[1600, 900, 3]]), min_face_size=24)
    assert result.ndim == 2 and result.shape[1] == 6
    # Em escalas já válidas, o algoritmo é equivalente ao original.
    shape = np.array([[100, 100, 3]])
    original_scales = build_scale_pyramid(100, 100, 24, .709)
    assert np.floor(100 * original_scales).min() >= 12
    image = tf.random.stateless_uniform((1, 100, 100, 3), [42, 8], minval=-1, maxval=1)
    expected = StagePNet.__call__(stage, image, shape, min_face_size=24)
    actual = stage(image, shape, min_face_size=24)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
