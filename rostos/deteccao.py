# MIT License
#
# Copyright (c) 2019-2024 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Compatibilidade com MTCNN 1.0: ignora escalas sem campo receptivo válido.

Adaptação de StagePNet: preserva pesos, escalas válidas, limiares e NMS.
A versão instalada pode produzir uma escala menor que 12 pixels quando
min_face_size=24. Essa escala não pode ser processada pela PNet.
"""
import numpy as np
from mtcnn.stages import StagePNet
from mtcnn.utils.images import build_scale_pyramid, apply_scales
from mtcnn.utils.bboxes import (generate_bounding_box, upscale_bboxes,
                                smart_nms_from_bboxes, resize_to_square)


def valid_scale_groups(image_shapes, min_face_size, scale_factor):
    groups = []
    for shape in image_shapes:
        scales = build_scale_pyramid(shape[1], shape[0], min_face_size=min_face_size,
                                     scale_factor=scale_factor)
        # tf.image.resize converte as dimensões para inteiros (truncamento).
        valid = np.floor(np.min(shape[:2]) * scales) >= 12
        groups.append(scales[valid])
    return groups


class SafeStagePNet(StagePNet):
    def __call__(self, images_normalized, images_oshapes, min_face_size=20,
                 min_size=12, scale_factor=0.709, threshold_pnet=0.6,
                 nms_pnet1=0.5, nms_pnet2=0.7, **kwargs):
        groups = valid_scale_groups(images_oshapes, min_face_size, scale_factor)
        if not groups or not any(len(group) for group in groups):
            return np.empty((0, 6))
        scaled_images, scales = apply_scales(images_normalized, groups)
        batch_size = images_normalized.shape[0]
        proposals = []
        for image, scale in zip(scaled_images, scales):
            regression, classification = self._model(image)
            boxes = generate_bounding_box(regression, classification, threshold_pnet)
            boxes = upscale_bboxes(boxes, np.asarray([scale] * batch_size))
            proposals.append(smart_nms_from_bboxes(boxes, threshold=nms_pnet1,
                                                   method="union", initial_sort=False))
        boxes = np.concatenate(proposals, axis=0)
        boxes = smart_nms_from_bboxes(boxes, threshold=nms_pnet2, method="union", initial_sort=True)
        return resize_to_square(boxes)
