# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
"""
python tools/vis_backbone_feature_only.py   -c /home/lzh/DAMSDet-master/configs/damsdet/damsdet_r50vd_vedai.yml   --infer_vis_dir /home/lzh/DAMSDet-master/dataset/coco_VEDAI/val_imgs/vis_imgs   --infer_ir_dir /home/lzh/DAMSDet-master/dataset/coco_VEDAI/val_imgs/ir_imgs   --output_dir output/feature_maps_val   -o weights=output/VEDAI/damsdet_r50vd_vedai/best_model.pdparams
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import re
import glob
import ast
import cv2
import numpy as np

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import warnings
warnings.filterwarnings('ignore')

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model
from ppdet.utils.logger import setup_logger

logger = setup_logger('vis_backbone_feature_only')


def parse_args():
    parser = ArgsParser()

    parser.add_argument(
        "--infer_vis_img",
        type=str,
        default=None,
        help="Path of visible image.")
    parser.add_argument(
        "--infer_ir_img",
        type=str,
        default=None,
        help="Path of infrared image.")

    parser.add_argument(
        "--infer_vis_dir",
        type=str,
        default=None,
        help="Directory of visible images.")
    parser.add_argument(
        "--infer_ir_dir",
        type=str,
        default=None,
        help="Directory of infrared images.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/feature_maps",
        help="Directory for saving feature visualization.")
    parser.add_argument(
        "--feature_norm",
        type=str,
        default="mean_abs",
        choices=["mean_abs", "mean", "max"],
        help="Channel reduction mode.")
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        choices=["jet", "viridis", "hot", "bone"],
        help="Colormap for feature heatmap.")
    parser.add_argument(
        "--save_single_feature",
        type=ast.literal_eval,
        default=False,
        help="Whether to save each feature map separately.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")

    return parser.parse_args()


def get_test_images(infer_dir, infer_img):
    assert infer_img is not None or infer_dir is not None, \
        "--infer_vis_img/--infer_ir_img or --infer_vis_dir/--infer_ir_dir should be set"

    if infer_img is not None:
        assert os.path.isfile(infer_img), "{} is not a file".format(infer_img)
        return [infer_img]

    assert infer_dir is not None and os.path.isdir(infer_dir), \
        "{} is not a directory".format(infer_dir)

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    exts += [ext.upper() for ext in exts]

    for ext in exts:
        images.update(glob.glob(os.path.join(infer_dir, '*.{}'.format(ext))))

    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} images in {}.".format(len(images), infer_dir))
    return images


def extract_last_number(path):
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r'\d+', name)
    if len(nums) == 0:
        return name
    return int(nums[-1])


class BackboneFeatureHook(object):
    def __init__(self, backbone):
        self.features = None
        self.hook = backbone.register_forward_post_hook(self._hook_fn)

    def _flatten_tensors(self, out):
        tensors = []
        if isinstance(out, paddle.Tensor):
            tensors.append(out)
        elif isinstance(out, (list, tuple)):
            for item in out:
                tensors.extend(self._flatten_tensors(item))
        elif isinstance(out, dict):
            for _, v in out.items():
                tensors.extend(self._flatten_tensors(v))
        return tensors

    def _hook_fn(self, layer, inputs, output):
        feats = self._flatten_tensors(output)
        feats = [x for x in feats if isinstance(x, paddle.Tensor) and len(x.shape) == 4]

        safe_feats = []
        for x in feats[:3]:
            safe_feats.append(x.detach().cpu().numpy())
        self.features = safe_feats

    def clear(self):
        self.features = None

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def get_cv2_colormap(name):
    mapping = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
    }
    return mapping[name]


def reduce_feature_map(feat, mode='mean_abs'):
    if feat.ndim == 4:
        feat = feat[0]

    feat = feat.astype('float32')

    if mode == 'mean_abs':
        feat = np.mean(np.abs(feat), axis=0)
    elif mode == 'mean':
        feat = np.mean(feat, axis=0)
    elif mode == 'max':
        feat = np.max(feat, axis=0)
    else:
        raise ValueError("Unsupported feature reduction mode: {}".format(mode))

    feat = feat - feat.min()
    feat = feat / (feat.max() + 1e-12)
    return feat


def feature_to_heatmap(feat, out_hw, reduce_mode='mean_abs', colormap='jet'):
    feat2d = reduce_feature_map(feat, mode=reduce_mode)
    feat2d = cv2.resize(feat2d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    feat_u8 = np.uint8(np.clip(feat2d * 255.0, 0, 255))
    color = cv2.applyColorMap(feat_u8, get_cv2_colormap(colormap))
    return color


def read_vis_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to read vis image: {}".format(path))
    return img


def read_ir_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read ir image: {}".format(path))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def put_title(img, title, font_scale=0.8, thickness=2):
    canvas = np.ones((img.shape[0] + 40, img.shape[1], 3), dtype=np.uint8) * 255
    canvas[:img.shape[0], :, :] = img
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = max((img.shape[1] - text_size[0]) // 2, 5)
    y = img.shape[0] + 28
    cv2.putText(canvas, title, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def make_panel(vis_img, ir_img, vis_heatmaps, ir_heatmaps, save_path):
    assert len(vis_heatmaps) == 3, "Expected 3 VIS heatmaps, got {}".format(len(vis_heatmaps))
    assert len(ir_heatmaps) == 3, "Expected 3 IR heatmaps, got {}".format(len(ir_heatmaps))

    h, w = vis_img.shape[:2]
    ir_img = cv2.resize(ir_img, (w, h), interpolation=cv2.INTER_LINEAR)
    vis_heatmaps = [cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR) for hm in vis_heatmaps]
    ir_heatmaps = [cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR) for hm in ir_heatmaps]

    items = [
        put_title(vis_img, "(a) RGB input"),
        put_title(ir_img, "(b) IR input"),
        put_title(vis_heatmaps[0], "(c) VIS res3"),
        put_title(vis_heatmaps[1], "(d) VIS res4"),
        put_title(vis_heatmaps[2], "(e) VIS res5"),
        put_title(ir_heatmaps[0], "(f) IR res3"),
        put_title(ir_heatmaps[1], "(g) IR res4"),
        put_title(ir_heatmaps[2], "(h) IR res5"),
    ]

    gap = 20
    total_w = sum(x.shape[1] for x in items) + gap * (len(items) - 1)
    total_h = max(x.shape[0] for x in items)
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    cur_x = 0
    for item in items:
        hh, ww = item.shape[:2]
        canvas[0:hh, cur_x:cur_x + ww] = item
        cur_x += ww + gap

    cv2.imwrite(save_path, canvas)


def save_feature_visualizations(vis_path,
                                ir_path,
                                vis_feats,
                                ir_feats,
                                save_dir,
                                reduce_mode='mean_abs',
                                colormap='jet',
                                save_single_feature=False):
    os.makedirs(save_dir, exist_ok=True)

    vis_img = read_vis_image(vis_path)
    ir_img = read_ir_image(ir_path)
    h, w = vis_img.shape[:2]

    if vis_feats is None or len(vis_feats) < 3:
        raise ValueError("Captured VIS backbone features < 3")
    if ir_feats is None or len(ir_feats) < 3:
        raise ValueError("Captured IR backbone features < 3")

    vis_feats = vis_feats[:3]
    ir_feats = ir_feats[:3]

    vis_heatmaps = []
    ir_heatmaps = []

    for feat in vis_feats:
        vis_heatmaps.append(
            feature_to_heatmap(
                feat=feat,
                out_hw=(h, w),
                reduce_mode=reduce_mode,
                colormap=colormap))

    for feat in ir_feats:
        ir_heatmaps.append(
            feature_to_heatmap(
                feat=feat,
                out_hw=(h, w),
                reduce_mode=reduce_mode,
                colormap=colormap))

    stem = os.path.splitext(os.path.basename(vis_path))[0]
    panel_path = os.path.join(save_dir, "{}_backbone_vis_ir.jpg".format(stem))
    make_panel(vis_img, ir_img, vis_heatmaps, ir_heatmaps, panel_path)

    if save_single_feature:
        vis_names = ['vis_res3', 'vis_res4', 'vis_res5']
        ir_names = ['ir_res3', 'ir_res4', 'ir_res5']

        for name, hm in zip(vis_names, vis_heatmaps):
            cv2.imwrite(os.path.join(save_dir, "{}_{}.jpg".format(stem, name)), hm)
        for name, hm in zip(ir_names, ir_heatmaps):
            cv2.imwrite(os.path.join(save_dir, "{}_{}.jpg".format(stem, name)), hm)

    logger.info("Saved backbone visualization to {}".format(panel_path))


def collect_image_pairs(flags):
    use_single = (flags.infer_vis_img is not None or flags.infer_ir_img is not None)
    use_dir = (flags.infer_vis_dir is not None or flags.infer_ir_dir is not None)

    assert use_single or use_dir, \
        "Please provide either single-image args or directory args."

    if use_single:
        assert flags.infer_vis_img is not None and flags.infer_ir_img is not None, \
            "Single-image mode requires both --infer_vis_img and --infer_ir_img"
        return [(flags.infer_vis_img, flags.infer_ir_img)]

    assert flags.infer_vis_dir is not None and flags.infer_ir_dir is not None, \
        "Directory mode requires both --infer_vis_dir and --infer_ir_dir"

    vis_images = get_test_images(flags.infer_vis_dir, None)
    ir_images = get_test_images(flags.infer_ir_dir, None)

    vis_images.sort(key=extract_last_number)
    ir_images.sort(key=extract_last_number)

    assert len(vis_images) == len(ir_images), \
        "vis_images and ir_images counts do not match: {} vs {}".format(
            len(vis_images), len(ir_images))

    return list(zip(vis_images, ir_images))


def run(FLAGS, cfg):
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)

    assert hasattr(trainer.model, 'backbone_vis'), "trainer.model has no backbone_vis"
    assert hasattr(trainer.model, 'backbone_ir'), "trainer.model has no backbone_ir"

    backbone_vis = trainer.model.backbone_vis
    backbone_ir = trainer.model.backbone_ir

    logger.info("Use backbone: trainer.model.backbone_vis")
    logger.info("Use backbone: trainer.model.backbone_ir")

    vis_hook = BackboneFeatureHook(backbone_vis)
    ir_hook = BackboneFeatureHook(backbone_ir)

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    image_pairs = collect_image_pairs(FLAGS)

    for idx, (vis_path, ir_path) in enumerate(image_pairs):
        logger.info("[{}/{}] Processing vis={}, ir={}".format(
            idx + 1, len(image_pairs), vis_path, ir_path))

        vis_hook.clear()
        ir_hook.clear()

        trainer.multi_predict(
            [vis_path],
            [ir_path],
            draw_threshold=0.0,
            output_dir=FLAGS.output_dir,
            save_results=False,
            visualize=False)

        if vis_hook.features is None:
            raise RuntimeError("No VIS backbone features captured for {}".format(vis_path))
        if ir_hook.features is None:
            raise RuntimeError("No IR backbone features captured for {}".format(ir_path))

        save_feature_visualizations(
            vis_path=vis_path,
            ir_path=ir_path,
            vis_feats=vis_hook.features,
            ir_feats=ir_hook.features,
            save_dir=FLAGS.output_dir,
            reduce_mode=FLAGS.feature_norm,
            colormap=FLAGS.colormap,
            save_single_feature=FLAGS.save_single_feature)

    vis_hook.close()
    ir_hook.close()


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    if 'use_npu' not in cfg:
        cfg.use_npu = False
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    if cfg.use_gpu:
        paddle.set_device('gpu:0')
    elif cfg.use_npu:
        paddle.set_device('npu')
    elif cfg.use_xpu:
        paddle.set_device('xpu')
    elif cfg.use_mlu:
        paddle.set_device('mlu')
    else:
        paddle.set_device('cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()