# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create
from ppdet.modeling.SRaid.deeplabedsr import DeepLab  # lzh add for SRaid
__all__ = ['DAMSDet']

@register
class DAMSDet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_vis,
                 backbone_ir,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_vis=None,
                 neck_ir=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False,
                 sr_branch=None,              # <—— 新增
                 sr_weight=0.1):              # <—— 新增
        
        super(DAMSDet, self).__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_vis = neck_vis
        self.neck_ir = neck_ir
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
                # SR 分支
        self.sr_branch = sr_branch
        self.sr_weight = sr_weight

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_vis = create(cfg['backbone_vis'])
        # backbone_ir
        backbone_ir = create(cfg['backbone_ir'])
        # neck
        kwargs = {'input_shape': backbone_vis.out_shape}
        neck_vis = create(cfg['neck_vis'], **kwargs) if cfg['neck_vis'] else None
        neck_ir = create(cfg['neck_ir'], **kwargs) if cfg['neck_ir'] else None

        # transformer
        if neck_vis is not None:
            kwargs = {'input_shape': neck_vis.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_vis.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)
        sr_branch = create(cfg['sr_branch']) if cfg.get('sr_branch') else None
        sr_weight = cfg.get('sr_weight', 1.0)

        return {
            'backbone_vis': backbone_vis,
            'backbone_ir': backbone_ir,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_vis": neck_vis,
            "neck_ir": neck_ir,
            'sr_branch': sr_branch,
            'sr_weight': sr_weight
        }

    def _forward(self):
        # Backbone
        vis_body_feats = self.backbone_vis(self.inputs,1)
        ir_body_feats = self.backbone_ir(self.inputs,2)
         # ========== SR branch ==========
        sr_loss = 0.0
        if self.training:
            if self.sr_branch is not None:
                # backbone outputs: [s2_rgb, s2_ir, s3_rgb, s3_ir, s4_rgb, s4_ir]
                # fusion of RGB and IR  
                low_feat =  ir_body_feats[0]
                # high_feat =  vis_body_feats[2]
                high_feat =  vis_body_feats[2]
                output_sr = self.sr_branch(low_feat, high_feat)

                image = self.inputs['vis_image']       # [B,3,H,W]
                ir_image = self.inputs['ir_image']  # [B,1,H,W]

                l1 = paddle.nn.L1Loss()
                sr_loss = self.sr_weight * (
                    l1(output_sr[:, 0:3, :, :], image) +
                    l1(output_sr[:, 3:4, :, :], ir_image)
                )
                # l1 = paddle.nn.L1Loss()
                # sr_loss = self.sr_weight * (
                #     l1(output_sr[:, 0:1, :, :], ir_image)
                # )
                # l1 = paddle.nn.L1Loss()
                # sr_loss = self.sr_weight * (
                #     l1(output_sr[:, 0:3, :, :], image)
                # )
        # Neck
        if self.neck_vis is not None:
            vis_body_feats = self.neck_vis(vis_body_feats)
            ir_body_feats = self.neck_ir(ir_body_feats)

        pad_mask = self.inputs.get('pad_mask', None)

        out_transformer = self.transformer(None,vis_body_feats, ir_body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            # detr_losses = self.detr_head(out_transformer, None,
            #                              self.inputs)
            # detr_losses.update({
            #     'loss': paddle.add_n(
            #         [v for k, v in detr_losses.items() if 'log' not in k])
            # })
            # return detr_losses
            detr_losses = self.detr_head(out_transformer, None, self.inputs)
            total_loss = paddle.add_n([v for k, v in detr_losses.items() if 'log' not in k])
            total_loss = total_loss + sr_loss
            detr_losses['sr_loss'] = paddle.to_tensor(sr_loss)
            detr_losses['loss'] = total_loss
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, None)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
