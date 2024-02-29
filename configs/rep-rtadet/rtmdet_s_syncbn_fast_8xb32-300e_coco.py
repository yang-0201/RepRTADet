_base_ = './rtmdet_l_syncbn_fast_8xb32-300e_coco_copypaste.py'
checkpoint = 'cspnext-s_imagenet_600e.pth'  # noqa

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.5
img_scale = _base_.img_scale


# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # Since the checkpoint includes CUDA:0 data,
        # it must be forced to set map_location.
        # Once checkpoint is fixed, it can be removed.
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint,
            map_location='cpu')),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

load_from = 'rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth'
