_base_ = './reprtadet_l_possion.py'

# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
load_from = 'rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth'
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])