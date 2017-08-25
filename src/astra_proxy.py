# /usr/bin/python
# coding=utf-8

"""Proxy functions for ASTRA tomography toolbox
author: sokolov v.
"""
import astra

class AstraProxy:
    def __init__(self, dims):
        if dims == 2:
            self.data_create = astra.data2d.create
            self.data_delete = astra.data2d.delete
            self.data_get = astra.data2d.get
            self.fp_algo_name = 'FP'
            self.bp_algo_name = 'BP'
            self.sirt_algo_name = 'SIRT'
            self.fbp_algo_name = 'FBP'
            self.dart_mask_name = 'DARTMASK'
            self.dart_smoothing_name = 'DARTSMOOTHING'
        elif dims == 3:
            self.data_create = astra.data3d.create
            self.data_delete = astra.data3d.delete
            self.data_get = astra.data3d.get
            self.fp_algo_name = 'FP3D'
            self.bp_algo_name = 'BP3D'
            self.sirt_algo_name = 'SIRT3D'
            self.fbp_algo_name = 'FBP3D'
            self.dart_mask_name = 'DARTMASK3D'
            self.dart_smoothing_name = 'DARTSMOOTHING3D'
        else:
            raise NotImplementedError


def gpu_fp(pg, vg, v, proj_id):
    ap = AstraProxy(len(v.shape))
    v_id = ap.data_create('-vol', vg, v)
    rt_id = ap.data_create('-sino', pg)
    fp_cfg = astra.astra_dict(ap.fp_algo_name)
    fp_cfg['VolumeDataId'] = v_id
    fp_cfg['ProjectionDataId'] = rt_id
    fp_cfg['ProjectorId'] = proj_id
    fp_id = astra.algorithm.create(fp_cfg)
    astra.algorithm.run(fp_id)
    out = ap.data_get(rt_id)
    astra.algorithm.delete(fp_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out


def gpu_bp(pg, vg, rt, proj_id, supersampling=1):
    ap = AstraProxy(len(rt.shape))
    v_id = ap.data_create('-vol', vg)
    rt_id = ap.data_create('-sino', pg, rt)
    bp_cfg = astra.astra_dict(ap.bp_algo_name)
    bp_cfg['ReconstructionDataId'] = v_id
    bp_cfg['ProjectionDataId'] = rt_id
    bp_cfg['ProjectorId'] = proj_id
    bp_id = astra.algorithm.create(bp_cfg)
    astra.algorithm.run(bp_id)
    out = ap.data_get(v_id)
    astra.algorithm.delete(bp_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out


def cpu_sirt(pg, vg, proj_id, sm, n_iters=100):
    ap = AstraProxy(len(sm.shape))
    rt_id = ap.data_create('-sino', pg, data=sm)
    v_id = ap.data_create('-vol', vg)
    sirt_cfg = astra.astra_dict(ap.sirt_algo_name)
    sirt_cfg['ReconstructionDataId'] = v_id
    sirt_cfg['ProjectionDataId'] = rt_id
    sirt_cfg['ProjectorId'] = proj_id
    sirt_id = astra.algorithm.create(sirt_cfg)
    astra.algorithm.run(sirt_id, n_iters)
    out = ap.data_get(v_id)

    astra.algorithm.delete(sirt_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out


def cpu_fbp(pg, vg, proj_id, sm, n_iters=100):
    ap = AstraProxy(len(sm.shape))
    rt_id = ap.data_create('-sino', pg, data=sm)
    v_id = ap.data_create('-vol', vg)
    fbp_cfg = astra.astra_dict(ap.fbp_algo_name)
    fbp_cfg['ReconstructionDataId'] = v_id
    fbp_cfg['ProjectionDataId'] = rt_id
    fbp_cfg['ProjectorId'] = proj_id
    fbp_id = astra.algorithm.create(fbp_cfg)
    astra.algorithm.run(fbp_id, n_iters)
    out = ap.data_get(v_id)

    astra.algorithm.delete(fbp_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out
# ---------------
