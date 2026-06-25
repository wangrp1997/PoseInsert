# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys

def _find_fp_dir():
  for candidate in (
      os.environ.get('FOUNDATIONPOSE_DIR', ''),
      '/home/rw/Documents/FoundationPose',
      os.path.expanduser('~/Documents/FoundationPose'),
  ):
    if candidate and os.path.isfile(os.path.join(candidate, 'estimater.py')):
      return os.path.abspath(candidate)
  raise RuntimeError(
      'FoundationPose not found. Set FOUNDATIONPOSE_DIR or install under ~/Documents/FoundationPose.'
  )

_code_dir = _find_fp_dir()
if _code_dir not in sys.path:
  sys.path.insert(0, _code_dir)
os.chdir(_code_dir)

from estimater import *
from datareader import *
import argparse


_AXIS_VECTORS = {
  'x': np.array([1.0, 0.0, 0.0]),
  'y': np.array([0.0, 1.0, 0.0]),
  'z': np.array([0.0, 0.0, 1.0]),
}


def _mat_to_quat_wxyz(R):
  trace = np.trace(R)
  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (R[2, 1] - R[1, 2]) * s
    y = (R[0, 2] - R[2, 0]) * s
    z = (R[1, 0] - R[0, 1]) * s
  elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
    w = (R[2, 1] - R[1, 2]) / s
    x = 0.25 * s
    y = (R[0, 1] + R[1, 0]) / s
    z = (R[0, 2] + R[2, 0]) / s
  elif R[1, 1] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
    w = (R[0, 2] - R[2, 0]) / s
    x = (R[0, 1] + R[1, 0]) / s
    y = 0.25 * s
    z = (R[1, 2] + R[2, 1]) / s
  else:
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    w = (R[1, 0] - R[0, 1]) / s
    x = (R[0, 2] + R[2, 0]) / s
    y = (R[1, 2] + R[2, 1]) / s
    z = 0.25 * s
  q = np.array([w, x, y, z], dtype=np.float64)
  return q / np.linalg.norm(q)


def _quat_to_mat(q):
  w, x, y, z = q
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ], dtype=np.float64)


def _quat_mult(q1, q2):
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ], dtype=np.float64)


def _quat_twist(q, axis):
  axis = axis / np.linalg.norm(axis)
  w, x, y, z = q
  v = np.array([x, y, z], dtype=np.float64)
  proj = np.dot(v, axis) * axis
  twist = np.array([w, proj[0], proj[1], proj[2]], dtype=np.float64)
  norm = np.linalg.norm(twist)
  if norm < 1e-8:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  return twist / norm


def _extract_twist_rotation(R, axis_obj):
  return _quat_to_mat(_quat_twist(_mat_to_quat_wxyz(R), axis_obj))


def _lock_pose_twist(pose, axis_obj, twist_ref):
  pose = pose.copy()
  R = pose[:3, :3]
  R_twist = _extract_twist_rotation(R, axis_obj)
  R_swing = R @ R_twist.T
  pose[:3, :3] = R_swing @ twist_ref
  return pose


def _pose_to_estimator_state(pose, est):
  tf_to_center = est.get_tf_to_centered_mesh().detach().cpu().numpy()
  return pose @ np.linalg.inv(tf_to_center)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = _code_dir
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--symm_axis', type=str, default='none', choices=['none', 'x', 'y', 'z'],
                      help='Cylinder continuous symmetry axis in mesh frame. none=default FP behavior.')
  parser.add_argument('--symm_step', type=int, default=15,
                      help='Symmetry sampling step in degrees when --symm_axis is set.')
  parser.add_argument('--lock_symm_axis', type=str, default='none', choices=['none', 'x', 'y', 'z'],
                      help='Lock in-plane rotation to the first frame around this mesh axis.')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  symmetry_tfs = None
  if args.symm_axis != 'none':
    axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[args.symm_axis]
    symmetry_tfs = symmetry_tfs_from_info({
      'symmetries_continuous': [{'axis': axis, 'offset': [0, 0, 0]}]
    }, rot_angle_discrete=args.symm_step)
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx, symmetry_tfs=symmetry_tfs)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  lock_axis_obj = None
  locked_twist_R = None

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    if args.lock_symm_axis != 'none':
      if lock_axis_obj is None:
        lock_axis_obj = _AXIS_VECTORS[args.lock_symm_axis]
        locked_twist_R = _extract_twist_rotation(pose[:3, :3], lock_axis_obj)
        logging.info(f'Locked twist around {args.lock_symm_axis}-axis from frame 0')
      else:
        pose = _lock_pose_twist(pose, lock_axis_obj, locked_twist_R)
        est.pose_last = torch.as_tensor(
          _pose_to_estimator_state(pose, est), device='cuda', dtype=torch.float
        )

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

