import torch
import numpy as np
from scipy.spatial.transform import Rotation as ROT
import torch.utils.data as data
import os
import cv2
import csv
import json
import random
from PIL import Image

class BlenderDataLoader(data.Dataset):
    """ Dataset for loading the Blender. In this case, images are
    chosen within a video.
    """

    def __init__(self, dataset, opt, seed=0, temporal_bound=0, pose_bound=2):
        super(BlenderDataLoader, self).__init__()
        self.initialize(opt.train_data_path, temporal_bound, pose_bound)

    def _get_camera_specifications(self, anno_file_path):
        with open(anno_file_path, "r") as read_file:
            camera_specs = json.load(read_file)
        return camera_specs

    def _get_camera_location(self, cam_id):
        loc_dict = self.camera_specs[cam_id]['location']
        loc = np.array([float(loc_dict['x']), float(loc_dict['y']), float(loc_dict['z'])])
        return loc

    def _get_camera_orientation(self, cam_id):
        ori_dict = self.camera_specs[cam_id]['orientation']
        ori = np.array([float(ori_dict['x']), float(ori_dict['y']), float(ori_dict['z']), float(ori_dict['w'])])
        return ori

    def _get_camera_intrinsic_matrix(self, cam_id):
        _K = np.array(self.camera_specs[cam_id]['K'])
        _K[0, :] /= 1920
        _K[1, :] /= 1080
        K = np.block(
            [[_K, np.zeros((3, 1))],
             [np.zeros((1, 3)), 1]]).astype(np.float32)
        return K

    def _build_connectivity_graph(self):
        self.connectivity_graph = {}
        for cam_id_1 in self.camera_specs.keys():
            self.connectivity_graph[cam_id_1] = {}
            neighbours = []
            distances = []
            for cam_id_2 in self.camera_specs.keys():
                # We enable this for consistency with the kitti training
                if cam_id_1 != cam_id_2:  # Check if we remove this so that the network learns to map to itself
                    loc_1 = self._get_camera_location(cam_id_1)
                    loc_2 = self._get_camera_location(cam_id_2)
                    ori_1 = self._get_camera_orientation(cam_id_1)
                    ori_2 = self._get_camera_orientation(cam_id_2)
                    neighbours.append(cam_id_2)
                    distances.append(np.linalg.norm(loc_1 - loc_2) + np.linalg.norm(ori_1 - ori_2))
            distances = np.array(distances)
            neighbours = np.array(neighbours)
            sorted_ids = np.argsort(distances)
            distances = distances[sorted_ids]
            neighbours = neighbours[sorted_ids]
            self.connectivity_graph[cam_id_1]['distances'] = distances
            self.connectivity_graph[cam_id_1]['neighbours'] = neighbours

    def _get_rgb_image_name(self, file_idx):
        return 'RGB_' + file_idx + '.jpg'

    def _get_depth_map_name(self, file_idx):
        return 'Depth_' + file_idx + '.exr'

    def _get_valid_path(self, file_name):
        file_path = os.path.join(self.dataroot, file_name)
        assert os.path.exists(file_path), "File {} does not exist. Aborting...".format(file_path)
        return file_path

    def _get_frame_n_camera_ids(self, file_idx):
        cam_str_prefix = "camera_"
        camera_id = cam_str_prefix + file_idx.split(cam_str_prefix)[-1].split('_')[0]
        frame_id = int(file_idx.split('_')[-1])
        return camera_id, frame_id

    def _check_bounds(self, file_id_1, file_id_2):
        c_id_1, f_id_1 = self._get_frame_n_camera_ids(file_id_1)
        c_id_2, f_id_2 = self._get_frame_n_camera_ids(file_id_2)
        return self._check_pose_bound(c_id_1, c_id_2) and self._check_temporal_bound(f_id_1, f_id_2)

    def _check_pose_bound(self, cam_id_1, cam_id_2):
        _neighbours = self.connectivity_graph[cam_id_1]['neighbours']
        _id_val = np.where(_neighbours == cam_id_2)[0][0]
        return _id_val <= self.pose_bound

    def _check_temporal_bound(self, frame_id_1, frame_id_2):
        return abs(frame_id_1 - frame_id_2) <= self.temporal_bound

    def _clip_depth_map(self, depth, clip_start=0.1, clip_end=1000):
        valid = (depth > clip_start) & (depth < clip_end)
        depth = np.where(valid, depth, np.nan)
        return depth

    def initialize(self, train_data_path, temporal_bound, pose_bound):
        self.dataroot = train_data_path
        anno_file_name = train_data_path.split('/')[-1].split('setup')[0] + "annotation.json"
        anno_file_path = os.path.join(self.dataroot, anno_file_name)
        assert os.path.exists(anno_file_path), "Annotation file missing!"
        self.camera_specs = self._get_camera_specifications(anno_file_path)
        self._build_connectivity_graph()
        self.min_z = 1
        self.max_z = 50
        for key in self.camera_specs.keys():
            self.min_z = self.camera_specs[key]['clip_start']
            self.max_z = self.camera_specs[key]['clip_end']
            break
        self.temporal_bound = temporal_bound  # if temporal bound = 0, per frame inference
        self.pose_bound = pose_bound

        self.ids = []
        self.num_frames = 0
        self.num_cameras = len(self.camera_specs.keys())
        for root, dirs, files in os.walk(self.dataroot, topdown=False):
            for file in files:
                file_name = file.split('/')[-1]
                if 'RGB' in file:
                    file_idx = file_name.split("RGB_")[-1].split('.')[0]
                    frame_num = int(file_idx.split('camera_')[-1].split('_')[-1])
                    self.num_frames = max(self.num_frames, frame_num)
                    if file_idx not in self.ids:
                        self.ids.append(file_idx)

        def _get_sorting_value(idx):
            cid, fid = self._get_frame_n_camera_ids(idx)
            cid = int(cid.split("camera_")[-1]) * self.num_frames
            return cid + fid

        self.ids.sort(key=_get_sorting_value)
        self.num_neighbours = ((self.pose_bound+1)*(self.temporal_bound+1))-1

        # Add functionality to shuffle the list
        self.dataset_size = self.num_cameras*self.num_frames*self.num_neighbours
        # self.dataset_size = int(len(self.ids))// ((self.pose_bound+self.temporal_bound)*2)
        # self.dataset_size = len(self.ids)
    def _get_ids(self, index):
        n_id = index % self.num_neighbours
        nf_nc = index // self.num_neighbours # frames*cameras
        f_id = nf_nc % self.num_frames
        c_id = nf_nc // self.num_frames
        np_id = (n_id + 1) // (self.temporal_bound + 1)
        nf_id = (n_id + 1) % (self.temporal_bound + 1)
        return c_id, f_id, n_id, np_id, nf_id

    def __getitem__(self, index):
        cid, fid, _, npid, nfid = self._get_ids(index)
        id_val_src = self.ids[(cid*self.num_frames) + fid]
        cam_id, _ = self._get_frame_n_camera_ids(id_val_src)
        trg_id = self.connectivity_graph[cam_id]['neighbours'][npid]
        id_val_trg = trg_id + "_" + str(fid+nfid+1).zfill(4)

        src_img_file = self._get_valid_path(self._get_rgb_image_name(id_val_src))
        src_dep_file = self._get_valid_path(self._get_depth_map_name(id_val_src))
        trg_img_file = self._get_valid_path(self._get_rgb_image_name(id_val_trg))
        trg_dep_file = self._get_valid_path(self._get_depth_map_name(id_val_trg))

        max_val = 255.
        B = self.load_image(src_img_file) / max_val * 2 - 1
        B = torch.from_numpy(B.astype(np.float32)).permute((2, 0, 1))
        max_val = 255.
        A = self.load_image(trg_img_file) / max_val * 2 - 1
        A = torch.from_numpy(A.astype(np.float32)).permute((2, 0, 1))

        depth_B = self.load_image(src_dep_file)[:, :, np.newaxis].astype(np.float32)
        depth_B = torch.from_numpy(depth_B).permute((2, 0, 1))

        depth_A = self.load_image(trg_dep_file)[:, :, np.newaxis].astype(np.float32)
        depth_A = torch.from_numpy(depth_A).permute((2, 0, 1))

        c_id_src, _ = self._get_frame_n_camera_ids(id_val_src)
        c_id_trg, _ = self._get_frame_n_camera_ids(id_val_trg)

        poseB = self._get_camera_orientation(c_id_src)
        poseA = self._get_camera_orientation(c_id_trg)

        TB = self._get_camera_location(c_id_src).reshape(3, 1)
        #         RB = ROT.from_euler('xyz', poseB).as_matrix()
        RB = ROT.from_quat(poseB).as_matrix()

        TA = self._get_camera_location(c_id_trg).reshape(3, 1)
        #         RA = ROT.from_euler('xyz', poseA).as_matrix()
        RA = ROT.from_quat(poseA).as_matrix()

        T = RA.T.dot(TB - TA)
        RMat = RA.T @ RB
        mat = np.block(
            [[RMat, T],
             [np.zeros((1, 3)), 1]])

        RT = mat.astype(np.float32)
        RTinv = np.linalg.inv(mat).astype(np.float32)
        identity = torch.eye(4)

        K_src = self._get_camera_intrinsic_matrix(c_id_src)
        K_trg = self._get_camera_intrinsic_matrix(c_id_trg)
        offset = np.array(
            [[2, 0, -1, 0], [0, 2, -1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        K_src = np.matmul(offset, K_src)
        K_trg = np.matmul(offset, K_trg)

        K_src_inv = np.linalg.inv(K_src).astype(np.float32)
        K_trg_inv = np.linalg.inv(K_trg).astype(np.float32)

        return {'images': [A, B], 'depths': [depth_A, depth_B], 'ids': [trg_img_file, src_img_file],
                'cameras': [{'Pinv': identity, 'P': identity, 'K': K_src, 'Kinv': K_src_inv},
                            {'Pinv': RTinv, 'P': RT, 'K': K_trg, 'Kinv': K_trg_inv}]
                }

    def load_image(self, image_path, dim=1080):
        if '.jpg' in image_path:
            pil_img = Image.open(image_path).convert('RGB')
            w, h = pil_img.size
            #             pil_img = pil_img.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))
            pil_img = pil_img.resize((256, 256))
            image = np.asarray(pil_img)
        elif '.exr' in image_path:
            image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            image = image[:, :, 1]
            image = self._clip_depth_map(image, clip_start=0.1, clip_end=50)
            h, w = np.shape(image)
            #             image = image[(h-dim)//2:(h+dim)//2, (w-dim)//2:(w+dim)//2]
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        return image

    def __len__(self):
        return self.dataset_size

    def toval(self, epoch):
        pass

    def totrain(self, epoch):
        pass