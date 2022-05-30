import os
import pandas as pd
from tqdm import tqdm
import pickle

import json
import pickle as pkl
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from bitrap.structures.trajectory_ops import *
from bitrap.utils.box_utils import signedIOU
# from datasets.PIE_origin import PIE
from . import transforms as T
from torchvision.transforms import functional as F
import copy
from bitrap.utils.dataset_utils import bbox_to_goal_map, squarify, img_pad
import glob
import time
import pdb

POS = "position"
X = "x"
Y = "y"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"

PRECEDING_X = "precedingX"
PRECEDING_Y = "precedingY"
PRECEDING_X_VELOCITY = "precedingXVelocity"

FOLLOWING_X = "followingX"
FOLLOWING_Y = "followingY"
FOLLOWING_X_VELOCITY = "followingXVelocity"

LEFT_PRECEDING_X = "leftPrecedingX"
LEFT_PRECEDING_Y = "leftPrecedingY"
LEFT_PRECEDING_X_VELOCITY = "leftPrecedingXVelocity"

LEFT_ALONGSIDE_X = "leftAlongsideX"
LEFT_ALONGSIDE_Y = "leftAlongsideY"
LEFT_ALONGSIDE_X_VELOCITY = "leftAlongsideXVelocity"

LEFT_FOLLOWING_X = "leftFollowingX"
LEFT_FOLLOWING_Y = "leftFollowingY"
LEFT_FOLLOWING_X_VELOCITY = "leftFollowingXVelocity"

RIGHT_PRECEDING_X = "rightPrecedingX"
RIGHT_PRECEDING_Y = "rightPrecedingY"
RIGHT_PRECEDING_X_VELOCITY = "rightPrecedingXVelocity"

RIGHT_ALONGSIDE_X = "rightAlongsideX"
RIGHT_ALONGSIDE_Y = "rightAlongsideY"
RIGHT_ALONGSIDE_X_VELOCITY = "rightAlongsideXVelocity"

RIGHT_FOLLOWING_X = "rightFollowingX"
RIGHT_FOLLOWING_Y = "rightFollowingY"
RIGHT_FOLLOWING_X_VELOCITY = "rightFollowingXVelocity"

TARGET_PRECEDING_X = "targetPrecedingX"
TARGET_PRECEDING_Y = "targetPrecedingY"
TARGET_PRECEDING_X_VELOCITY = "targetPrecedingXVelocity"

TARGET_FOLLOWING_X = "targetFollowingX"
TARGET_FOLLOWING_Y = "targetFollowingY"
TARGET_FOLLOWING_X_VELOCITY = "targetFollowingXVelocity"

LINEAR_FEATURE = "linearFeature"


class HighDDataSet(data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.root = cfg.DATASET.ROOT
        self.cfg = cfg
        # NOTE: add downsample function
        self.downsample_step = int(30 / self.cfg.DATASET.FPS)
        # traj_data_opts = {
        #     'fstride': 1,
        #     'sample_type': 'all',
        #     'height_rng': [0, float('inf')],
        #     'squarify_ratio': 0,
        #     'data_split_type': 'default',  # kfold, random, default
        #     'seq_type': 'trajectory',
        #     'min_track_size': 61,
        #     'random_params': {
        #         'ratios': None,
        #         'val_data': True,
        #         'regen_data': True
        #     },
        #     'kfold_params': {
        #         'num_folds': 5,
        #         'fold': 1
        #     }
        # }

        traj_model_opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,  # 切分时重叠多少
            'observe_length': 25,
            'predict_length': 75,
            'enc_input_type': [LINEAR_FEATURE],  # TODO
            'dec_input_type': [],
            'prediction_type': [POS]  # TODO
        }
        self.relevant_input_type = [
            X_VELOCITY,
            Y_VELOCITY,
            PRECEDING_X,
            PRECEDING_Y,
            PRECEDING_X_VELOCITY,
            FOLLOWING_X,
            FOLLOWING_Y,
            FOLLOWING_X_VELOCITY,
            LEFT_PRECEDING_X,
            LEFT_PRECEDING_Y,
            LEFT_PRECEDING_X_VELOCITY,
            LEFT_ALONGSIDE_X,
            LEFT_ALONGSIDE_Y,
            LEFT_ALONGSIDE_X_VELOCITY,
            LEFT_FOLLOWING_X,
            LEFT_FOLLOWING_Y,
            LEFT_FOLLOWING_X_VELOCITY,
            RIGHT_PRECEDING_X,
            RIGHT_PRECEDING_Y,
            RIGHT_PRECEDING_X_VELOCITY,
            RIGHT_ALONGSIDE_X,
            RIGHT_ALONGSIDE_Y,
            RIGHT_ALONGSIDE_X_VELOCITY,
            RIGHT_FOLLOWING_X,
            RIGHT_FOLLOWING_Y,
            RIGHT_FOLLOWING_X_VELOCITY,
        ]
        # imdb = PIE(data_path=self.root)

        # traj_model_opts['enc_input_type'].extend(
        #     self.relevant_input_type)  # TODO
        # traj_model_opts['prediction_type'].extend(
        #     ['obd_speed', 'heading_angle'])
        # beh_seq = imdb.generate_data_trajectory_sequence(
        #     self.split, **traj_data_opts)
        self.data = self.get_traj_data(**traj_model_opts)

    def __getitem__(self, index):
        linear_feature = torch.FloatTensor(self.data['linear_feature'][index])
        pred_xy = torch.FloatTensor(self.data['pred_xy'][index])
        # cur_image_file = self.data['obs_image'][index][-1]
        # pred_resolution = torch.FloatTensor(
        #     self.data['pred_resolution'][index])

        ret = {
            'input_x': linear_feature,
            'target_y': pred_xy,
            # 'cur_image_file': cur_image_file
        }

        # ret['timestep'] = int(cur_image_file.split('/')[-1].split('.')[0])

        # ret['pred_resolution'] = pred_resolution
        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def get_traj_tracks(self, dataset, data_types, observe_length,
                        predict_length, overlap, normalize):
        """
        Generates tracks by sampling from pedestrian sequences
        :param dataset: The raw data passed to the method
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
        :param observe_length: The length of the observation (i.e. time steps of the encoder)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
        :return: A dictinary containing sampled tracks for each data modality
        """
        #  Calculates the overlap in terms of number of frames
        seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride
        # from IPython import embed
        # embed()
        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except:  # KeyError:
                raise KeyError('Wrong data type is selected %s' % dt)

        # d['image'] = dataset['image']
        # d['pid'] = dataset['pid']
        # d['resolution'] = dataset['resolution']

        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            for track in d[k]:
                for i in range(0, len(track) - seq_length + 1, overlap_stride):
                    tracks.append(track[i:i + seq_length])
            d[k] = tracks
        #  Normalize tracks using FOL paper method,
        # d['bbox'] = self.convert_normalize_bboxes(d['bbox'], d['resolution'],
        #                                           self.cfg.DATASET.NORMALIZE,
        #                                           self.cfg.DATASET.BBOX_TYPE)
        return d

    def convert_normalize_bboxes(self, all_bboxes, all_resolutions, normalize,
                                 bbox_type):
        '''input box type is x1y1x2y2 in original resolution'''
        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = np.array(all_bboxes[i])
            # NOTE ltrb to cxcywh
            if bbox_type == 'cxcywh':
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]] / 2
            # NOTE Normalize bbox
            if normalize == 'zero-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (bbox - _min) / (_max - _min)
            elif normalize == 'plus-minus-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            elif normalize == 'none':
                pass
            else:
                raise ValueError(normalize)
            all_bboxes[i] = bbox
        return all_bboxes

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        :param data: A dictionary of different data types
        :param data_type: The data types defined for encoder and decoder input/output
        :return: A unified data representation as a list
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))

        #  Concatenate different data points into a single representation
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        elif len(d) == 1:
            return d[0]
        else:
            return d

    def get_traj_data(self, **model_opts):
        """
        Main data generation function for training/testing
        :param data: The raw data
        :param model_opts: Control parameters for data generation characteristics (see below for default values)
        :return: A dictionary containing training and testing data
        """

        opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,
            'observe_length': self.cfg.MODEL.INPUT_LEN,
            'predict_length': self.cfg.MODEL.PRED_LEN,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }
        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] +
                         opts['prediction_type'])

        file_dir = self.root
        data_path = os.path.join(file_dir, 'data.pickle')
        try:
            with open(data_path, 'rb') as data_file:
                print("Loading Pickle Data")
                data = pickle.load(data_file)
        except:
            print("Loading Raw Data CSV")
            files = os.listdir(file_dir)
            data = {
                POS: [],
                X_VELOCITY: [],
                Y_VELOCITY: [],
                PRECEDING_X: [],
                PRECEDING_Y: [],
                PRECEDING_X_VELOCITY: [],
                FOLLOWING_X: [],
                FOLLOWING_Y: [],
                FOLLOWING_X_VELOCITY: [],
                LEFT_PRECEDING_X: [],
                LEFT_PRECEDING_Y: [],
                LEFT_PRECEDING_X_VELOCITY: [],
                LEFT_ALONGSIDE_X: [],
                LEFT_ALONGSIDE_Y: [],
                LEFT_ALONGSIDE_X_VELOCITY: [],
                LEFT_FOLLOWING_X: [],
                LEFT_FOLLOWING_Y: [],
                LEFT_FOLLOWING_X_VELOCITY: [],
                RIGHT_PRECEDING_X: [],
                RIGHT_PRECEDING_Y: [],
                RIGHT_PRECEDING_X_VELOCITY: [],
                RIGHT_ALONGSIDE_X: [],
                RIGHT_ALONGSIDE_Y: [],
                RIGHT_ALONGSIDE_X_VELOCITY: [],
                RIGHT_FOLLOWING_X: [],
                RIGHT_FOLLOWING_Y: [],
                RIGHT_FOLLOWING_X_VELOCITY: [],
                LINEAR_FEATURE: []
            }  #TODO
            for file_name in tqdm(files):
                if 'set' in file_name:
                    df = pd.read_csv(os.path.join(file_dir, file_name))
                    data[POS].append(np.array(df[[X, Y]]))
                    # for input_type in self.relevant_input_type:
                    #     data[input_type].append(np.array(df[[input_type]]))
                    data[LINEAR_FEATURE].append(
                        np.array(df[[X, Y] + self.relevant_input_type]))
            data_file = open(data_path, 'wb')
            pickle.dump(data, data_file)
            data_file.close()

        data_tracks = self.get_traj_tracks(data, data_types, observe_length,
                                           opts['predict_length'],
                                           opts['track_overlap'],
                                           opts['normalize_bbox'])
        obs_slices = {}
        pred_slices = {}
        #  Generate observation/prediction sequences from the tracks
        for k in data_tracks.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            # NOTE: Add downsample function
            down = self.downsample_step
            obs_slices[k].extend(
                [d[down - 1:observe_length:down] for d in data_tracks[k]])
            pred_slices[k].extend(
                [d[observe_length + down - 1::down] for d in data_tracks[k]])

        ret = {
            # 'obs_image': obs_slices['image'],
            # 'obs_pid': obs_slices['pid'],
            # 'obs_resolution': obs_slices['resolution'],
            # 'pred_image': pred_slices['image'],
            # 'pred_pid': pred_slices['pid'],
            # 'pred_resolution': pred_slices['resolution'],
            'linear_feature':
            np.array(obs_slices[LINEAR_FEATURE]),  #enc_input,\
            'pred_xy': np.array(pred_slices[POS]),  #pred_target,
        }

        return ret

    def get_path(self,
                 file_name='',
                 save_folder='models',
                 dataset='pie',
                 model_type='trajectory',
                 save_root_folder='data/'):
        """
        A path generator method for saving model and config data. It create directories if needed.
        :param file_name: The actual save file name , e.g. 'model.h5'
        :param save_folder: The name of folder containing the saved files
        :param dataset: The name of the dataset used
        :param save_root_folder: The root folder
        :return: The full path for the model name and the path to the final folder
        """
        save_path = os.path.join(save_root_folder, dataset, model_type,
                                 save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path