# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from new_predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default = "./configs/MDMTREID/sbs_R50-ibn.yml",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./demo/vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=3,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=5,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    args.config_file = "./configs/MDMTREID/sbs_R50-ibn.yml"  # config路径
    args.dataset_name = 'MDMTREID'  # 数据集名字
    args.vis_label = False  # 是否显示正确label结果
    args.rank_sort = 'descending'  # 从大到小显示关联结果
    args.label_sort = 'descending'  # 从大到小显示关联结果

    cfg = setup_cfg(args)
    # cfg["MODEL"]["WEIGHTS"] = './configs/Market1501/bagtricks_R50.yml'
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)
    print("num_query = ",num_query)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    for (feat, pid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()

    # logger.info("Computing APs for all query images ...")
    # cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    # logger.info("Finish computing APs for all query images!")

    visualizer = Visualizer(test_loader.dataset)
    visualizer.get_model_output_no_ap( distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("Saving rank list result ...")
    query_indices = visualizer.vis_rank_list_no_ap(args.output, args.vis_label, num_query,args.num_vis,
                                             args.rank_sort, args.label_sort, args.max_rank)
    logger.info("Finish saving rank list results!")
