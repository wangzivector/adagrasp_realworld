## cd ~/SpatialHybridGen/codes/adagrasp && ca adacopygen && source ~/tf_catkin_ws/devel/setup.sh
## python realgrasp.py --gripper_types robotiq_2f_85 --load_checkpoint pretrained_models/adagrasp.pth

import argparse
import numpy as np
import torch
import utils
from models.grasping_model import GraspingModel
from models.model_utils import get_action, get_affordance
from sim import RealWorldServer
from sensornode import SensorServer

import sys, signal
def signal_handler(signal, frame): sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()

# global
parser.add_argument('--mode', default='fixed_gripper', choices=['fixed_gripper'], help='test mode')
parser.add_argument('--save', default='test_vis', type=str, help='where to save the visualizations')
parser.add_argument('--gui', action="store_true", help='use GUI')
parser.add_argument('--gpu', default='0', type=str, help='GPU device ID. -1 means cpu')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')

# environment
parser.add_argument('--gripper_types', default=None, type=str, nargs='+', help='list of gripper_name to be used, separated by space')
parser.add_argument('--num_open_scale', default=5, type=int, help='number of open scales')
parser.add_argument('--min_open_scale', default=0.5, type=float, help='minimal open scale')
parser.add_argument('--max_open_scale', default=1, type=float, help='maximum open scale')	
parser.add_argument('--random_open_scale', action="store_true", help='if only choose 1 open scale')
parser.add_argument('--seq_len', default=1, type=int, help='number of steps for each sequence')

# model
parser.add_argument('--model_type', default='adagrasp', type=str,choices=['adagrasp'], help='the type of grasping model to test')
parser.add_argument('--num_rotations', default=16, type=int, help='number of rotations')
parser.add_argument('--load_checkpoint', default=None, type=str, help='path of model checkpoint (suffix is .pth)')


def main():
    args = parser.parse_args()
          
    # Reset random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device('cpu') if args.gpu == '-1' else torch.device(f'cuda:{args.gpu}')

    # Set model & load checkpoint
    model = GraspingModel(num_rotations=args.num_rotations, gripper_final_state=True)
    model = model.to(device)
    checkpoint = torch.load(args.load_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'==> Loaded checkpoint from {args.load_checkpoint}')
    model.eval()
    torch.set_grad_enabled(False)

    # Create worker for observation
    env = RealWorldServer(gui_enabled=args.gui, num_cam=1)
    azure_node = SensorServer()

    # Prepare Gripper TSDF
    assert len(args.gripper_types) == 1
    open_scales = np.linspace(args.min_open_scale, args.max_open_scale, args.num_open_scale, True)
    gripper_observation = env.load_gripper(gripper_type=args.gripper_types, gripper_size=1, 
                                               open_scales=open_scales, gripper_final_state=True)

    data = dict()
    rewards, scores = list(), list()
    n_vis_stored = 0
    
    for step in range(args.seq_len):
        scene_observation = azure_node.fetch_single_grid(grid_type='scene') 
        # 'scene_tsdf': np.load("data/samples/test_sample_scene_tsdf.npy"),
        # 'obstacle_vol': np.load("data/samples/test_sample_obstacle_vol.npy"),
        # 'valid_pix_heightmap': np.load("data/samples/test_sample_valid_pix_heightmap.npy"),

        observation = {**gripper_observation, **scene_observation}
        affordance_maps = get_affordance([observation], model, device, gripper_final_state=(args.model_type=='adagrasp'))

        # Zero out predicted action values for all pixels outside workspace
        valid = observation['valid_pix_heightmap']
        s = affordance_maps.shape
        affordance_maps[np.logical_not(np.tile(valid,(s[0],s[1],s[2],1,1)))] = 0
        action, score, others = get_action(affordance_maps[0], 0, observation['open_scales'])
        
        ## Here return reward
        reward = 1
        rewards.append(reward)
        scores.append(score)

        # store data for visualization
        rank = 0 # deprecated
        data[(rank, step)] = observation
        data[(rank, step)]['affordance_map'] = affordance_maps[0]
        data[(rank, step)]['open_scale_idx'] = others['open_scale_idx']
        data[(rank, step)]['grasp_pixel'] = np.array(action[:2])
        data[(rank, step)]['grasp_angle'] = action[2]
        data[(rank, step)]['score'] = score
        data[(rank, step)]['reward'] = reward
        n_vis_stored += 1

    ## print result
    success_rate = np.mean(rewards)
    print(f'{args.gripper_types} test success rate: {success_rate}\n')

    ## visualization
    num_open_scale = 1 if args.random_open_scale else args.num_open_scale
    utils.visualization(data, n_vis_stored, args.seq_len, num_open_scale, args.num_rotations, n_vis_stored, args.save)


if __name__=='__main__':
    main()
