## cd ~/SpatialHybridGen/codes/adagrasp_realworld && ca adacopygen && source ~/tf_catkin_ws/devel/setup.sh
## python realgrasp.py --gripper_types robotiq_2f_85 --load_checkpoint pretrained_models/adagrasp.pth

import argparse
import numpy as np
import torch
import utils
from models.grasping_model import GraspingModel
from models.model_utils import get_action, get_affordance
from sim import RealWorldServer
from sensornode import SensorServer
from motionnode import PoseClient, GripperClient
from scipy.spatial.transform import Rotation as SR

import sys, signal
def signal_handler(signal, frame): sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()

# global
parser.add_argument('--mode', default='fixed_gripper', choices=['fixed_gripper'], help='test mode')
parser.add_argument('--save', default='test_vis/object', type=str, help='where to save the visualizations')
parser.add_argument('--gui', action="store_true", help='use GUI')
parser.add_argument('--gpu', default='0', type=str, help='GPU device ID. -1 means cpu')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')

# environment
parser.add_argument('--gripper_types', default=None, type=str, nargs='+', help='list of gripper_name to be used, separated by space')
parser.add_argument('--num_open_scale', default=5, type=int, help='number of open scales')
parser.add_argument('--min_open_scale', default=0.5, type=float, help='minimal open scale')
parser.add_argument('--max_open_scale', default=1, type=float, help='maximum open scale')	
parser.add_argument('--random_open_scale', action="store_true", help='if only choose 1 open scale')

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
    print(f'[Pipeline] ==> Loaded checkpoint from {args.load_checkpoint}')
    model.eval()
    torch.set_grad_enabled(False)

    # Create worker for observationsteps
    print(f'[Pipeline] ==> Loading RealWorldServer ...')
    env = RealWorldServer(gui_enabled=args.gui, num_cam=1)
    print(f'[Pipeline] ==> Loading SensorServer ...')
    camera_node = SensorServer()
    gripper_node = GripperClient(args.gripper_types[0])
    print("[pipeline]: GripperClient initialized.")
    pose_node = PoseClient()
    pose_node.characterize_grasp_transform(armend2gripper=gripper_node.armend2gripper,
                                           gripper2grip=gripper_node.gripper2grip,
                                           gripper_tfname=gripper_node.gripper_tfname)
    print("[pipeline]: PoseClient initialized.")

    # Prepare Gripper TSDF
    print(f'[Pipeline] ==> Loading gripper TSDF ...')
    assert len(args.gripper_types) == 1
    open_scales = np.linspace(args.min_open_scale, args.max_open_scale, args.num_open_scale, True)
    gripper_observation = env.load_gripper(gripper_type=args.gripper_types[0], gripper_size=1, 
                                               open_scales=open_scales, gripper_final_state=True)

    data = dict()
    rewards, scores = list(), list()
    ranks, steps = 0, 0
    
    while (input("[Action] +++> keep runing adagrasp to initial pose ? 9/0 for quit : ") not in ['9', '0']):
        
        pose_node.publish_ur_pose("APOSE", camera_node.end2base, repeat=True)

        if not input_signal("[Action] +++> Reach to initial pose to fetch tsdf? "): continue
        print(f'[Pipeline] ==> Fetching scene grid tsdf ...')
        scene_observation = camera_node.fetch_single_grid(grid_type='scene', issue_data=True)

        print(f'[Pipeline] ==> Getting inference affordance map ...')
        observation = {**gripper_observation, **scene_observation}
        affordance_maps = get_affordance([observation], model, device, gripper_final_state=(args.model_type=='adagrasp'))

        # Zero out predicted action values for all pixels outside workspace
        print(f'[Pipeline] ==> Obtaining action for gripper ...')
        s = affordance_maps.shape
        affordance_maps[np.logical_not(np.tile(observation['valid_pix_heightmap'],(s[0],s[1],s[2],1,1)))] = 0
        action, score, others = get_action(affordance_maps[0], 0, observation['open_scales'])
        
        ## Here execute the grasping outcomes
        grasp_pose, grasp_joints = grasppose_from_action(action, args.gripper_types[0])
        pose_node.dynamic_tf_publish(father_frame="grid_ws", child_frame="object", PosRotVec=grasp_pose)
        pose_objupper = pose_node.fetch_tf_state_posrot("base", "object_upper", True)
        pose_objgrip = pose_node.fetch_tf_state_posrot("base", "object_grip_endpos", True)
        pose_drop = [0.357, 0.466, 0.325, 2.11, -2.11, -0.196]

        action_flow = [
            {'name':'UR_TO_pose_upper', 'type':'APOSE', 'value':pose_objupper},
            {'name':'GRIPPER_INIT', 'type':'GRASP', 'value':grasp_joints},
            {'name':'UR_TO_pose_grip', 'type':'APOSE', 'value':pose_objgrip},
            {'name':'GRIPPER_CLOSE', 'type':'GRASP', 'value':grasp_joints},
            {'name':'UR_RE_pose_upper', 'type':'APOSE', 'value':pose_objupper},
            {'name':'UR_RE_pose_drop', 'type':'APOSE', 'value': pose_drop},
            {'name':'GRIPPER_OPEN', 'type':'GRASP', 'value':grasp_joints},
        ]
        print("Excute action list:", action_flow)
        excute_graspactions(action_flow, gripper_node, pose_node)

        ## Here return reward
        reward = input("[Reward] +++> Success or not [1 / Enter]: ") in ['1', '']
        rewards.append(reward)
        scores.append(score)

        # store data for visualization
        data[(ranks, steps)] = observation
        data[(ranks, steps)]['affordance_map'] = affordance_maps[0]
        data[(ranks, steps)]['open_scale_idx'] = others['open_scale_idx']
        data[(ranks, steps)]['grasp_pixel'] = np.array(action[:2])
        data[(ranks, steps)]['grasp_angle'] = action[2]
        data[(ranks, steps)]['score'] = score
        data[(ranks, steps)]['reward'] = reward
        steps += 1

    ## visualization
    print(f'[Pipeline] ==> {args.gripper_types[0]} test success rate: {np.mean(rewards)} and scores {np.mean(scores)}\n')
    print(f'[Pipeline] ==> Visualizing grasp results ...')
    num_open_scale = 1 if args.random_open_scale else args.num_open_scale
    ranks += 1
    if steps > 0: utils.visualization(data, ranks, steps, num_open_scale, args.num_rotations, ranks, args.save)
    print(f'[Pipeline] ==> Grasp pipeline finished.')


def grasppose_from_action(action, gripper_type):
    print("[INFO] Raw information of grasp: ", action)
    pixel_dim = 0.002
    vec_theta = - np.array([np.sin(action[2]), np.cos(action[2])])
    re_theta = np.arctan2(vec_theta[0], vec_theta[1])
    print("[INFO] Re theta angle of grasp: ", re_theta)
    original = SR.from_euler('xyz', [0, np.pi, np.pi/2], False)
    grasp_ori = SR.from_euler('xyz', [0, 0 , - re_theta], False)
    ori = (grasp_ori * original).as_rotvec()

    grasp_pose = [action[0] * pixel_dim, action[1] * pixel_dim, 0, ori[0], ori[1], ori[2]]

    if gripper_type == "robotiq_2f_85":
        grasp_joints = action[3] * 0.085
    elif gripper_type == "robotiq_3f":
        grasp_joints = action[3]
        raise NotImplementedError(f"{gripper_type} is not implemented yet.")
    else: raise KeyError("wrong gripper type for execution.")
    return grasp_pose, grasp_joints

def excute_graspactions(action_flow, gripper_client, pose_node):
    for action in action_flow:
        if not input_signal("\n" + action['type'] + " " + 
                            action['name'] + " {} Execute ? :".format(action['value'])): return
        if action['type'] == 'GRASP':
            gripper_client.grasp_execution(action['name'], action['value'])
        if action['type'] in ['APOSE', 'DPOSE']:
            pose_node.publish_ur_pose(action['type'], action['value'], repeat=True)

def input_signal(reminder="Enter to contine", supple_txt="[0:End, 9:Retn] "):
    a= input(supple_txt + reminder)
    if a == '0':
        print("\n\nTRY TO SAVE YOUR DATA IF YOU WANT IT !!!\n\n")
        exit(0)
    return (a != '9')


if __name__=='__main__':
    main()
