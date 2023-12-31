# cd /home/smarnlab/SpatialHybridGen/codes/adagrasp_realworld && source ~/tf_catkin_ws/devel/setup.bash && ca adacopygen
# python learntest.py

from sim import RealWorldServer
import numpy as np


def main():
    # Work
    # gt = "robotiq_2f_85"
    # env = RealWorldServer(gui_enabled=True, num_cam=1)
    # open_scales = np.linspace(0.5, 1.0, 1, True)
    # gripper_observation = env.fetch_gripper_tsdf(gripper_type=gt, gripper_size=1, 
    #                                            open_scales=open_scales, gripper_final_state=True)
    # input("Terminate software.? ")

    # Create worker for observationsteps
    print(f'[Pipeline] ==> Loading RealWorldServer ...')
    env = RealWorldServer(gui_enabled=True, num_cam=1, gripper_home_position=[0, 0, 0]) # , gripper_home_position=[0, 0, 0]
    open_scales = np.linspace(0.5, 1.0, 5, True)
    gt = "finray_2f" # robotiq_2f_85 robotiq_3f finray_2f
    gripper = env.load_gripper(gripper_type=gt, gripper_size=1, open_scales=open_scales, gripper_final_state=True)

    ## Take picture
    _home_position = env._gripper_home_position
    _tsdf_size = [64, 64, 32]
    _voxel_size = 0.004
    bond = np.array(_tsdf_size) * _voxel_size
    _vol_bnds = np.array([[-bond[0]/2, bond[0]/2], [-bond[1]/2, bond[1]/2], [0, bond[2]]])
    _vol_bnds += np.array(_home_position).reshape(3, -1)
    cam_up_direction = [0, 0, 1]
    side_look_directions = np.linspace(0, 2*np.pi, num=16, endpoint=False)
    cam_distance = 1
    cam_lookat = _vol_bnds.mean(1)
    for direction in side_look_directions:
        cam_position = [
            _home_position[0] + cam_distance * np.cos(direction),
            _home_position[1] + cam_distance * np.sin(direction),
            _home_position[2]
        ]
        # input("Continue software. -> get_gripper_cam_data(cam_position) : ")
        color_image, depth_image, _, cam_pose_matrix = gripper.get_gripper_cam_data(cam_position, cam_lookat, cam_up_direction)
    input("Close -> Continue software. -> get_gripper_cam_data([0, 0, 2) : ")
    gripper.close()
    color_image, depth_image, _, cam_pose_matrix = gripper.get_gripper_cam_data([0, 0, 2], cam_lookat, [1, 0, 0]) # top 
    input("Open 0 -> Continue software. -> get_gripper_cam_data([0, 0, 0]) : ")
    gripper.open(open_scale=0)
    color_image, depth_image, _, cam_pose_matrix = gripper.get_gripper_cam_data([0, 0, 0], cam_lookat, [1, 0, 0]) # bottom

    ## Motion
    while True:
        open_scale = input("open_scale : ")
        if open_scale == 'q':
            break
        gripper.open(open_scale=float(open_scale))

    input("Terminate software.? ")


if __name__=='__main__':
    main()