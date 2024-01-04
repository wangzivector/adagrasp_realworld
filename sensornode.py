#!/usr/bin/env python

## Azure camera
## roslaunch azure_kinect_ros_driver driver.launch depth_mode:=NFOV_UNBINNED color_resolution:=720P
## roslaunch azure_kinect_ros_driver kinect_rgbd.launch depth_mode:=WFOV_2X2BINNED color_resolution:=720P
## roslaunch azure_kinect_ros_driver kinect_rgbd.launch depth_mode:=NFOV_UNBINNED color_resolution:=720P

## cd ~/SpatialHybridGen/codes/adagrasp && ca adacopygen && source ~/tf_catkin_ws/devel/setup.sh

import numpy as np
import rospy
import ros_numpy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from fusion import TSDFVolume
from misc.sensor_tools import create_voxel
from misc.sensor_tools import CameraIntrinsic
from misc.sensor_tools import Transform
from misc.sensor_tools import PosRotToTransMsg

import tf2_ros


"""
Azure RGBD ROS Node for grid
"""
class SensorServer:
    def __init__(self):
        rospy.init_node("Sensor_server")
        
        self.voxel_size = 0.002
        self.voxel_disc = np.array([192, 192, 64])
        self.voxel_bound = np.array([[-0.192, 0.192], # 3x2 rows: x,y,z cols: min,max
                                      [-0.192, 0.192],
                                      [ 0.000, 0.128]])
        
        ## For visualization tfs
        self.grid2base = [0.5, 0.1, -0.01, 0.0, 0.0, 0.0]
        self.end2base = [0.4135, 0.1289, 0.5320, 1.94, -1.953, -0.475]
        self.end2base_joints = [-3.152, -1.6026, 1.400, -0.8803, 4.7004, 0.00173]
        self.cam2end = [0.03200, -0.11724, 0.08453, 0.48, 0.0, 0.0]

        self.auto_visualize_grid_flag = False
        self.show_grid_with_issue = True
        
        self.grid2base_tranf = None
        self.end2base_tranf = None
        self.cam2end_tranf = None

        depth_topic = '/depth_to_rgb/image_raw'
        color_topic = '/rgb/image_raw'
        caminfo_topic = '/depth_to_rgb/camera_info'
        points_topic = '/depth/points'
        grid_topic = '/grid_raw'
        grid_request_topic = '/grid_request_topic'

        self.depth_buff = None
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(depth_topic))

        self.color_buff = None
        rospy.Subscriber(color_topic, Image, self.color_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(color_topic))

        self.caminfo_buff = None
        rospy.Subscriber(caminfo_topic, CameraInfo, self.caminfo_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(caminfo_topic))

        self.points_buff = None
        rospy.Subscriber(points_topic, PointCloud2, self.points_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(points_topic))

        self.grid_request = rospy.Subscriber(grid_request_topic, Float32MultiArray, self.grid_request_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(grid_request_topic))

        self.grid_pub = rospy.Publisher(grid_topic, Float32MultiArray, queue_size=1, latch=True)
        self.reset_depthpose_buff()
        self.TF_stuff_manage()

    def TF_stuff_manage(self):
        grid_origin = np.array(self.grid2base)
        grid_origin[:3] += self.voxel_bound[:, 0].T
        
        static_br = tf2_ros.StaticTransformBroadcaster()
        trans_wor2base = PosRotToTransMsg("world", "base", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        trans_ee2end = PosRotToTransMsg("ee_link", "armend", [0, 0, 0], [-1.20919958, 1.20919958, -1.20919958])
        trans_end2cam = PosRotToTransMsg("armend", "rgb_camera_link", self.cam2end[:3], self.cam2end[3:])
        trans_world2gridws = PosRotToTransMsg("world", "grid_ws", grid_origin[:3], grid_origin[3:])
        trans_cam2sen = PosRotToTransMsg("rgb_camera_link", "camera", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        static_br.sendTransform([trans_wor2base, trans_ee2end, trans_end2cam, trans_world2gridws, trans_cam2sen])
        rospy.sleep(0.2)
        self.grid2base_tranf = Transform.from_list_transrotvet(self.grid2base)
        self.end2base_tranf = Transform.from_list_transrotvet(self.end2base)
        self.cam2end_tranf = Transform.from_list_transrotvet(self.cam2end)

    def reset_depthpose_buff(self):
        self.g3d_extrinsics_buff = None
        self.g3d_depth_buff = None

    def append_depthpose_buff(self, end2base_k):
        extrinsics = np.expand_dims(self.convert_extrinsics(end2base_k), axis=0)
        depth_image = np.expand_dims(self.fetch_depth_image(), axis=0)

        if self.g3d_extrinsics_buff is None: self.g3d_extrinsics_buff = extrinsics
        else: self.g3d_extrinsics_buff = np.vstack((self.g3d_extrinsics_buff, extrinsics))

        if self.g3d_depth_buff is None: self.g3d_depth_buff = depth_image
        else: self.g3d_depth_buff = np.vstack((self.g3d_depth_buff, depth_image))

    def grid_request_callback(self, msg):
        data = np.array(msg.data)
        if data.mean() == 0:
            rospy.loginfo('[Sensor]: clear buff by msg: {}'.format(data))
            self.reset_depthpose_buff()
        elif data.mean() == 1:
            rospy.loginfo('[Sensor]: process tsdf buff by msg: {}'.format(data))
            self.issue_grid_using_buff(grid_type='tsdf')
        elif data.mean() == 2:
            rospy.loginfo('[Sensor]: process voxel buff by msg: {}'.format(data))
            self.issue_grid_using_buff(grid_type='voxel')
        else:
            rospy.loginfo("Buffing depth with pose data: \n{}".format(data))
            self.append_depthpose_buff(end2base_k=data)

    def depth_callback(self, msg):
        self.depth_buff = msg

    def color_callback(self, msg):
        self.color_buff = msg

    def caminfo_callback(self, msg):
        self.caminfo_buff = msg

    def points_callback(self, msg):
        self.points_buff = msg

    def fetch_depth_image(self):
        ## Depth Image
        while self.depth_buff is None:
            rospy.loginfo("[Sensor]: depth_image buff is None, retrying...")
            rospy.sleep(0.5)
        
        depth_image = ros_numpy.image.image_to_numpy(self.depth_buff).astype(np.float32)
        depth_image = np.nan_to_num(depth_image)
        # rospy.loginfo('Depth_image size: {}'.format(depth_image.shape))
        depth_image = self.depth_inpaint(depth_image, missing_value = 0)
        if depth_image.mean() > 1: depth_image = depth_image / 1000.0
        return depth_image

    def fetch_color_image(self):
        ## Color Image
        while self.color_buff is None:
            rospy.loginfo("[Sensor]: color_buff buff is None, retrying...")
            rospy.sleep(0.50)
        color_image = ros_numpy.image.image_to_numpy(self.color_buff).astype(np.float32)
        return color_image

    @staticmethod
    def depth_inpaint(image, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in the depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (image == missing_value).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        imax, imin = np.abs(image).max(), np.abs(image).min()
        irange = imax - imin
        image = ((image - imin) / irange).astype(np.float32) # Has be float32, 64 not supported. get -1:1
        image = cv2.inpaint(image, mask, 2, cv2.INPAINT_NS) # repair with fluid alg. radius 1
        # Back to original size and value range.
        image = image[1:-1, 1:-1] # cut the 1 pixel boarder
        image = image.astype(np.float32) * irange + imin
        return image

    def fetch_intrinsic(self):
        ## Camera Info
        while self.caminfo_buff is None:
            rospy.loginfo("[Sensor]: depth_image buff is None, retrying...")
            rospy.sleep(1.0)
        else: cam_info = self.caminfo_buff
        fx, fy, cx, cy = cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]
        width, height = cam_info.width, cam_info.height
        intrinsics = CameraIntrinsic(width, height, fx, fy, cx, cy)
        return intrinsics

    def fetch_extrinsics(self):
        # depth_camera_link
        # use the frame_id in self.extrinsics_info to get the transform between base and depth cam

        transformation = self.grid2base_tranf.inverse() * self.end2base_tranf * self.cam2end_tranf
        return (transformation.inverse()).to_list()

    def convert_extrinsics(self, end2base_k):
        # depth_camera_link
        # use the frame_id in self.extrinsics_info to get the transform between base and depth cam
        end2base_k_tranf = Transform.from_list_transrotvet(end2base_k)
        transformation = self.grid2base_tranf.inverse() * end2base_k_tranf * self.cam2end_tranf
        return (transformation.inverse()).to_list()

    def fetch_pointcloud(self):
        if self.depth_buff is None: return None
        return self.points_buff
        # pip install ros_numpy
        # ros_numpy.point_cloud2.get_xyz_points(cloud_array, remove_nans = True)
        # return 3xN matrix
        raise NotImplementedError("please wait to implement")
        points = [] # pointlist
        points = Transform.from_matrix(self.grid_shift).transform_point(points)

    def fetch_single_grid(self, grid_type, issue_data=False):
        # return np.load("data/samples/test_sample_scene_tsdf.npy")
        intrinsic = self.fetch_intrinsic()
        depth_imgs = np.expand_dims(self.fetch_depth_image(), axis=0)
        color_img = self.fetch_color_image()
        extrinsics = np.expand_dims(self.fetch_extrinsics(), axis=0)
        grid_data = self.gen_grid(color_img, depth_imgs, intrinsic, extrinsics, grid_type)
        if issue_data: self.show_grid(grid_data['scene_tsdf'] if grid_type=='scene' else grid_data)
        return grid_data

    def issue_grid(self, grid_data):
        msg = Float32MultiArray(data=grid_data.astype(np.float32).reshape(-1))
        self.grid_pub.publish(msg)
        rospy.loginfo("[Sensor]: issuing grid : {}".format(grid_data.shape))

    def gen_grid(self, color_img, depth_imgs, intrinsic, extrinsics, grid_type):
        if grid_type == 'voxel':
            voxel = create_voxel(self.voxel_bound, self.voxel_size, self.voxel_bound[:, 0].T, 
                                 depth_imgs, intrinsic, extrinsics)
            grid_data = voxel.get_grid()
            return grid_data
        scene_data = self.create_scene(volume_bounds=self.voxel_bound, voxel_size=self.voxel_size, 
                                        depth_image=depth_imgs[0], cam_intrinsic=intrinsic.K, 
                                        cam_pose_matrix=extrinsics[0], color_img=color_img, num_cam=1)
        if grid_type == 'tsdf':
            return scene_data['scene_tsdf']
        elif grid_type == 'scene':
            return scene_data

    @staticmethod
    def create_scene(volume_bounds, voxel_size, depth_image, cam_intrinsic, cam_pose_matrix, color_img, num_cam=1):
        _scene_tsdf = TSDFVolume(volume_bounds, voxel_size=voxel_size, use_gpu=False)
        if num_cam == 1:
            obstacle_mask = np.zeros_like(np.stack((depth_image, depth_image, depth_image), axis=-1))
            cam_extrinsic = Transform.from_list(cam_pose_matrix).inverse().as_matrix()
            _scene_tsdf.integrate(obstacle_mask, depth_image, cam_intrinsic, cam_extrinsic, obs_weight=1.)
        else:
            raise NotImplementedError("TSDF for multiple cameras to be implemented.")
        # get scene_tsdf(WxHxD) and obstacle_vol(WxHxDx1)
        scene_tsdf, obstacle_vol = _scene_tsdf.get_volume()

        # make the empty space 0 in obstacle_vol
        obstacle_vol *= (scene_tsdf < 0).astype(np.int32)
        
        # scene_tsdf = np.transpose(scene_tsdf, [1, 0, 2]) # swap x-axis and y-axis to make it consitent with heightmap
        # obstacle_vol = np.transpose(obstacle_vol, [1, 0, 2])
        
        chn_1d = (np.clip(np.mean(scene_tsdf, axis=2), 0 ,1) * 255).astype(np.uint8)
        valid_pix_heightmap, marg_size, map_size = np.zeros_like(chn_1d), 40, chn_1d.shape[0]
        valid_pix_heightmap[marg_size: map_size-marg_size, marg_size: map_size-marg_size] = 1
        valid_pix_heightmap = valid_pix_heightmap > 0.5
        scene_observation = {
            'scene_tsdf': scene_tsdf,
            'obstacle_vol': obstacle_vol,
            'valid_pix_heightmap': valid_pix_heightmap, # np.load("data/samples/test_sample_valid_pix_heightmap.npy"),
            'color_heightmap': np.stack([chn_1d, chn_1d, chn_1d], axis=-1),
            'target_heightmap': np.zeros_like(chn_1d),
            'color_image': color_img
        }
        return scene_observation
    
    def issue_grid_using_buff(self, grid_type):
        intrinsic = self.fetch_intrinsic()
        depth_imgs = self.g3d_depth_buff
        extrinsics = self.g3d_extrinsics_buff
        frame_size = extrinsics.shape[0]
        self.reset_depthpose_buff()
        grid_data = self.gen_grid(depth_imgs, intrinsic, extrinsics, grid_type)
        rospy.loginfo("[Sensor]: issuing {} with buff size: {}".format(grid_type, frame_size))
        self.issue_grid(grid_data)

        if self.show_grid_with_issue: self.show_grid(grid_data)
        rospy.loginfo("[Sensor]: visualize buffed {} data".format(grid_type))
        return grid_data

    def show_grid(self, grid):
        import spahybgen.utils.utils_rosvis as ut_vis
        ut_vis.clear_tsdf('grid_ws')
        ut_vis.draw_tsdf(grid, self.voxel_size, threshold=-1, frame_id = 'grid_ws')
    
    def visualize_single_grid(self, _):
        if not self.auto_visualize_grid_flag: return
        self.show_grid(self.fetch_single_grid(self.default_visual_grid_type))

    def ur_pose_state_CB(self, msg):
        # if not self.auto_visualize_grid_flag: return
        if msg.header.frame_id != "base2end":
            rospy.logwarn("frame_id is not right: {}".format(msg.header.frame_id))
            return
        translation = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        rotationvet = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        self.end2base_tranf = Transform.from_list_transrotvet(translation + rotationvet)

    def auto_visualize_grid(self, rate, grid_type):
        self.default_visual_grid_type = grid_type
        from geometry_msgs.msg import PoseStamped
        self.sub_servo = rospy.Subscriber("/ur_pose_state", PoseStamped, self.ur_pose_state_CB)
        import spahybgen.utils.utils_rosvis as ut_vis
        ut_vis.draw_workspace(0.4, frame="grid_ws")
        rospy.Timer(rospy.Duration(1/rate), self.visualize_single_grid)


if __name__ == '__main__':
    ## Ctrl-C stop stuff
    import sys, signal
    def signal_handler(signal, frame): sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    azure_node = SensorServer()
    scene_data = azure_node.fetch_single_grid(grid_type='scene')
    print(f"scene_tsdf shape generated: {scene_data['scene_tsdf'].shape}")

    while not rospy.is_shutdown():
        a= input("1/0 switch auto_visualize_grid ON, q for quit :")
        if a=='1': grid_type='tsdf'
        if a=='2': grid_type='voxel'
        if a=='q': break

        loop_rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            scene_tsdf = azure_node.fetch_single_grid(grid_type=grid_type)
            azure_node.show_grid(scene_tsdf)
            loop_rate.sleep()


