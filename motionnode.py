### TF tree and robot execution management, including UR5e
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as SR

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int16MultiArray
from geometry_msgs.msg import Vector3Stamped


"""
URClient Node
"""
class PoseClient:
    def __init__(self):
        ### Init static pose sequence
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.static_br = tf2_ros.StaticTransformBroadcaster()
        self.dynamic_br = tf2_ros.TransformBroadcaster()
        rospy.sleep(0.1) # [pipeline]: Awaiting all TCP connection done.

        self.ur_pose_pub = rospy.Publisher('/pose_servo_cmd', PoseStamped, queue_size=1)
        self.sub_servo = rospy.Subscriber("/ur_pose_state", PoseStamped, self.ur_pose_state_CB)
        rospy.sleep(0.1) # [pipeline]: Awaiting all TCP connection done.


    def characterize_grasp_transform(self, armend2gripper, gripper2grip, gripper_tfname):
        upper_offset_height = - (0.1 + gripper2grip)
        grip_offset_height = - gripper2grip
        trans_gri2grispe = self.PosRotToTransMsg("gripper", gripper_tfname, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        trans_gri2grispe_vis = self.PosRotToTransMsg("object_grip", gripper_tfname + '_vis', [0.0, 0.0, 0], [0.0, 0.0, 0.0])
        trans_obj2upp = self.PosRotToTransMsg("object", "object_upper", [0.0, 0.0, upper_offset_height], [0.0, 0.0, 0.0]) # motion prepare pose
        trans_obj2gri = self.PosRotToTransMsg("object", "object_grip", [0.0, 0.0, grip_offset_height], [0.0, 0.0, 0.0]) # grip excution pose
        trans_end2gri = self.PosRotToTransMsg("armend", "gripper", [0.0, 0.0, armend2gripper], [0.0, 0.0, 0.0])
        trans_end2gri_rev4vis = self.PosRotToTransMsg("object_grip", "object_grip_endpos", [0.0, 0.0, - armend2gripper], [0.0, 0.0, 0.0])

        self.static_br.sendTransform([trans_obj2upp, trans_end2gri, trans_obj2gri, trans_gri2grispe, 
                                     trans_gri2grispe_vis, trans_end2gri_rev4vis])
        rospy.sleep(0.2)

    def ur_pose_state_CB(self, msg):
        if msg.header.frame_id != "base2end":
            rospy.logwarn("frame_id is not right: {}".format(msg.header.frame_id))
            return
        translation = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        rotationvet = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        self.dynamic_tf_publish("base", "armend", translation + rotationvet)

    def publish_ur_pose(self, pose_type, pose, repeat=False):
        pose_to_send = PoseStamped()
        pose_to_send.header.frame_id = "COM_" + pose_type
        if pose_type in ['APOSE', 'POSE', 'JOINT', "AJOINT"]:
            pose_to_send.pose.position.x = pose[0]
            pose_to_send.pose.position.y = pose[1]
            pose_to_send.pose.position.z = pose[2]
            # direct axis-angle rotation 
            pose_to_send.pose.orientation.x = pose[3]
            pose_to_send.pose.orientation.y = pose[4]
            pose_to_send.pose.orientation.z = pose[5]
            pose_to_send.pose.orientation.w = 1
            # print(pose_type, pose_to_send)
        else: raise KeyError("pose_type for publish_ur_pose() is wrong : {}".format(pose_type))

        self.ur_pose_pub.publish(pose_to_send)
        if repeat: 
            rospy.sleep(0.1)
            self.ur_pose_pub.publish(pose_to_send)
        # print(pose_type, pose_to_send)

    def dynamic_tf_publish(self, father_frame, child_frame, PosRotVec):
        translation, rotvec = PosRotVec[:3], PosRotVec[3:]
        trans_dynamic = self.PosRotToTransMsg(father_frame, child_frame, translation, rotvec)
        self.dynamic_br.sendTransform(trans_dynamic)
        rospy.sleep(0.1)
        self.dynamic_br.sendTransform(trans_dynamic)
    
    def fetch_tf_state_posquat(self, target_frame, source_frame, wait_block=True):
        """
        Transformation from source_frame to target_frame
        """
        loop_rate_rece = rospy.Rate(2)
        while(not rospy.is_shutdown()):
            try:
                trans_listener = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time())
                return trans_listener
            except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Fail to fetch transformation... from {} to {}".format(target_frame, source_frame))
                if wait_block is False: return None

                loop_rate_rece.sleep()
                continue

    def fetch_tf_state_posrot(self, target_frame, source_frame, wait_block=True):
        """
        Transformation from source_frame to target_frame
        Frame: "world", "base", "armend", "camera", "gripper", "object", "object_upper", "object_grip"
        """
        Trans_msg = self.fetch_tf_state_posquat(target_frame, source_frame, wait_block)
        Pos_rota = self.TransMsgToPosRot(Trans_msg) if Trans_msg is not None else None
        return Pos_rota

    @staticmethod
    def PosRotToTransMsg(father_frame, child_frame, translation, rot_vecter):
        trans_msg = TransformStamped()
        trans_msg.header.frame_id = father_frame
        trans_msg.child_frame_id = child_frame
        trans_msg.header.stamp = rospy.Time.now()

        trans_msg.transform.translation.x = translation[0]
        trans_msg.transform.translation.y = translation[1]
        trans_msg.transform.translation.z = translation[2]

        q = SR.from_rotvec(rot_vecter).as_quat()

        trans_msg.transform.rotation.x = q[0]
        trans_msg.transform.rotation.y = q[1]
        trans_msg.transform.rotation.z = q[2]
        trans_msg.transform.rotation.w = q[3]
        return trans_msg

    @staticmethod
    def TransMsgToPosRot(trans_msg):
        translation = [
            trans_msg.transform.translation.x, 
            trans_msg.transform.translation.y, 
            trans_msg.transform.translation.z
        ]
        quat_num = [
            trans_msg.transform.rotation.x,
            trans_msg.transform.rotation.y,
            trans_msg.transform.rotation.z,
            trans_msg.transform.rotation.w
        ]
        # rot_vecter = tf_conversions.Quaternion(trans_msg.transform.rotation)
        quat = SR.from_quat(quat_num)
        rot_vecter = quat.as_rotvec().tolist()
        pos_rot = translation + rot_vecter
        return pos_rot
    
    @staticmethod
    def PosRotVec2TransMat(pose_posrotvec):
        pose_rotation_SR = SR.from_rotvec(pose_posrotvec[3:])
        position_vec = np.array([pose_posrotvec[:3]]).T
        homon_vec = np.array([[0.0, 0.0, 0.0, 1.0]])
        tran_mat  = np.concatenate((np.concatenate((pose_rotation_SR.as_dcm(), position_vec), axis=1), homon_vec), axis=0)
        return tran_mat


"""
Gripper Client Node
"""
class GripperClient:
    def __init__(self, gripper):
        """
        Grippers: "robotiq2f", "robotiq3f", "finray2f"
        """
        gripper_name_remap = {
            "robotiq_2f_85": "robotiq2f",
            "robotiq_3f": "robotiq3f",
            "finray_2f": "finray2f",
        }
        self.gripper = gripper_name_remap[gripper]

        ServerHint = {
            "robotiq2f": "roslaunch robotiq_2f_gripper_control external_robotiq_msgctl.launch",
            "robotiq3f": "roslaunch robotiq_3f_gripper_control robotiq_3f_gripper_TCP_nodes.launch",
            "finray2f": "Open bluetooth and cmd: rosrun rosserial_python serial_node.py /dev/rfcomm0",
        }

        ExecutionFun = {
            "robotiq2f": self.robotiq2f_execution,
            "robotiq3f": self.robotiq3f_execution,
            "finray2f": self.finray2f_execution,
        }
        
        PublisherMsgName = {
            "robotiq2f":  "gripper_action_" + "ROBOTIQ_2F",
            "robotiq3f":  "gripper_action_" + "ROBOTIQ_3F",
            "finray2f":  "easy_gripper_cmd",
        }

        PublisherMsgType = {
            "robotiq2f": Float32MultiArray,
            "robotiq3f": Int16MultiArray,
            "finray2f": Vector3Stamped,
        }

        GripperTFName = {
            "robotiq2f": 'gripper_robotiq_2f',
            "robotiq3f": 'gripper_robotiq_3f',
            "finray2f": 'gripper_finray_2f',
        }

        Armend2Gripper = {
            "robotiq2f": 0.01,
            "robotiq3f": 0.00,
            "finray2f": 0.00,
        }

        Gripper2Grip = {
            "robotiq2f": 0.18,
            "robotiq3f": 0.26,
            "finray2f": 0.205,
        }

        self.gripper_pulisher = rospy.Publisher(PublisherMsgName[self.gripper], PublisherMsgType[self.gripper], queue_size=1)
        
        self.gripper_tfname = GripperTFName[self.gripper]
        self.armend2gripper = Armend2Gripper[self.gripper]
        self.gripper2grip = Gripper2Grip[self.gripper]
        
        self.grasp_execution = ExecutionFun[self.gripper]

        rospy.loginfo("[Gripper]: {} server: \n{}".format(self.gripper, ServerHint[self.gripper]))
    

    def robotiq2f_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m] 
        palm_position: the position of angle or distance of some grippers
        """
        data_to_send = Float32MultiArray()
        maximal_open = 0.085
        open_bias = 0.035
        position = joints
        
        if action_name == "GRIPPER_CLOSE": position = position / 10
        if action_name == "GRIPPER_OPEN": position = maximal_open

        if position > maximal_open: position = maximal_open
        force, speed = 40.0, 0.01
        data_to_send.data = [position, speed, force]
        self.gripper_pulisher.publish(data_to_send)
        rospy.logwarn("Robotiq 2f openwidth: {} m".format(position))


    def robotiq3f_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m] 
        palm_position: the position of angle or distance of some grippers
        """
        joint_position = 1 - joints
        force = 100
        speed = 255
        palm_position = 140
        position_open = 10
        position_close = 120
        range_pp = position_close - position_open
        data_to_send = Int16MultiArray()
        if action_name == "GRIPPER_CLOSE": opening_position = position_close+50
        elif action_name == "GRIPPER_OPEN": opening_position = position_open
        else: opening_position = int(range_pp * joint_position + position_open)

        data_action = [opening_position, palm_position, speed, force] # all are 0-255
        data_to_send.data = data_action
        self.gripper_pulisher.publish(data_to_send)



    def finray2f_execution(self, action_name, joints):
        """
        rostopic pub /easy_gripper_cmd geometry_msgs/Vector3Stamped '{header: {frame_id:  STEP},  vector: {x: .0}}'  -1
        """
        dis_opening = 0.12
        init_opening = 0.0

        opening_distance = joints
        # force = 30.0
        # speed = 0.01
        data_to_send = Vector3Stamped()
        data_to_send.header.frame_id = "STEP"

        # data_to_send.data = [0.04, 0.01, 20.0] # position, speed, force
        
        if action_name == "GRIPPER_CLOSE": opening_position = init_opening
        elif action_name == "GRIPPER_OPEN": opening_position = dis_opening
        else: opening_position = opening_distance

        if opening_position > dis_opening: opening_position = dis_opening
        
        data_to_send.vector.x = opening_position / dis_opening
        self.gripper_pulisher.publish(data_to_send)