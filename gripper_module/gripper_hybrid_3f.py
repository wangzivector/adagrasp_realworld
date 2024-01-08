import numpy as np
from gripper_module.gripper_base import GripperBase


class GripperHybrid3F(GripperBase):
    def __init__(self, bullet_client, gripper_size, gripper_height=None):
        r""" Initialization of Hybrid 3 finger gripper
        specific args for Hybrid:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__()

        self._bullet_client = bullet_client
        self._gripper_size = gripper_size

        # offset the gripper to a down facing pose for grasping
        _gripper_height = gripper_height if gripper_height is not None else 0.22
        print("Hybrid 3f _gripper_height is : ", _gripper_height)
        self._pos_offset = np.array([0, 0, _gripper_height * self._gripper_size]) # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi/2])

        self._finger_lower_bound = -0.785
        self._finger_upper_bound = 0.785

        self._palm_joint_ids = [0, 3, 6]
        self._prox_joint_ids = [1, 4, 7]
        self._dist_joint_ids = [2, 5, 8]
        
        # define force and speed (grasping)
        self._force = 100
        self._grasp_speed = 0.5

        
    def load(self, basePosition):
        gripper_urdf = "assets/gripper/hybrid_3f/urdf/hybrid_3f.urdf"
        body_id = self._bullet_client.loadURDF(
            gripper_urdf, flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=basePosition
        )
        # change color
        for link_id in range(-1, self._bullet_client.getNumJoints(body_id)):
           self._bullet_client.changeVisualShape(body_id, link_id, rgbaColor=[0.5, 0.5, 0.5, 1])
        return body_id
        
    
    def configure(self, mount_gripper_id, n_links_before):
        # Set friction coefficients for gripper fingers
        for i in range(n_links_before, self._bullet_client.getNumJoints(mount_gripper_id)):
            self._bullet_client.changeDynamics(mount_gripper_id,i,lateralFriction=1.0,spinningFriction=1.0,rollingFriction=0.0001,frictionAnchor=True)


    def step_constraints(self, mount_gripper_id, n_joints_before):
        # fix palm joints
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            [id+n_joints_before for id in self._palm_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[0.0] * len(self._palm_joint_ids),
            forces=[self._force] * len(self._palm_joint_ids),
            positionGains=[1.6] * len(self._palm_joint_ids)
        )

        # prox joints   
        pos_prox = self._bullet_client.getJointState(mount_gripper_id, self._prox_joint_ids[0]+n_joints_before)[0]
        joint_ids_prox = [i+n_joints_before for i in self._prox_joint_ids[1:]]
        target_states_prox = [pos_prox] * len(joint_ids_prox)
        
        # prox joints
        pos_dist = self._bullet_client.getJointState(mount_gripper_id, self._dist_joint_ids[0]+n_joints_before)[0]
        joint_ids_dist = [i+n_joints_before for i in self._dist_joint_ids[1:]]
        target_states_dist = [pos_dist] * len(joint_ids_dist)

        joint_ids_fingers = joint_ids_prox + joint_ids_dist
        target_states_fingers = target_states_prox + target_states_dist

        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            joint_ids_fingers,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_states_fingers, 
            forces=[self._force] * len(joint_ids_fingers),
            positionGains=[1] * len(joint_ids_fingers)
        )

        return target_states_fingers


    def open(self, mount_gripper_id, n_joints_before, open_scale):
        finger_range = self._finger_upper_bound - self._finger_lower_bound
        target_state_prox = self._finger_lower_bound + (1 - open_scale) * finger_range
        target_state_dist = self._finger_lower_bound + (open_scale) * finger_range
        
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            [self._prox_joint_ids[0]+n_joints_before, self._dist_joint_ids[0]+n_joints_before],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[target_state_prox, target_state_dist],
            forces=[self._force]*2,
            positionGains=[1] * 2
        )

        for i in range(240 * 4):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            self._bullet_client.stepSimulation()

    
    def close(self, mount_gripper_id, n_joints_before):
        
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._prox_joint_ids[0]+n_joints_before,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force,
        )
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._dist_joint_ids[0]+n_joints_before,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity= - self._grasp_speed,
            force=self._force,
        )

        for i in range(240 * 2):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            self._bullet_client.stepSimulation()

    
    def get_pos_offset(self):
        return self._pos_offset

    
    def get_orn_offset(self):
        return self._orn_offset


    def get_vis_pts(self, open_scale):
        finger_range = self._finger_upper_bound - self._finger_lower_bound
        open_length = finger_range * (open_scale) * 0.05
        return np.array([
            [open_length, 0.025],
            [open_length, -0.025],
            [-open_length, 0],
        ])