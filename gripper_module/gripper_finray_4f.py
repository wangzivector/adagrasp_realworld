import numpy as np
from gripper_module.gripper_base import GripperBase


class GripperFinray4F(GripperBase):
    def __init__(self, bullet_client, gripper_size, gripper_height=None):
        r""" Initialization of Finray 3 finger gripper
        specific args for Finray:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__()

        self._bullet_client = bullet_client
        self._gripper_size = gripper_size*0.8

        # offset the gripper to a down facing pose for grasping
        _gripper_height = gripper_height if gripper_height is not None else 0.12
        print("Finray _gripper_height is : ", _gripper_height)
        self._pos_offset = np.array([0, 0, _gripper_height * self._gripper_size]) # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi/2])

        self._finger_open_distance = 0.7

        self._moving_joint_ids = [0, 1, 2, 3]
        
        # define force and speed (grasping)
        self._force = 100
        self._grasp_speed = 0.3


    def load(self, basePosition):
        gripper_urdf = "assets/gripper/finray4f/urdf/finray4f.urdf"
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
        pos = self._bullet_client.getJointState(mount_gripper_id, self._moving_joint_ids[0]+n_joints_before)[0]
        joint_ids = [i+n_joints_before for i in self._moving_joint_ids[1:]]
        target_states = [pos] * len(joint_ids)
        
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_states, 
            forces=[self._force] * len(joint_ids),
            positionGains=[1] * len(joint_ids)
        )
        return pos

    def open(self, mount_gripper_id, n_joints_before, open_scale):
        target_state = self._finger_open_distance * (1 - open_scale)
        
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._moving_joint_ids[0]+n_joints_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=target_state,
            force=self._force
        )

        for i in range(240 * 4):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            self._bullet_client.stepSimulation()

    
    def close(self, mount_gripper_id, n_joints_before):
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._moving_joint_ids[0]+n_joints_before,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force
        )

        for i in range(240 * 4):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            self._bullet_client.stepSimulation()


    def get_pos_offset(self):
        return self._pos_offset


    def get_orn_offset(self):
        return self._orn_offset


    def get_vis_pts(self, open_scale):
        open_length = self._finger_open_distance * (open_scale) *0.08
        return np.array([
            [-open_length, 0.025],
            [open_length, 0.025],
            [open_length, -0.025],
            [-open_length, -0.025],
        ])