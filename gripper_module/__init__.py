# 2 fingers
from gripper_module.gripper_wsg_50 import GripperWSG50
from gripper_module.gripper_wsg_32 import GripperWSG32
from gripper_module.gripper_robotiq_2f_85 import GripperRobotiq2F85
from gripper_module.gripper_robotiq_2f_140 import GripperRobotiq2F140
from gripper_module.gripper_ezgripper import GripperEZGripper
from gripper_module.gripper_sawyer import GripperSawyer
from gripper_module.gripper_franka import GripperFranka
from gripper_module.gripper_rg2 import GripperRG2
from gripper_module.gripper_barrett_hand_2f import GripperBarrettHand2F
# # 3 fingers
from gripper_module.gripper_robotiq_3f import GripperRobotiq3F
from gripper_module.gripper_barrett_hand import GripperBarrettHand
from gripper_module.gripper_kinova_3f import GripeprKinova3F
from gripper_module.gripper_finray_2f import GripperFinray2F
from gripper_module.gripper_finray_3f import GripperFinray3F
from gripper_module.gripper_finray_4f import GripperFinray4F
from gripper_module.gripper_softpneu_3f import GripperSoftpneu3F
from gripper_module.gripper_rochu_2f import GripperRochu2F
from gripper_module.gripper_rochu_4f import GripperRochu4F
from gripper_module.gripper_hybrid_3f import GripperHybrid3F


def load_gripper(gripper_type):
    if gripper_type == 'wsg_50':
        return GripperWSG50
    elif gripper_type == 'wsg_32':
        return GripperWSG32
    elif gripper_type == 'robotiq_2f_85':
        return GripperRobotiq2F85
    elif gripper_type == 'robotiq_2f_140':
        return GripperRobotiq2F140
    elif gripper_type == 'ezgripper':
        return GripperEZGripper
    elif gripper_type == 'sawyer':
        return GripperSawyer
    elif gripper_type == 'franka':
        return GripperFranka
    elif gripper_type == 'rg2':
        return GripperRG2
    elif gripper_type == 'barrett_hand_2f':
        return GripperBarrettHand2F
    elif gripper_type == 'kinova_3f':
        return GripeprKinova3F
    elif gripper_type == 'robotiq_3f':
        return GripperRobotiq3F
    elif gripper_type == 'barrett_hand':
        return GripperBarrettHand
    elif gripper_type == 'finray_2f':
        return GripperFinray2F
    elif gripper_type == 'finray_3f':
        return GripperFinray3F
    elif gripper_type == 'finray_4f':
        return GripperFinray4F
    elif gripper_type == 'softpneu_3f':
        return GripperSoftpneu3F
    elif gripper_type == 'rochu_2f':
        return GripperRochu2F
    elif gripper_type == 'rochu_4f':
        return GripperRochu4F
    elif gripper_type == 'hybrid_3f':
        return GripperHybrid3F
    else:
        raise NotImplementedError(f'Does not support {gripper_type}')