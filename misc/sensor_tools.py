import numpy as np
import scipy.spatial.transform as tf_tool
from geometry_msgs.msg import TransformStamped
import rospy
from scipy.spatial.transform import Rotation as SR
import open3d as o3d


def PosRotToTransMsg(father_frame, child_frame, translation, rot_vecter):
    trans_msg = TransformStamped()
    trans_msg.header.frame_id = father_frame
    trans_msg.child_frame_id = child_frame
    trans_msg.header.stamp = rospy.Time.now()

    trans_msg.transform.translation.x = translation[0]
    trans_msg.transform.translation.y = translation[1]
    trans_msg.transform.translation.z = translation[2]

    q = SR.from_rotvec(rot_vecter).as_quat()

    # angle = tf_conversions.transformations.vector_norm(rot_vecter)
    # rot_vecter_unit = tf_conversions.transformations.unit_vector(rot_vecter) if angle != 0 else (1, 0, 0)
    # q = tf_conversions.transformations.quaternion_about_axis(angle, rot_vecter_unit)

    trans_msg.transform.rotation.x = q[0]
    trans_msg.transform.rotation.y = q[1]
    trans_msg.transform.rotation.z = q[2]
    trans_msg.transform.rotation.w = q[3]
    return trans_msg


class Rotation(tf_tool.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])
    

class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.

    Attributes:
        rotation (scipy.spatial.transform.Rotation)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, tf_tool.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_dict(self):
        """Serialize Transform object into a dictionary."""
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)
    
    @classmethod
    def from_list_transrotvet(cls, list):
        translation = list[:3]
        rotation = Rotation.from_rotvec(list[3:])
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.

        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


class VoxelVolume(object):
    """Integration of multiple depth images for Voxel."""

    def __init__(self, dimensions, voxel_size, origin=np.array([0.0, 0.0, 0.0])):
        self.voxel_size = voxel_size
        self.origin = origin
        self.width = dimensions[0, 1] - dimensions[0, 0]
        self.height = dimensions[1, 1] - dimensions[1, 0]
        self.depth = dimensions[2, 1] - dimensions[2, 0]
        # setup dense voxel grid
        self.voxel_carving = o3d.geometry.VoxelGrid.create_dense(
            width=self.width,
            height=self.height,
            depth=self.depth,
            voxel_size=self.voxel_size,
            origin=origin,
            color=[1.0, 1.0, 1.0])
        

    def integrate(self, depth_img, intrinsic, extrinsic):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )
        extrinsic = extrinsic.as_matrix()
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = intrinsic
        param.extrinsic = extrinsic

        self.voxel_carving.carve_depth_map(o3d.geometry.Image(depth_img), param)

    def get_grid(self):
        voxels = self.voxel_carving.get_voxels()
        grid = np.zeros((1, int(self.width/self.voxel_size), int(self.height/self.voxel_size), 
                         int(self.depth/self.voxel_size)), dtype=np.float32)
        for voxel in voxels:
            i, j, k = voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]
            grid[0, i, j, k] = voxel.color[0]
        return grid
    

def create_voxel(dimensions, voxel_size, origin, depth_imgs, intrinsic, extrinsics):
    voxel = VoxelVolume(dimensions, voxel_size, origin)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        voxel.integrate(depth_imgs[i], intrinsic, extrinsic)
    return voxel
