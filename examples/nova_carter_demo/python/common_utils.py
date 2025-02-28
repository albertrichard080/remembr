from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

def format_pose_msg(msg: PoseWithCovarianceStamped):

    position = np.array([
        msg.pose.pose.position.x, 
        msg.pose.pose.position.y, 
        msg.pose.pose.position.z
    ])

    quat = np.array([
        msg.pose.pose.orientation.x, 
        msg.pose.pose.orientation.y, 
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ])

    angle = R.from_quat(quat).as_euler('xyz')[-1] # take z rotation #euler_rot_z

    stamp = msg.header.stamp #odom_msg.header.stamp
    converted_time = float(str(stamp.sec) + '.' + str(stamp.nanosec))
    

    return position, angle, converted_time
