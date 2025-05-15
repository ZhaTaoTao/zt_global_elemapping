#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

class OdomToTf:
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/fixposition/odometry_enu", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp  # 关键：使用消息时间戳
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_footprint"
        # transform.child_frame_id = "odom"
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(transform)

if __name__ == "__main__":
    rospy.init_node("odom_to_tf")
    OdomToTf()
    rospy.spin()