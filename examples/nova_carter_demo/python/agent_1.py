import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped 
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.remembr_agent import ReMEmbRAgent
from scipy.spatial.transform import Rotation as R
from common_utils import format_pose_msg
import numpy as np
import traceback
import json

class AgentNode(Node):

    def __init__(self):
        super().__init__("AgentNode")

        self.declare_parameter("llm_type", "llama3.1:8b")
        self.declare_parameter("db_collection", "test_4_collection")
        self.declare_parameter("db_ip", "127.0.0.1")
        self.declare_parameter("query_topic", "/speech")
        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("goal_pose_topic", "/goal_pose")

        # Look for "robot" keyword
        self.query_filter = lambda text: "robot" in text.lower()

        self.query_subscriber = self.create_subscription(
            String,
            self.get_parameter("query_topic").value,
            self.query_callback,
            10
        )

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10
        )

        self.goal_pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_parameter("goal_pose_topic").value,
            10
        )

        self.memory = MilvusMemory(
            self.get_parameter("db_collection").value,
            self.get_parameter("db_ip").value
        )
        self.agent = ReMEmbRAgent(
            llm_type=self.get_parameter("llm_type").value
        )
        self.agent.set_memory(self.memory)  # Set memory for the agent

        self.last_pose = None
        self.logger = self.get_logger()

    def query_callback(self, msg: String):
        if not self.query_filter(msg.data):
            self.logger.info(f"Skipping query {msg.data} because it does not have keyword")
            return 

        try:
            query = msg.data

            # Add additional context information to query if a pose is available
            if self.last_pose is not None:
                position, angle, pose_time = format_pose_msg(self.last_pose)
                # Get the current time from the node's clock
                current_time_msg = self.get_clock().now().to_msg()
                # Format the current time as a string (e.g., "seconds.nanoseconds")
                current_time = f"{current_time_msg.sec}.{current_time_msg.nanosec}"
                query += f"\nYou are currently located at {position} and the time is {current_time}."

            # Run the Remembr Agent
            response = self.agent.query(query)
            
            # Handle the orientation (convert from string to list of floats)
            try:
                # Use ast.literal_eval for safe conversion if the orientation is a string representation
                import ast
                orientation = ast.literal_eval(response.orientation)
                quat = R.from_euler('z', orientation[2]).as_quat()  # Use only z-axis rotation if needed
                quat = np.squeeze(quat)
            except Exception as e:
                self.logger.error(f"Failed to parse orientation: {response.orientation}. Error: {e}")
                quat = [0.0, 0.0, 0.0, 1.0]  # Default to no rotation if parsing fails

            # Generate the goal pose from the response
            position = response.position
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = float(position[0])
            goal_pose.pose.position.y = float(position[1])
            goal_pose.pose.position.z = float(position[2])
            goal_pose.pose.orientation.x = float(quat[0])
            goal_pose.pose.orientation.y = float(quat[1])
            goal_pose.pose.orientation.z = float(quat[2])
            goal_pose.pose.orientation.w = float(quat[3])

            # Publish the result
            self.logger.info("Query executed: ")
            self.logger.info("\tText: " + response.text)
            self.logger.info(f"\tPosition: {response.position}")
            self.logger.info(f"\tOrientation: {response.orientation}")
        
            self.goal_pose_publisher.publish(goal_pose)
        except Exception as e:
            self.logger.error(f"FAILED. Returning: {str(e)}")
            self.logger.error(traceback.format_exc())
            return

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.last_pose = msg


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

