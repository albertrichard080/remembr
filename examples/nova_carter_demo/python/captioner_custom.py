import rclpy
import numpy as np
from PIL import Image as PILImage
import time
from threading import Thread
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from remembr.captioners.vila_captioner import VILACaptioner
import argparse

def create_args(model_path, prompt):
    """Function to create the argument dictionary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=prompt)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args([])  # Pass an empty list to avoid actual command-line arguments
    return args

class CaptionerNode(Node):
    
    def __init__(self):
        super().__init__("CaptionerNode")

        # Declare and retrieve ROS parameters
        self.declare_parameter("model", "Efficient-Large-Model/VILA1.5-3b")
        self.declare_parameter("image_topic", "/front_stereo_camera/left/image_raw")
        self.declare_parameter("caption_topic", "/caption")
        self.declare_parameter(
            "prompt",
            "<video> Please describe in detail what you see in the few seconds of " + \
            "the video. Specifically focus on the people, objects, environmental " + \
            "features, events/ectivities, and other interesting details. Think step " + \
            "by step about these details and be very specific."
        )
        self.declare_parameter("use_every_nth_image", 15)
        self.declare_parameter("caption_image_count", 6)
        self.declare_parameter("caption_interval", 3.0)

        # Initialize ROS topics
        self.image_subscriber = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.image_callback,
            10
        )

        self.caption_publisher = self.create_publisher(
            String,
            self.get_parameter("caption_topic").value,
            10
        )

        # Initialize class variables
        self.debug = False
        self.cv_bridge = CvBridge()
        self.prompt = self.get_parameter("prompt").value.strip("][()")
        self.use_every_nth_image = self.get_parameter("use_every_nth_image").value
        self.caption_image_count = self.get_parameter("caption_image_count").value
        self.caption_interval = self.get_parameter("caption_interval").value
        self.image_buffer = []
        self.image_counter = 0
        self.caption_loop_thread = None
        self.caption_loop_running = False

        # Get model argument from parameters
        model = self.get_parameter("model").value
        args = create_args(model, self.prompt)  # Pass both model and prompt to the args function

        # Pass args to VILACaptioner as it expects
        if not self.debug:
            self.model = VILACaptioner(args)  # Pass args as a dictionary

        self.logger = self.get_logger()

    def start_caption_loop(self):
        """Start the captioning loop in a separate thread."""
        if not self.caption_loop_running:
            self.caption_loop_running = True
            thread = Thread(target=self.caption_loop, daemon=True)
            thread.start()
            self.caption_loop_thread = thread

    def stop_caption_loop(self):
        """Stop the captioning loop gracefully."""
        self.caption_loop_running = False
        if self.caption_loop_thread:
            self.caption_loop_thread.join()
            self.caption_loop_thread = None

    def caption_loop(self):
        """Periodically generate and publish captions based on buffered images."""
        last_publish = time.perf_counter()

        while self.caption_loop_running:
            dt = time.perf_counter() - last_publish

            if dt < self.caption_interval:
                time.sleep(self.caption_interval - dt)

            # Use buffered images for captioning
            if len(self.image_buffer) < self.caption_image_count:
                self.logger.info("Skipped image captioning. Not enough images in buffer.")
                continue

            images = list(self.image_buffer)
            caption = self.caption_images(images)
            self.logger.info(f"Generated caption using {len(images)} images.")
            self.publish_caption(caption)

            last_publish = time.perf_counter()

    def image_callback(self, image_msg: Image):
        """Callback to process incoming image messages."""
        if self.image_counter % self.use_every_nth_image == 0:
            try:
                self.logger.info("Processing received image.")
                np_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'rgb8')
                pil_image = PILImage.fromarray(np_image)
                self.image_buffer.append(pil_image)

                # Keep the buffer size within the limit
                if len(self.image_buffer) > self.caption_image_count:
                    self.image_buffer.pop(0)
            except Exception as e:
                self.logger.error(f"Failed to process image: {e}")

        self.image_counter += 1

    def caption_images(self, images):
        """Generate a caption for a list of images."""
        if self.debug:
            return "Dummy caption"
    
        try:
            captions = []
            for image in images:
                # Call the caption method, not generate_caption
                caption = self.model.caption([image])  # Corrected to use the 'caption()' method
                captions.append(caption)
    
            # Combine captions into a single string
            combined_caption = " ".join(captions)
            max_length = 3000
            if len(combined_caption) > max_length:
                combined_caption = combined_caption[:max_length]
                self.logger.warning("Caption was truncated to fit the maximum length allowed by Milvus.")
            return combined_caption
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            return "Error generating caption"

    def publish_caption(self, caption: str):
        """Publish the generated caption to the ROS topic."""
        caption_msg = String()
        caption_msg.data = caption
        self.caption_publisher.publish(caption_msg)


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)
    node = CaptionerNode()

    try:
        node.start_caption_loop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down CaptionerNode.")
    finally:
        node.stop_caption_loop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

