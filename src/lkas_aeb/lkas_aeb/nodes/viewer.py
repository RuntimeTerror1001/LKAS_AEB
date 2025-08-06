#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

"""
VIEWER NODE
"""

class Viewer(Node):
    """
    Simple ROS2 Node for displaying camera feed in an OpenCV window.
    
    Subscribes to:
        - /carla/hero/rgb_view/image: Camera image feed for visualization
    """
    def __init__(self):
        super().__init__('viewer')

        # ========================
        # ROS COMMUNICATION SETUP
        # ========================
        
        self.image_sub = self.create_subscription(Image, '/carla/hero/rgb_view/image', self.image_cb, 10)

        # ========================
        # OPENCV SETUP
        # ========================
        
        self.bridge = CvBridge()
        self.window_name = 'Vehicle View'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def image_cb(self, msg):
        """
        Callback for displaying received images in OpenCV window.
        
        Args:
            msg (Image): ROS Image message to display
        """
        try:
            # Convert ROS image to OpenCV format and display
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imshow(self.window_name, cv_image)
            cv2.waitKey(1)
        except Exception as e :
            self.get_logger().error(f'Image Processing Error: {str(e)}')

    def __del__(self):
        """
        Destructor
        """
        cv2.destroyAllWindows()


# ========================
# MAIN FUNCTION
# ========================
def main(args=None):
    """
    Main entry point for the viewer node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = Viewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()