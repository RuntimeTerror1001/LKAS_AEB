import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ========================
# CONSOLE COLORS FOR LOGGING
# ========================
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"

"""
VEHICLE MARKER VISUALIZATION NODE
"""

class CarleVehicleMarker(Node):
    """
    ROS2 node that creates visualization markers for vehicle position and trajectory.
    
    Subscribes to:
    - /carla/hero/odometry: Vehicle position and orientation
    - /carla/hero/waypoints: Planned path waypoints
    
    Publishes:
    - /carla/hero/vehicle_marker: MarkerArray for RViz visualization
    
    Creates rectangular markers representing the vehicle at each odometry update.
    """
    
    def __init__(self):
        super().__init__('carla_vehicle_marker')

        # ========================
        # STATE TRACKING
        # ========================
        self.odom_received = False
        self.waypoints_received = False

        # ========================
        # ROS COMMUNICATION SETUP
        # ========================
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/hero/odometry', self.odom_cb, 10
        )
        self.waypoint_sub = self.create_subscription(
            Path, '/carla/hero/waypoints', self.waypoints_cb, 10
        )

        # Publishers
        self.vehicle_marker_pub = self.create_publisher(
            MarkerArray, 'carla/hero/vehicle_marker', 10
        )

        # ========================
        # VISUALIZATION PARAMETERS
        # ========================
        self.marker_array = MarkerArray()
        self.vehicle_width = 3   # meters
        self.vehicle_length = 3  # meters
        self.itr = 0  # Marker ID counter

    def marker_msg(self, pose):
        """
        Create a rectangular marker representing the vehicle at given pose.
        
        Args:
            pose: geometry_msgs/Point containing vehicle position
            
        Returns:
            Marker: LINE_STRIP marker forming a rectangle around vehicle
        """
        rect_marker = Marker()
        rect_marker.header.frame_id = 'map'
        rect_marker.header.stamp = self.get_clock().now().to_msg()
        rect_marker.ns = 'vehicle'+str(self.itr)
        rect_marker.id = self.itr
        rect_marker.type = Marker.LINE_STRIP
        rect_marker.action = Marker.ADD
        
        # Marker Appearance
        rect_marker.scale.x = 0.5 # Line width
        rect_marker.color.r = 0.0
        rect_marker.color.g = 1.0 # Green color
        rect_marker.color.b = 0.0
        rect_marker.color.a = 1.0 # Fully opaque

        # ========================
        # RECTANGLE CORNER CALCULATION
        # ========================
        # Define corners 
        tr = Point(x=(pose.x + self.vehicle_width/2), y=(pose.y + self.vehicle_length/2), z=0.1)
        tl = Point(x=(pose.x - self.vehicle_width/2), y=(pose.y + self.vehicle_length/2), z=0.1)
        br = Point(x=(pose.x + self.vehicle_width/2), y=(pose.y - self.vehicle_length/2), z=0.1)
        bl = Point(x=(pose.x - self.vehicle_width/2), y=(pose.y - self.vehicle_length/2), z=0.1)

        # Add points to create closed rectangle
        rect_marker.points.append(tr)
        rect_marker.points.append(tl)
        rect_marker.points.append(bl)
        rect_marker.points.append(br)
        rect_marker.points.append(tr) # Close the rectangle

        return rect_marker

    def odom_cb(self, msg):
        """
        Odometry callback that creates and publishes vehicle marker.
        
        Args:
            msg (Odometry): Vehicle odometry message containing pose and twist
        """
        # if self.odom_received:
        #     return
        
        # Extract position and motion data
        position = msg.pose.pose.position
        #orientation = msg.pose.pose.orientation
        #velocity = msg.twist.twist.linear

        self.odom_received = True

        # Create and add marker to array
        rect_marker = self.marker_msg(position)
        self.marker_array.markers.append(rect_marker)

        # Publish updated marker array
        self.vehicle_marker_pub.publish(self.marker_array)
        self.itr+=1
    
    def waypoints_cb(self, msg):
        """
        Waypoints callback for debugging path information.
        Currently logs waypoint information once per session.
        
        Args:
            msg (Path): Path message containing planned waypoints
        """
        if self.waypoints_received:
            return
        
        # ========================
        # WAYPOINT LOGGING (DEBUG)
        # ========================
        print("\n\n=----------------------------------------------------------------")
        for pose_stamped in msg.poses:
            position = pose_stamped.pose.position
            #self.get_logger().info(f'Waypoint Position: x={position.x}, y={position.y}, z={position.z}')

        start = msg.poses[0].pose.position
        end = msg.poses[1].pose.position

        # position = start.pose.position
        #print("\n\n=----------------------------------------------------------------")
        #self.get_logger().info(f'{GREEN}Waypoint Position: x={start.x}, y={start.y}, z={start.z}{RESET}')
        #self.get_logger().info(f'{GREEN}Waypoint Position: x={end.x}, y={end.y}, z={end.z}{RESET}')

        self.waypoints_received = True 

# ========================
# MAIN FUNCTION
# ========================

def main(args=None):
    """
    Main function to initialize and run the vehicle marker node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    carla_vehicle_marker = CarleVehicleMarker()
    rclpy.spin(carla_vehicle_marker)
    carla_vehicle_marker.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()