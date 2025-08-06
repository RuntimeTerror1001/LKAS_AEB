import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import lanelet2
from lanelet2.projection import UtmProjector

"""
LANELET2 MAP PUBLISHER NODE
"""

class Lanelet2Publisher(Node):
    """
    ROS2 node that publishes Lanelet2 map data as visualization markers.
    
    Loads a Lanelet2 OSM map file and publishes lane boundaries as LINE_STRIP markers
    for visualization in RViz. The map is published at 1Hz for real-time visualization.
    
    Publishes:
    - /lanelet2_map: MarkerArray containing lane boundary markers
    """
    
    def __init__(self):
        super().__init__('map_publisher')
        
        # ========================
        # ROS COMMUNICATION SETUP
        # ========================
        self.publisher_ = self.create_publisher(
            MarkerArray, '/lanelet2_map', 10
            )
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # ========================
        # MAP LOADING
        # ========================
        # Initialize UTM projector for coordinate transformation
        projector = UtmProjector(lanelet2.io.Origin(0, 0))
        
        # Load Lanelet2 map from OSM file
        map_path = "/home/redpaladin/Projects/lkas_aeb_ws/src/lkas_aeb/maps/Town10HD.osm"
        self.lanelet_map = lanelet2.io.load(map_path, projector)
        
        self.get_logger().info("Lanelet Map loaded successfully")

        
    def timer_callback(self):
        """
        Timer callback that publishes lane boundary markers.
        Called at 1Hz to provide continuous map visualization.
        
        Creates LINE_STRIP markers for each lanelet's left and right boundaries.
        """
        marker_array = MarkerArray()
        marker_id = 0  # Initialize marker ID
        # print("=----------------------------")
        # print("self.lanelet_map: ",dir(self.lanelet_map))
        # print("laneletLayer: ",type(self.lanelet_map.laneletLayer))
        
        # ========================
        # PROCESS EACH LANELET
        # ========================
        for lanelet_ in self.lanelet_map.laneletLayer:
            # Marker Configuration
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.3
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.ns = "lanelet"
            marker.id = marker_id  # Assign unique ID
            
            # Boundary Points Extraction
            for point in lanelet2.geometry.to2D(lanelet_.leftBound):
                marker.points.append(Point(x=point.x, y=point.y, z=0.0))

            for point in lanelet2.geometry.to2D(lanelet_.rightBound):
                marker.points.append(Point(x=point.x, y=point.y, z=0.0))

            marker_array.markers.append(marker)
            marker_id += 1

            
        self.publisher_.publish(marker_array)

# ========================
# MAIN FUNCTION
# ========================

def main(args=None):
    """
    Main function to initialize and run the map publisher node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = Lanelet2Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()