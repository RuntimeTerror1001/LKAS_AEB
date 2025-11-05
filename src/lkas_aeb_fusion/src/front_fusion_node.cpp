#include "lkas_aeb_fusion/front_fusion_node.hpp"
#include "lkas_aeb_msgs/msg/obstacle.hpp"
#include <tf2_ros/create_timer_ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> // For getMinMax3D
#include <tf2_eigen/tf2_eigen.hpp>
#include <string> // For std::to_string
#include <cmath> // For std::hypot

namespace lkas_aeb_fusion{
    FrontFusionNode::FrontFusionNode(const rclcpp::NodeOptions & options)
    : Node("front_fusion_node", options){
        // ========================
        // DECLARE PARAMS
        // ========================
        this->declare_parameter<double>("iou_threshold", 0.3);
        this->declare_parameter<std::string>("base_link_frame", "hero");
        this->declare_parameter<std::string>("camera_optical_frame", "hero/rgb_front");

        // ========================
        // RETRIEVE PARAMS
        // ========================
        iou_thresh_ = this->get_parameter("iou_threshold").as_double();
        base_link_frame_ = this->get_parameter("base_link_frame").as_string();
        camera_optical_frame_ = this->get_parameter("camera_optical_frame").as_string();

        // ========================
        // TF2 SETUP
        // ========================
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
            this->get_node_base_interface(),
            this->get_node_timers_interface());
        tf_buffer_->setCreateTimerInterface(timer_interface);
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ========================
        // ROS COMMUNICATION SETUP
        // ========================
        // Publishers
        pub_fused_obstacles_ = this->create_publisher<lkas_aeb_msgs::msg::ObstacleArray>(
            "/perception/fused_obstacles_front", 10);
        pub_fused_detections_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
            "/perception/fused_detections_front", 10);

        // Subscribers
        sub_cam_detections_ = std::make_shared<message_filters::Subscriber<ObstaclesMsg>>(
            this, "/perception/obstacles_info");
        sub_lidar_detections_ = std::make_shared<message_filters::Subscriber<Detections3DMsg>>(
            this, "/perception/lidar/obstacles_front");
        sub_camera_info_ = std::make_shared<message_filters::Subscriber<CameraInfoMsg>>(
            this, "/carla/hero/rgb_front/camera_info");

        // Synchronizer
        synchronizer_ = std::make_shared<message_filters::Synchronizer<ApproxTimeSyncPolicy>>(
            ApproxTimeSyncPolicy(10), *sub_cam_detections_, *sub_lidar_detections_, *sub_camera_info_);

        synchronizer_->registerCallback(
            std::bind(&FrontFusionNode::data_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)
        );

        RCLCPP_INFO(this->get_logger(), "FrontFusionNode has started.");
    }

    std::vector<geometry_msgs::msg::Point> FrontFusionNode::get_bbox_corners(
        const vision_msgs::msg::BoundingBox3D& bbox){
            std::vector<geometry_msgs::msg::Point> corners(8);
            const auto& pos = bbox.center.position;
            const auto& size = bbox.size;

            // Assume BBox is axis-aligned in own frame.
            // Apply pose (orientation) of the center to the corners
            tf2::Quaternion q(
                bbox.center.orientation.x,
                bbox.center.orientation.y,
                bbox.center.orientation.z,
                bbox.center.orientation.w
            );
            q.normalize();
            tf2::Matrix3x3 rot_matrix(q);

            // Create 8 corners in the box's local frame
            std::vector<tf2::Vector3> local_corners;
            local_corners.push_back(tf2::Vector3(-size.x / 2, -size.y / 2, -size.z / 2));
            local_corners.push_back(tf2::Vector3(-size.x / 2, -size.y / 2, +size.z / 2));
            local_corners.push_back(tf2::Vector3(-size.x / 2, +size.y / 2, -size.z / 2));
            local_corners.push_back(tf2::Vector3(-size.x / 2, +size.y / 2, +size.z / 2));
            local_corners.push_back(tf2::Vector3(+size.x / 2, -size.y / 2, -size.z / 2));
            local_corners.push_back(tf2::Vector3(+size.x / 2, -size.y / 2, +size.z / 2));
            local_corners.push_back(tf2::Vector3(+size.x / 2, +size.y / 2, -size.z / 2));
            local_corners.push_back(tf2::Vector3(+size.x / 2, +size.y / 2, +size.z / 2));

            // Rotate & Translate corners to the frame defined by the pose
            for(int i = 0; i < 8; ++i){
                tf2::Vector3 rotated_corner = rot_matrix * local_corners[i];
                corners[i].x = rotated_corner.x() + pos.x;
                corners[i].y = rotated_corner.y() + pos.y;
                corners[i].z = rotated_corner.z() + pos.z;
            }
            
            return corners;
    }

    cv::Rect FrontFusionNode::get_2d_bbox(
        const std::vector<geometry_msgs::msg::Point>& corners_3d_cam_frame,
        const image_geometry::PinholeCameraModel& cam_model){
            
            std::vector<cv::Point2d> corners2d;
            for(const auto& corner_msg : corners_3d_cam_frame){
                // Convert from ROS msg::Point to cv::Point3d
                cv::Point3d corner_cv(corner_msg.x, corner_msg.y, corner_msg.z);

                // Project points only in front of the camera
                if(corner_cv.z > 0){
                    corners2d.push_back(cam_model.project3dToPixel(corner_cv));
                }
            }

            if(corners2d.empty()) return cv::Rect(0, 0, 0, 0); // All corners were behind the camera

            // Find min/max x/y to create 2D bounding box
            auto min_x = corners2d[0].x;
            auto max_x = corners2d[0].x;
            auto min_y = corners2d[0].y;
            auto max_y = corners2d[0].y;

            for(const auto& corner : corners2d){
                min_x = std::min(min_x, corner.x);
                max_x = std::max(max_x, corner.x);
                min_y = std::min(min_y, corner.y);
                max_y = std::max(max_y, corner.y);
            }

            // Clamp to image bounds
            min_x = std::max(0.0, min_x);
            min_y = std::max(0.0, min_y);
            max_x = std::min((double)cam_model.cameraInfo().width - 1, max_x);
            max_y = std::min((double)cam_model.cameraInfo().height - 1, max_y);

            if(max_x < min_x || max_y < min_y) return cv::Rect(0, 0, 0, 0);

            return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);  // [x,y,w,h]  
    }

    double FrontFusionNode::calculate_iou(const cv::Rect& box1, const cv::Rect& box2){
        cv::Rect intersection = box1 & box2;
        double intersection_area = intersection.area();
        double union_area = box1.area() + box2.area() - intersection_area;

        if(union_area == 0) return 0.0;

        return intersection_area / union_area;
    }

    void FrontFusionNode::data_callback(
        ObstaclesMsg::ConstSharedPtr cam_msg,
        Detections3DMsg::ConstSharedPtr lidar_msg,
        CameraInfoMsg::ConstSharedPtr info_msg){
            
            // Initialize camera model
            image_geometry::PinholeCameraModel cam_model;
            cam_model.fromCameraInfo(info_msg);

            // Output Messages
            // 1. Control Node
            lkas_aeb_msgs::msg::ObstacleArray fused_obstacles;
            fused_obstacles.header.stamp = this->get_clock()->now();
            fused_obstacles.header.frame_id = base_link_frame_;

            // 2. RViz2
            vision_msgs::msg::Detection3DArray fused_detections;
            fused_detections.header = fused_obstacles.header;

            // Get transforms
            geometry_msgs::msg::TransformStamped tf_lidar_cam;
            geometry_msgs::msg::TransformStamped tf_lidar_base;
            try {
                // Transform for PROJECTION in camera_optical_frame
                tf_lidar_cam = tf_buffer_->lookupTransform(
                    camera_optical_frame_, lidar_msg->header.frame_id,
                    lidar_msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
                
                // Traansformm for FUSED_OBJECT in base_link
                tf_lidar_base = tf_buffer_->lookupTransform(
                    base_link_frame_, lidar_msg->header.frame_id,
                    lidar_msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
            }
            catch (const tf2::TransformException & ex){
                RCLCPP_WARN(this->get_logger(), "Could not transforms : %s", ex.what());
                return;
            }

            // Loop through all 3D LiDAR Detections
            for(const auto& lidar_det : lidar_msg->detections){
                // ====================
                // FUSION TRANSFORM
                // ====================
                // Get 8 corners in original LiDAR Frame
                std::vector<geometry_msgs::msg::Point> original_corners = get_bbox_corners(lidar_det.bbox);
                
                // Transform all 8 corners to base_link
                pcl::PointCloud<pcl::PointXYZ> transformed_corners_base;
                for(const auto& corner : original_corners){
                    geometry_msgs::msg::Point transformed_pt;
                    tf2::doTransform(corner, transformed_pt, tf_lidar_base);
                    transformed_corners_base.push_back(pcl::PointXYZ(transformed_pt.x, transformed_pt.y, transformed_pt.z));
                }

                // Create new axis-aligned BBox in base_link
                pcl::PointXYZ min_pt, max_pt;
                pcl::getMinMax3D(transformed_corners_base, min_pt, max_pt);

                // 3D BBox for RViz
                vision_msgs::msg::BoundingBox3D base_link_bbox;
                base_link_bbox.center.position.x = (max_pt.x + min_pt.x) / 2.0;
                base_link_bbox.center.position.y = (max_pt.y + min_pt.y) / 2.0;
                base_link_bbox.center.position.z = (max_pt.z + min_pt.z) / 2.0;
                base_link_bbox.center.orientation.w = 1.0; // Is Axis-Aligned
                base_link_bbox.size.x = max_pt.x - min_pt.x;
                base_link_bbox.size.y = max_pt.y - min_pt.y;
                base_link_bbox.size.z = max_pt.z - min_pt.z;

                // ====================
                // PROJECTION TRANSFORM
                // ====================
                std::vector<geometry_msgs::msg::Point> cam_frame_corners(8);
                for(size_t i = 0; i < original_corners.size(); ++i){
                    tf2::doTransform(original_corners[i], cam_frame_corners[i], tf_lidar_cam);
                }

                // Project 3D Box (now in camera frame) to 2D
                cv::Rect projected_bbox = get_2d_bbox(cam_frame_corners, cam_model);
                if(projected_bbox.area() == 0) continue;

                // ====================
                // ASSOCIATION
                // ====================
                double best_iou = 0.0;
                int best_match_idx = -1;

                for(size_t i = 0; i < cam_msg->obstacles.size(); ++i){
                    const auto& cam_obstacle = cam_msg->obstacles[i];

                    int x1 = cam_obstacle.bbox[0];
                    int y1 = cam_obstacle.bbox[1];
                    int x2 = cam_obstacle.bbox[2];
                    int y2 = cam_obstacle.bbox[3];
                    int width = x2 - x1;
                    int height = y2 - y1;
                    cv::Rect cam_bbox(x1, y1, width, height);

                    double iou = calculate_iou(projected_bbox, cam_bbox);

                    if(iou > best_iou){
                        best_iou = iou;
                        best_match_idx = i;
                    }
                }

                // ====================
                // FUSION
                // ====================
                lkas_aeb_msgs::msg::Obstacle fused_obstacle; // For Control Node
                
                vision_msgs::msg::Detection3D fused_detection; // For RViz2
                fused_detection.header = fused_detections.header;
                fused_detection.bbox = base_link_bbox;

                vision_msgs::msg::ObjectHypothesisWithPose hypothesis;

                // Data from Camera (Classification)
                if(best_iou > iou_thresh_ && best_match_idx >= 0){
                    // Match found - use camera data
                    fused_obstacle.class_id = cam_msg->obstacles[best_match_idx].class_id;
                    fused_obstacle.track_id = cam_msg->obstacles[best_match_idx].track_id;
                    fused_obstacle.speed = cam_msg->obstacles[best_match_idx].speed;
                    fused_obstacle.relative_speed = cam_msg->obstacles[best_match_idx].relative_speed;
                    fused_obstacle.distance = cam_msg->obstacles[best_match_idx].distance;
                    
                    // Use camera's bbox
                    fused_obstacle.bbox = cam_msg->obstacles[best_match_idx].bbox;

                    hypothesis.hypothesis.class_id = std::to_string(fused_obstacle.class_id);
                    hypothesis.hypothesis.score = 1.0;
                }
                else{
                    // No match - LiDAR only
                    fused_obstacle.class_id = 0;  // Unknown
                    fused_obstacle.track_id = 0;  // No tracking
                    fused_obstacle.speed = 0.0;
                    fused_obstacle.relative_speed = 0.0;
                    fused_obstacle.distance = base_link_bbox.center.position.x;
                    
                    // Use projected bbox
                    fused_obstacle.bbox[0] = projected_bbox.x;
                    fused_obstacle.bbox[1] = projected_bbox.y;
                    fused_obstacle.bbox[2] = projected_bbox.x + projected_bbox.width;
                    fused_obstacle.bbox[3] = projected_bbox.y + projected_bbox.height;

                    hypothesis.hypothesis.class_id = "unknown";
                    hypothesis.hypothesis.score = 0.5;
                }

                fused_obstacles.obstacles.push_back(fused_obstacle);
                
                fused_detection.results.clear();
                fused_detection.results.push_back(hypothesis);
                fused_detections.detections.push_back(fused_detection);
            }
            
            pub_fused_obstacles_->publish(fused_obstacles);
            pub_fused_detections_->publish(fused_detections);
        }
} // namespace lkas_aeb_fusion

RCLCPP_COMPONENTS_REGISTER_NODE(lkas_aeb_fusion::FrontFusionNode)