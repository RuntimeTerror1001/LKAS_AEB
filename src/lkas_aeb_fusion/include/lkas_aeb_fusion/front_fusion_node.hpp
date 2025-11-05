#pragma once

// Core
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

// Message Types
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "geometry_msgs/msg/point.h"
#include "lkas_aeb_msgs/msg/obstacle_array.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

// Message Filters & Sync.
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

// TF2 & Geometry
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

// Camera Projection
#include "image_geometry/pinhole_camera_model.h"
#include "opencv2/core/types.hpp"

namespace lkas_aeb_fusion {

    using ObstaclesMsg      = lkas_aeb_msgs::msg::ObstacleArray;
    using Detections3DMsg   = vision_msgs::msg::Detection3DArray;
    using CameraInfoMsg     = sensor_msgs::msg::CameraInfo;

    using ApproxTimeSyncPolicy = message_filters::sync_policies::ApproximateTime<
        ObstaclesMsg, Detections3DMsg, CameraInfoMsg>;

    class FrontFusionNode : public rclcpp::Node{
        public:
            explicit FrontFusionNode(const rclcpp::NodeOptions & options);
        
        private:
            void data_callback(
                ObstaclesMsg::ConstSharedPtr obstacles,
                Detections3DMsg::ConstSharedPtr detections3d,
                CameraInfoMsg::ConstSharedPtr cam_info
            );

            double calculate_iou(const cv::Rect& box1, const cv::Rect& box2);

            std::vector<geometry_msgs::msg::Point> get_bbox_corners(
                const vision_msgs::msg::BoundingBox3D& bbox
            );

            cv::Rect get_2d_bbox(
                const std::vector<geometry_msgs::msg::Point>& corners_3d_cam_frame,
                const image_geometry::PinholeCameraModel& cam_model
            );

            std::shared_ptr<message_filters::Subscriber<lkas_aeb_msgs::msg::ObstacleArray>> sub_cam_detections_;
            std::shared_ptr<message_filters::Subscriber<vision_msgs::msg::Detection3DArray>> sub_lidar_detections_;
            std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> sub_camera_info_;

            std::shared_ptr<message_filters::Synchronizer<ApproxTimeSyncPolicy>> synchronizer_;

            std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
            std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

            rclcpp::Publisher<lkas_aeb_msgs::msg::ObstacleArray>::SharedPtr pub_fused_obstacles_;
            rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_fused_detections_;

            double iou_thresh_;
            std::string base_link_frame_;
            std::string camera_optical_frame_;
    };

} //namespace lkas_aeb_fusion