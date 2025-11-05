#pragma once

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/vector3.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

namespace lkas_aeb_perception{

    struct ROIParams{
        double x_min, x_max, y_min, y_max, z_min, z_max;
    };

    struct ClusterParams{
        double tolerance;
        int min_size;
        int max_size;
    };

    class LidarPreprocNode : public rclcpp::Node{
        public:
            explicit LidarPreprocNode(const rclcpp::NodeOptions & options);

        private:
            void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

            vision_msgs::msg::Detection3D createBBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                                                    const std_msgs::msg::Header& header);

            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;
            rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr publisher_;

            double voxel_leaf_size_;
            ROIParams roi_params_;
            ClusterParams cluster_params_;
    };
} // namespace lkas_aeb_perception