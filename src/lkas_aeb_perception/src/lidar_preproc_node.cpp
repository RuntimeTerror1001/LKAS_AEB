#include "lkas_aeb_perception/lidar_preproc_node.hpp"

namespace lkas_aeb_perception{
    LidarPreprocNode::LidarPreprocNode(const rclcpp::NodeOptions & options)
    : Node("lidar_preproc_node", options){
        // ========================
        // DECLARE PARAMS
        // ========================
        this->declare_parameter<double>("voxel_leaf_size", 0.25);

        // ROI Parameters
        this->declare_parameter<double>("roi.x_min", 0.0);
        this->declare_parameter<double>("roi.x_max", 40.0);
        this->declare_parameter<double>("roi.y_min", -6.0);
        this->declare_parameter<double>("roi.y_max", 6.0);
        this->declare_parameter<double>("roi.z_min", -2.5);
        this->declare_parameter<double>("roi.z_max", 2.5);

        // Cluster Parameters
        this->declare_parameter<double>("cluster.tolerance", 0.6);
        this->declare_parameter<int>("cluster.min_size", 5);
        this->declare_parameter<int>("cluster.max_size", 2000);

        // ========================
        // RETRIEVE PARAMS
        // ========================
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();

        roi_params_.x_min = this->get_parameter("roi.x_min").as_double();
        roi_params_.x_max = this->get_parameter("roi.x_max").as_double();
        roi_params_.y_min = this->get_parameter("roi.y_min").as_double();
        roi_params_.y_max = this->get_parameter("roi.y_max").as_double();
        roi_params_.z_min = this->get_parameter("roi.z_min").as_double();
        roi_params_.z_max = this->get_parameter("roi.z_max").as_double();

        cluster_params_.tolerance = this->get_parameter("cluster.tolerance").as_double();
        cluster_params_.min_size = this->get_parameter("cluster.min_size").as_int();
        cluster_params_.max_size = this->get_parameter("cluster.max_size").as_int();

        // ========================
        // ROS COMMUNICATION SETUP
        // ========================
        // Publishers
        publisher_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
            "/perception/lidar/obstacles_front", 10);

        // Subscribers
        subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/hero/lidar", 10, std::bind(&LidarPreprocNode::pointCloudCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "LidarPreprocNode has started.");
    }

    vision_msgs::msg::Detection3D LidarPreprocNode::createBBox(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
        const std_msgs::msg::Header& header){
            // Find cluster's min/max points (Axis Aligned Bounding Box)
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            vision_msgs::msg::Detection3D detection;
            detection.header = header;

            // Bounding Box
            detection.bbox.center.position.x = (max_pt.x + min_pt.x) / 2.0;
            detection.bbox.center.position.y = (max_pt.y + min_pt.y) / 2.0;
            detection.bbox.center.position.z = (max_pt.z + min_pt.z) / 2.0;
            detection.bbox.center.orientation.w = 1.0; 

            detection.bbox.size.x = max_pt.x - min_pt.x;
            detection.bbox.size.y = max_pt.y - min_pt.y;
            detection.bbox.size.z = max_pt.z - min_pt.z;

            // Create hypothesis
            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = "unknown"; // Generic Object
            hypothesis.hypothesis.score = 1.0;
            detection.results.push_back(hypothesis);

            return detection;
    }

    void LidarPreprocNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
        // 1. Convert to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // 2. ROI Crop
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(roi_params_.x_min, roi_params_.x_max);
        pass.filter(*cloud_cropped);
        pass.setInputCloud(cloud_cropped);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(roi_params_.y_min, roi_params_.y_max);
        pass.filter(*cloud_cropped);
        pass.setInputCloud(cloud_cropped);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(roi_params_.z_min, roi_params_.z_max);
        pass.filter(*cloud_cropped);

        // 3. Voxel Grid Downsampling
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_cropped);
        sor.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        sor.filter(*cloud_filtered);

        // 4. Euclidean Clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_params_.tolerance);
        ec.setMinClusterSize(cluster_params_.min_size);
        ec.setMaxClusterSize(cluster_params_.max_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        // 5. BBox Fitting & Publishing
        vision_msgs::msg::Detection3DArray detection_array;
        detection_array.header = msg->header;

        for(const auto& indices : cluster_indices){
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for(const auto& idx : indices.indices){
                cluster->push_back((*cloud_filtered)[idx]);
            }
            cluster->width = cluster->size();
            cluster->height = 1;
            cluster->is_dense = true;

            vision_msgs::msg::Detection3D detection = createBBox(cluster, msg->header);
            detection_array.detections.push_back(detection);
        }

        // Publish
        publisher_->publish(detection_array);
    }
} // namespace lkas_aeb_perception

// Register component
RCLCPP_COMPONENTS_REGISTER_NODE(lkas_aeb_perception::LidarPreprocNode)