#pragma once

#include "face_core.h"
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <mutex>
#include <onnxruntime_cxx_api.h>

class FaceRecognition
{
public:
    /** @brief Constructor for face recognition system. */
    FaceRecognition(size_t input_width, size_t input_height, size_t input_channel,
                   float confidence_thresh);

    ~FaceRecognition();

    /**
     * @brief Process input image and run face detection/recognition.
     * @param rgb_data      Pointer to the input image data in RGB format.
     * @param image_width   Width of the display image.
     * @param image_height  Height of the display image.
     * @param result        Reference to the structure where the face recognition results will be stored.
     */
    void ProcessImage(uint8_t *rgb_data, int image_width, int image_height, FaceRecognitionResult &result);

    /** @brief Draw detected faces and identities on the provided image. */
    void DrawResult(FaceRecognitionResult &result, cv::Mat &image);

    /** @brief Clear the face recognition results. */
    void ClearDetectionResults(FaceRecognitionResult &result);

    /** @brief Setters and getters for the confidence threshold. */
    void SetConfidenceThreshold(float confidence);
    float GetConfidenceThreshold();

    /** @brief Compute padding values for letterboxing from the display image. */
    void ComputePadding(int disp_width, int disp_height);

    /** @brief Ensure the input dimensions are valid for horizontal display images only. */
    bool IsHorizontalInput(int disp_width, int disp_height);

    /** @brief Add a new identity to the face database. */
    int AddIdentity(const std::string& name);

    /** @brief Add an embedding to an existing identity. */
    void AddEmbeddingToIdentity(int id, const std::vector<float>& embedding);

private:
    /** @brief Structure representing per-layer information of face detection output. */
    struct LayerParams
    {
        uint8_t bbox_ofmap_flow_id;
        uint8_t confidence_ofmap_flow_id;
        uint8_t keypoint_ofmap_flow_id;
        size_t width;
        size_t height;
        size_t ratio;
        size_t bbox_fmap_size;
        size_t keypoint_fmap_size;
    };

    /** @brief Initialization method to set up face recognition model parameters. */
    void Init(size_t accl_input_width, size_t accl_input_height, size_t accl_input_channel,
              float confidence_thresh);

    /** @brief Helper methods for building face detections from model output. */
    void GetFaceDetection(std::queue<FaceBox> &face_boxes, int layer_id,
                         float *confidence_buffer, float *bbox_buffer, float *keypoint_buffer,
                         int row, int col);

    /** @brief Helper methods for calculating face bounding box parameters. */
    void CalculateFaceParams(const float *feature_values, int layer_id, int row, int col,
                            float &center_x, float &center_y, float &box_width, float &box_height);

    /** @brief Extract face region for recognition. */
    cv::Mat ExtractFace(const cv::Mat& image, const FaceBox& face_box,
                       const std::vector<FaceKeypoint>& keypoints);

    /** @brief Process face recognition embedding from cropped face. */
    void ProcessFaceEmbedding(const cv::Mat& face_image, std::vector<float>& embedding,
                             std::vector<float *> output_buffers);

    /** @brief Process ONNX model output for face detection. */
    void ProcessDetectionOutput(std::vector<Ort::Value>& output_tensors, FaceRecognitionResult &result);

    /** @brief Apply Non-Maximum Suppression to face detections. */
    void ApplyNMS(std::vector<FaceBox>& faces, float iou_threshold = 0.45f);

    /** @brief Calculate IoU between two face boxes. */
    float CalculateIoU(const FaceBox& box1, const FaceBox& box2);

    static constexpr size_t kNumPostProcessLayers = 3;
    struct LayerParams face_detection_layers_[kNumPostProcessLayers];

    // Model-specific parameters
    size_t accl_input_width_;   // Input width to accelerator
    size_t accl_input_height_;  // Input height to accelerator
    size_t accl_input_channel_; // Input channel to accelerator

    // Colors for face visualization
    std::vector<cv::Scalar> face_box_colors_;
    std::vector<cv::Scalar> face_text_colors_;

    // Confidence threshold
    std::mutex confidence_mutex_;
    float confidence_thresh_;

    // Letterbox ratio and padding
    float letterbox_ratio_;
    int letterbox_width_;
    int letterbox_height_;
    int padding_height_;
    int padding_width_;
    bool valid_input_;

    // Face database for identity management
    FaceDatabase face_database_;

    // ONNX Runtime session and environment
    Ort::Env ort_env_;
    Ort::Session ort_session_;
    Ort::MemoryInfo memory_info_;

    // Model input/output names and shapes
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
};