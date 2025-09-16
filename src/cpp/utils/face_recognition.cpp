#include "face_recognition.h"
#include <algorithm>
#include <cmath>
#include <iostream>

FaceRecognition::FaceRecognition(size_t input_width, size_t input_height, size_t input_channel,
                               float confidence_thresh)
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "FaceRecognition"),
      ort_session_(nullptr),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Init(input_width, input_height, input_channel, confidence_thresh);
}

FaceRecognition::~FaceRecognition()
{
    // Cleanup if needed
}

void FaceRecognition::Init(size_t accl_input_width, size_t accl_input_height, size_t accl_input_channel,
                          float confidence_thresh)
{
    accl_input_width_ = accl_input_width;
    accl_input_height_ = accl_input_height;
    accl_input_channel_ = accl_input_channel;
    confidence_thresh_ = confidence_thresh;

    // Load ONNX model
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        std::string model_path = "models/yolov8n-face_post.onnx";
        ort_session_ = Ort::Session(ort_env_, model_path.c_str(), session_options);

        // Get input/output names and shapes
        size_t num_input_nodes = ort_session_.GetInputCount();
        size_t num_output_nodes = ort_session_.GetOutputCount();

        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = ort_session_.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            input_names_.push_back(input_name.get());
            auto input_type_info = ort_session_.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            input_shapes_.push_back(input_dims);
        }

        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = ort_session_.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            output_names_.push_back(output_name.get());
            auto output_type_info = ort_session_.GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_dims = output_tensor_info.GetShape();
            output_shapes_.push_back(output_dims);
        }

        std::cout << "Loaded ONNX model: " << model_path << std::endl;
        std::cout << "Model has " << num_input_nodes << " inputs and " << num_output_nodes << " outputs" << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        throw;
    }

    // Initialize layer parameters for YOLOv8n-Face (similar to YOLOv8 but with keypoints)
    // Layer 0: 80x80 feature map
    face_detection_layers_[0].bbox_ofmap_flow_id = 0;
    face_detection_layers_[0].confidence_ofmap_flow_id = 0;
    face_detection_layers_[0].keypoint_ofmap_flow_id = 0;
    face_detection_layers_[0].width = 80;
    face_detection_layers_[0].height = 80;
    face_detection_layers_[0].ratio = 8;  // 640/80 = 8
    face_detection_layers_[0].bbox_fmap_size = 80 * 80 * 4;      // 4 bbox coordinates
    face_detection_layers_[0].keypoint_fmap_size = 80 * 80 * 10; // 5 keypoints * 2 coords

    // Layer 1: 40x40 feature map
    face_detection_layers_[1].bbox_ofmap_flow_id = 0;
    face_detection_layers_[1].confidence_ofmap_flow_id = 0;
    face_detection_layers_[1].keypoint_ofmap_flow_id = 0;
    face_detection_layers_[1].width = 40;
    face_detection_layers_[1].height = 40;
    face_detection_layers_[1].ratio = 16;  // 640/40 = 16
    face_detection_layers_[1].bbox_fmap_size = 40 * 40 * 4;
    face_detection_layers_[1].keypoint_fmap_size = 40 * 40 * 10;

    // Layer 2: 20x20 feature map
    face_detection_layers_[2].bbox_ofmap_flow_id = 0;
    face_detection_layers_[2].confidence_ofmap_flow_id = 0;
    face_detection_layers_[2].keypoint_ofmap_flow_id = 0;
    face_detection_layers_[2].width = 20;
    face_detection_layers_[2].height = 20;
    face_detection_layers_[2].ratio = 32;  // 640/20 = 32
    face_detection_layers_[2].bbox_fmap_size = 20 * 20 * 4;
    face_detection_layers_[2].keypoint_fmap_size = 20 * 20 * 10;

    // Initialize colors
    face_box_colors_ = FACE_BOX_COLORS;
    face_text_colors_ = FACE_TEXT_COLORS;

    // Initialize padding
    letterbox_ratio_ = 1.0f;
    letterbox_width_ = accl_input_width_;
    letterbox_height_ = accl_input_height_;
    padding_height_ = 0;
    padding_width_ = 0;
    valid_input_ = true;
}

void FaceRecognition::ProcessImage(uint8_t *rgb_data, int image_width, int image_height,
                                   FaceRecognitionResult &result)
{
    result.clear();

    // Create OpenCV Mat from RGB data
    cv::Mat input_image(image_height, image_width, CV_8UC3, rgb_data);

    // Convert RGB to BGR for OpenCV
    cv::Mat bgr_image;
    cv::cvtColor(input_image, bgr_image, cv::COLOR_RGB2BGR);

    // Resize with letterboxing to maintain aspect ratio
    cv::Mat resized_image;
    cv::resize(bgr_image, resized_image, cv::Size(letterbox_width_, letterbox_height_));

    // Add padding if needed
    cv::Mat padded_image = resized_image;
    if (padding_height_ > 0 || padding_width_ > 0) {
        cv::copyMakeBorder(resized_image, padded_image,
                          padding_height_ / 2, padding_height_ - padding_height_ / 2,
                          padding_width_ / 2, padding_width_ - padding_width_ / 2,
                          cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    }

    // Normalize to [0, 1] and convert to float
    cv::Mat normalized_image;
    padded_image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);

    // Convert from HWC to CHW format for model input
    std::vector<cv::Mat> bgr_channels;
    cv::split(normalized_image, bgr_channels);

    // Prepare input tensor
    size_t input_tensor_size = accl_input_width_ * accl_input_height_ * 3;
    std::vector<float> input_tensor_values(input_tensor_size);

    // Copy BGR channels in CHW format
    int img_size = accl_input_width_ * accl_input_height_;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_tensor_values.data() + c * img_size, bgr_channels[c].data, img_size * sizeof(float));
    }

    // Create input tensor
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values.data(), input_tensor_size,
        input_shapes_[0].data(), input_shapes_[0].size()));

    // Run inference
    try {
        auto output_tensors = ort_session_.Run(Ort::RunOptions{nullptr},
                                              input_names_.data(), input_tensors.data(), 1,
                                              output_names_.data(), output_names_.size());

        // Process outputs for face detection
        ProcessDetectionOutput(output_tensors, result);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
    }
}

void FaceRecognition::ProcessDetectionOutput(std::vector<Ort::Value>& output_tensors, FaceRecognitionResult &result)
{
    result.clear();
    std::vector<FaceBox> faces_vector;

    // Process the ONNX model output
    // Assuming YOLOv8n-face output format: [batch, 5 + num_keypoints*2 + num_classes, num_detections]
    if (output_tensors.empty()) {
        return;
    }

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // YOLOv8 output format: [batch, 4+1+10, 8400] where:
    // 4: bbox coords (cx, cy, w, h)
    // 1: confidence
    // 10: 5 keypoints (x, y for each)
    int num_detections = output_shape[2];
    int num_features = output_shape[1];  // Should be 15 (4+1+10)

    for (int i = 0; i < num_detections; ++i) {
        // Extract confidence
        float confidence = output_data[4 * num_detections + i];

        if (confidence < confidence_thresh_) {
            continue;
        }

        // Extract bbox coordinates (center format)
        float cx = output_data[0 * num_detections + i];
        float cy = output_data[1 * num_detections + i];
        float w = output_data[2 * num_detections + i];
        float h = output_data[3 * num_detections + i];

        // Convert to corner format
        FaceBox face_box;
        face_box.confidence = confidence;
        face_box.x_min = cx - w / 2.0f;
        face_box.y_min = cy - h / 2.0f;
        face_box.x_max = cx + w / 2.0f;
        face_box.y_max = cy + h / 2.0f;

        // Extract keypoints (5 points, 2 coords each)
        for (int kp = 0; kp < 5; ++kp) {
            float kp_x = output_data[(5 + kp * 2) * num_detections + i];
            float kp_y = output_data[(5 + kp * 2 + 1) * num_detections + i];
            face_box.keypoints[kp] = FaceKeypoint(kp_x, kp_y, 1.0f);
        }

        faces_vector.push_back(face_box);
    }

    // Apply Non-Maximum Suppression
    ApplyNMS(faces_vector);

    // Process face recognition for each detected face
    for (auto& face : faces_vector) {
        // For now, assign unknown identity
        // In a full implementation, you would extract the face region,
        // run it through the FaceNet model, and match against the database
        face.identity_id = -1;
        face.identity_name = "Unknown";

        result.add_face(face);
    }
}

void FaceRecognition::GetFaceDetection(std::queue<FaceBox> &face_boxes, int layer_id,
                                      float *confidence_buffer, float *bbox_buffer,
                                      float *keypoint_buffer, int row, int col)
{
    const auto& layer = face_detection_layers_[layer_id];
    int grid_idx = row * layer.width + col;

    // Get confidence score
    float confidence = mxutil_prepost_sigmoid(confidence_buffer[grid_idx]);

    if (confidence < confidence_thresh_) {
        return;
    }

    // Create face box
    FaceBox face_box;
    face_box.confidence = confidence;

    // Calculate bounding box coordinates
    float center_x, center_y, box_width, box_height;
    CalculateFaceParams(&bbox_buffer[grid_idx * 4], layer_id, row, col,
                       center_x, center_y, box_width, box_height);

    // Convert to corner coordinates
    face_box.x_min = center_x - box_width / 2.0f;
    face_box.y_min = center_y - box_height / 2.0f;
    face_box.x_max = center_x + box_width / 2.0f;
    face_box.y_max = center_y + box_height / 2.0f;

    // Process keypoints (5 facial landmarks)
    for (int kp = 0; kp < 5; ++kp) {
        int kp_idx = grid_idx * 10 + kp * 2;
        float kp_x = (col + mxutil_prepost_sigmoid(keypoint_buffer[kp_idx])) * layer.ratio;
        float kp_y = (row + mxutil_prepost_sigmoid(keypoint_buffer[kp_idx + 1])) * layer.ratio;
        face_box.keypoints[kp] = FaceKeypoint(kp_x, kp_y, 1.0f);
    }

    face_boxes.push(face_box);
}

void FaceRecognition::CalculateFaceParams(const float *feature_values, int layer_id, int row, int col,
                                         float &center_x, float &center_y, float &box_width, float &box_height)
{
    const auto& layer = face_detection_layers_[layer_id];

    // YOLOv8 bbox format: center_x, center_y, width, height
    center_x = (col + mxutil_prepost_sigmoid(feature_values[0])) * layer.ratio;
    center_y = (row + mxutil_prepost_sigmoid(feature_values[1])) * layer.ratio;
    box_width = exp(feature_values[2]) * layer.ratio;
    box_height = exp(feature_values[3]) * layer.ratio;
}

void FaceRecognition::ApplyNMS(std::vector<FaceBox>& faces, float iou_threshold)
{
    // Sort by confidence score (descending)
    std::sort(faces.begin(), faces.end(),
              [](const FaceBox& a, const FaceBox& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(faces.size(), false);

    for (size_t i = 0; i < faces.size(); ++i) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < faces.size(); ++j) {
            if (suppressed[j]) continue;

            float iou = CalculateIoU(faces[i], faces[j]);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    // Remove suppressed faces
    auto it = std::remove_if(faces.begin(), faces.end(),
                            [&](const FaceBox& face) {
                                size_t idx = &face - &faces[0];
                                return suppressed[idx];
                            });
    faces.erase(it, faces.end());
}

float FaceRecognition::CalculateIoU(const FaceBox& box1, const FaceBox& box2)
{
    float x1 = mxutil_max(box1.x_min, box2.x_min);
    float y1 = mxutil_max(box1.y_min, box2.y_min);
    float x2 = mxutil_min(box1.x_max, box2.x_max);
    float y2 = mxutil_min(box1.y_max, box2.y_max);

    if (x2 <= x1 || y2 <= y1) return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min);
    float area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min);
    float union_area = area1 + area2 - intersection;

    return union_area > 0 ? intersection / union_area : 0.0f;
}

void FaceRecognition::DrawResult(FaceRecognitionResult &result, cv::Mat &image)
{
    for (const auto& face : result.faces) {
        // Choose color based on identity
        int color_idx = (face.identity_id >= 0) ? (face.identity_id % face_box_colors_.size()) : 0;
        cv::Scalar box_color = face_box_colors_[color_idx];
        cv::Scalar text_color = face_text_colors_[color_idx];

        // Draw bounding box
        cv::Point top_left(static_cast<int>(face.x_min), static_cast<int>(face.y_min));
        cv::Point bottom_right(static_cast<int>(face.x_max), static_cast<int>(face.y_max));
        cv::rectangle(image, top_left, bottom_right, box_color, 2);

        // Draw keypoints
        for (const auto& keypoint : face.keypoints) {
            if (keypoint.confidence > 0.5f) {
                cv::circle(image, cv::Point(static_cast<int>(keypoint.x), static_cast<int>(keypoint.y)),
                          3, box_color, -1);
            }
        }

        // Draw identity label
        std::string label = face.identity_name + " (" + std::to_string(static_cast<int>(face.confidence * 100)) + "%)";

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::Point label_pos(top_left.x, top_left.y - 5);
        cv::rectangle(image,
                     cv::Point(label_pos.x, label_pos.y - text_size.height - baseline),
                     cv::Point(label_pos.x + text_size.width, label_pos.y),
                     box_color, -1);

        cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
    }
}

void FaceRecognition::ClearDetectionResults(FaceRecognitionResult &result)
{
    result.clear();
}

void FaceRecognition::SetConfidenceThreshold(float confidence)
{
    std::lock_guard<std::mutex> lock(confidence_mutex_);
    confidence_thresh_ = confidence;
}

float FaceRecognition::GetConfidenceThreshold()
{
    std::lock_guard<std::mutex> lock(confidence_mutex_);
    return confidence_thresh_;
}

void FaceRecognition::ComputePadding(int disp_width, int disp_height)
{
    if (!IsHorizontalInput(disp_width, disp_height)) {
        valid_input_ = false;
        return;
    }

    valid_input_ = true;

    // Calculate letterbox parameters
    float scale = std::min(static_cast<float>(accl_input_width_) / disp_width,
                          static_cast<float>(accl_input_height_) / disp_height);

    letterbox_width_ = static_cast<int>(disp_width * scale);
    letterbox_height_ = static_cast<int>(disp_height * scale);

    padding_width_ = accl_input_width_ - letterbox_width_;
    padding_height_ = accl_input_height_ - letterbox_height_;

    letterbox_ratio_ = scale;
}

bool FaceRecognition::IsHorizontalInput(int disp_width, int disp_height)
{
    return disp_width >= disp_height;
}

int FaceRecognition::AddIdentity(const std::string& name)
{
    return face_database_.add_identity(name);
}

void FaceRecognition::AddEmbeddingToIdentity(int id, const std::vector<float>& embedding)
{
    face_database_.add_embedding_to_identity(id, embedding);
}