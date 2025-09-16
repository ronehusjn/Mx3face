#pragma once

#include <queue>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#define mxutil_prepost_sigmoid(_x_) (1.0 / (1.0 + expf(-1.0 * (_x_))))
#define mxutil_max(_x_, _y_) (((_x_) > (_y_)) ? (_x_) : (_y_))
#define mxutil_min(_x_, _y_) (((_x_) < (_y_)) ? (_x_) : (_y_))

// Face detection colors
static const std::vector<cv::Scalar> FACE_BOX_COLORS = {
    {0, 255, 0, 0.6},    // Green
    {255, 0, 0, 0.6},    // Blue
    {0, 0, 255, 0.6},    // Red
    {255, 255, 0, 0.6},  // Cyan
    {255, 0, 255, 0.6},  // Magenta
};

static const std::vector<cv::Scalar> FACE_TEXT_COLORS = {
    {255, 255, 255},     // White
    {0, 0, 0},           // Black
    {255, 255, 0},       // Yellow
    {0, 255, 255},       // Cyan
    {255, 0, 255},       // Magenta
};

// Face keypoint structure for facial landmarks
struct FaceKeypoint
{
    float x;
    float y;
    float confidence;

    FaceKeypoint() : x(-1), y(-1), confidence(-1) {}
    FaceKeypoint(float _x, float _y, float _conf) : x(_x), y(_y), confidence(_conf) {}
};

// Face detection bounding box with keypoints
struct FaceBox
{
    float confidence;      // Face detection confidence
    float x_min;          // Top-left x coordinate
    float y_min;          // Top-left y coordinate
    float x_max;          // Bottom-right x coordinate
    float y_max;          // Bottom-right y coordinate
    std::vector<FaceKeypoint> keypoints;  // 5 facial keypoints (eyes, nose, mouth corners)
    std::vector<float> embedding;         // 128-dimensional face embedding
    int identity_id;      // Assigned identity ID (-1 for unknown)
    std::string identity_name;  // Identity name

    // Default constructor
    FaceBox() : confidence(-1), x_min(-1), y_min(-1), x_max(-1), y_max(-1),
                identity_id(-1), identity_name("Unknown") {
        keypoints.resize(5);
        embedding.resize(128, 0.0f);
    }

    // Parameterized constructor
    FaceBox(float _conf, float _x_min, float _y_min, float _x_max, float _y_max)
        : confidence(_conf), x_min(_x_min), y_min(_y_min), x_max(_x_max), y_max(_y_max),
          identity_id(-1), identity_name("Unknown") {
        keypoints.resize(5);
        embedding.resize(128, 0.0f);
    }
};

// Face recognition result structure
struct FaceRecognitionResult
{
    std::vector<FaceBox> faces;
    int num_faces;

    FaceRecognitionResult() : num_faces(0) {}

    void clear() {
        faces.clear();
        num_faces = 0;
    }

    void add_face(const FaceBox& face) {
        faces.push_back(face);
        num_faces = faces.size();
    }
};

// Face database entry for storing known identities
struct FaceIdentity
{
    int id;
    std::string name;
    std::vector<std::vector<float>> embeddings;  // Multiple embeddings per identity

    FaceIdentity() : id(-1), name("Unknown") {}
    FaceIdentity(int _id, const std::string& _name) : id(_id), name(_name) {}

    void add_embedding(const std::vector<float>& embedding) {
        embeddings.push_back(embedding);
    }
};

// Simple face database for identity management
class FaceDatabase
{
private:
    std::vector<FaceIdentity> identities_;
    int next_id_;

public:
    FaceDatabase() : next_id_(0) {}

    int add_identity(const std::string& name) {
        FaceIdentity identity(next_id_, name);
        identities_.push_back(identity);
        return next_id_++;
    }

    void add_embedding_to_identity(int id, const std::vector<float>& embedding) {
        for (auto& identity : identities_) {
            if (identity.id == id) {
                identity.add_embedding(embedding);
                break;
            }
        }
    }

    int recognize_face(const std::vector<float>& embedding, float threshold = 0.6f) {
        float best_similarity = -1.0f;
        int best_id = -1;

        for (const auto& identity : identities_) {
            for (const auto& stored_embedding : identity.embeddings) {
                float similarity = cosine_similarity(embedding, stored_embedding);
                if (similarity > best_similarity && similarity > threshold) {
                    best_similarity = similarity;
                    best_id = identity.id;
                }
            }
        }

        return best_id;
    }

    std::string get_identity_name(int id) {
        for (const auto& identity : identities_) {
            if (identity.id == id) {
                return identity.name;
            }
        }
        return "Unknown";
    }

    size_t size() const { return identities_.size(); }

private:
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return -1.0f;

        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (size_t i = 0; i < a.size(); ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) return -1.0f;

        return dot_product / (sqrt(norm_a) * sqrt(norm_b));
    }
};