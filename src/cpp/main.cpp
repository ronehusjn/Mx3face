/*
╔═════════════════════ Main Execution Loop ═══════════════════════════════╗

   ┌─────────────────────── Input Callback ────────────────────────────┐
   │                                                                   │
   │   Retrieves frames                                                │
   │   Pre-processes frames                                            │
   │   Sends data to accelerator                                       │
   │                                                                   │
   └───────────────────────────────────────────────────────────────────┘
                                🔽
                                🔽
                                🔽
   ┌─────────────────────── Output Callback ───────────────────────────┐
   │                                                                   │
   │   Receives data from accelerator                                  │
   │   Applies post-processing                                         │
   │   Draws detection and renders output                              │
   │                                                                   │
   └───────────────────────────────────────────────────────────────────┘

╚═════════════════════════════════════════════════════════════════════════╝
*/
#include <stdio.h>
#include <signal.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

// Removed MemryX dependency - using CPU-only ONNX Runtime

#include "utils/gui_view.h"
#include "utils/input_source.h"
#include "utils/vms.h"
#include "utils/face_recognition.h"

constexpr int kMaxNumChannels = 100;
constexpr int kFpsCountMax = 120;
constexpr char kDefaultConfigPath[] = "assets/config.txt";

struct ChannelObject
{
    InputSource *input_source; // input capture device, could be usb-cam, ip-cam, or video
    DisplayScreen *screen;     // associated with a specific screen
    uint32_t disp_width;
    uint32_t disp_height;
    int frame_count;
    float fps_number;
    std::chrono::milliseconds start_ms;
    std::unique_ptr<FaceRecognition> face_recognition_handle;
    std::queue<cv::Mat *> disp_frames; // sync queue between processing threads
    std::mutex disp_frames_mutex;
    std::condition_variable disp_frames_cv;
};

// Global variables
ChannelObject g_chan_objs[kMaxNumChannels];
VmsCfg g_config;
vector<InputSource *> g_input_sources;
std::atomic<uint64_t> g_frame_count(0);
int g_duration_in_secs = 5;
bool g_is_running = true;
YOLOGuiView *g_gui = NULL;

// Forward declarations
float UpdatedFPS(int idx);

void signalHandler(int /*pSignal*/)
{
    g_is_running = false;
    if (g_gui)
        g_gui->Exit();
}

// CPU-only processing function for each channel
void ProcessChannel(int channel_idx)
{
    auto &chan_obj = g_chan_objs[channel_idx];
    auto &input_source = chan_obj.input_source;
    auto &screen = chan_obj.screen;
    auto face_recognition_handle = chan_obj.face_recognition_handle.get();

    while (g_is_running) {
        // Compute letterbox padding for face recognition model
        face_recognition_handle->ComputePadding(chan_obj.disp_width, chan_obj.disp_height);

        // Retrieve display frame buffer
        cv::Mat *disp_frame = screen->GetDisplayFrameBuf(channel_idx);

        // Fill frame from input source
        input_source->GetFrame(*disp_frame);

        // Run face recognition processing
        FaceRecognitionResult result;
        float confidence = (screen->GetConfidenceValue() == -1.0) ? g_config.inf_confidence : screen->GetConfidenceValue();
        face_recognition_handle->SetConfidenceThreshold(confidence);
        face_recognition_handle->ProcessImage(disp_frame->data, chan_obj.disp_width, chan_obj.disp_height, result);
        face_recognition_handle->DrawResult(result, *disp_frame);

        // FPS Calculation
        float fps_number = UpdatedFPS(channel_idx);

        // Set frame to display with FPS overlay
        screen->SetDisplayFrame(channel_idx, disp_frame, fps_number);

        // Sleep briefly to avoid overwhelming the CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // ~33 FPS
    }
}

float CalculateFPS(ChannelObject &channel)
{
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()) -
                    channel.start_ms;
    float frames_per_second = static_cast<float>(kFpsCountMax) * 1000.0f / duration.count();
    channel.fps_number = frames_per_second;
    channel.frame_count = 0;
    return frames_per_second;
}

float UpdatedFPS(int idx)
{
    auto &chan_obj = g_chan_objs[idx];
    ++chan_obj.frame_count;
    ++g_frame_count;

    if (chan_obj.frame_count == 1)
    {
        chan_obj.start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    }
    else if (chan_obj.frame_count == (kFpsCountMax + 1))
    {
        return CalculateFPS(chan_obj);
    }

    return 0.0f;
}

// Removed MemryX callback functions - using simple CPU processing threads

void Clean()
{
    for (long unsigned int i = 0; i < g_input_sources.size(); i++)
    {
        if (g_input_sources.at(i))
            delete g_input_sources.at(i);
    }
}

pair<long, long> GetCPUTimes()
{
    ifstream stat_file("/proc/stat");
    string line;
    getline(stat_file, line);
    stat_file.close();

    vector<long> times;
    string cpu;
    long value;
    istringstream iss(line);
    iss >> cpu; // Skip "cpu" field
    while (iss >> value)
    {
        times.push_back(value);
    }

    long idle_time = times[3]; // idle time (4th field)
    long total_time = accumulate(times.begin(), times.end(), 0L);

    return {idle_time, total_time};
}

double CalculateCPULoad(pair<long, long> prev, pair<long, long> current)
{
    long idle_diff = current.first - prev.first;
    long total_diff = current.second - prev.second;
    return 100.0 * (1.0 - static_cast<double>(idle_diff) / total_diff);
}

void InfoWatcher(int monitoring_duration_seconds)
{
    if (monitoring_duration_seconds <= 0)
    {
        std::cerr << "Error: monitoring_duration_seconds must be greater than 0" << std::endl;
        return;
    }

    uint64_t prev_frame_count = g_frame_count;
    pair<long, long> prev_times = GetCPUTimes();
    int idx_print = 0;
    int run_count = 0;
    unsigned int sleep_duration_ms = 100;
    int target_count = monitoring_duration_seconds * 1000 / sleep_duration_ms;
    while (g_is_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration_ms));
        run_count++;
        if (run_count == target_count)
        {
            {
                pair<long, long> current_times = GetCPUTimes();
                double cpu_load = CalculateCPULoad(prev_times, current_times);
                prev_times = current_times;

                // FPS
                uint64_t diff_count = g_frame_count - prev_frame_count;
                prev_frame_count = g_frame_count;
                printf("%d: FPS %.1f | CPU_load %.1f %%\n", idx_print++, (float)diff_count / monitoring_duration_seconds, cpu_load);
            }
            run_count = 0;
        }
    }
    return;
}

void InitScreen(YOLOGuiView &gui, int screen_idx)
{
    auto screen = gui.screens.at(screen_idx).get();
    screen->SetSquareLayout(g_config.num_chs);           // Specify num_chs as 4 for a 2x2 square layout.
    screen->SetConfidenceValue(g_config.inf_confidence); // Add this if the demo needs to modify confidence at runtime
    if (!g_config.model_name.empty())
        screen->SetModelName(g_config.model_name);
    g_gui = &gui;
}

void InitChannelObjects(YOLOGuiView &gui, int idx)
{
    // Initialize face recognition with standard input size
    int model_input_width = 640;   // YOLOv8n-face standard input
    int model_input_height = 640;
    int model_input_channel = 3;

    auto screen = gui.screens.at(g_config.screen_idx).get();
    g_chan_objs[idx].screen = screen;
    g_chan_objs[idx].disp_width = screen->GetViewerWidth(idx);
    g_chan_objs[idx].disp_height = screen->GetViewerHeight(idx);
    g_chan_objs[idx].frame_count = 0;
    g_chan_objs[idx].input_source = g_input_sources.at(idx);
    g_chan_objs[idx].face_recognition_handle = std::make_unique<FaceRecognition>(
        model_input_width, model_input_height, model_input_channel,
        g_config.inf_confidence);
}

// No buffer allocation needed for CPU-only processing

void ParseArgs(int argc, char *argv[])
{
    int opt;
    string config_path;
    while ((opt = getopt(argc, argv, "c:d:h")) != -1)
    {
        switch (opt)
        {
        case 'c':
            config_path = string(optarg);
            ReadVmsConfigFromFile(config_path.c_str(), g_config);
            break;
        case 'd':
            g_duration_in_secs = std::stoi(optarg);
            break;
        case 'h':
        default:
            printf("-c: config file for the demo,\t\t\tdefault: %s\n", kDefaultConfigPath);
            printf("-d: duration to measure FPS and CPU loading,\tdefault: 5 seconds\n");
            exit(1);
            break;
        }
    }
    if (config_path.empty())
    {
        ReadVmsConfigFromFile(kDefaultConfigPath, g_config);
    }
    return;
}

int main(int argc, char *argv[])
{
    // Handle SIGINT (Ctrl+C) signal
    signal(SIGINT, signalHandler);

    // Parse command-line arguments
    ParseArgs(argc, argv);

    // Initialize GUI and specify the screen for display
    YOLOGuiView gui(argc, argv);
    InitScreen(gui, g_config.screen_idx);
    auto screen = gui.screens.at(g_config.screen_idx).get();

    // Initialize input capture sources
    InitCaps(screen, g_config, g_input_sources);

    // Initialize channel objects with face recognition
    std::vector<std::thread> processing_threads;
    for (uint32_t channel_idx = 0; channel_idx < screen->NumViewers(); channel_idx++)
    {
        InitChannelObjects(gui, channel_idx);

        // Start a processing thread for each channel
        processing_threads.emplace_back(ProcessChannel, channel_idx);
    }

    printf("Started %d processing threads for face recognition\n", (int)processing_threads.size());

    // Start a separate thread to watch for runtime info
    std::thread info_watcher = std::thread(InfoWatcher, g_duration_in_secs);

    // Run GUI (blocks main thread until exit button is pressed)
    printf("press exit button (at top right) or ctrl-c to exit\n");

    screen->Show();
    gui.Run();

    // Mark the application as not running
    g_is_running = false;

    // Wait for all processing threads to complete
    for (auto& thread : processing_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    info_watcher.join();

    // Cleanup allocated resources
    Clean();

    printf("Exit successfully.\n");

    return 0;
}
