
// g++ -std=c++11 -I sampler.cpp path_to_ggml/build/libggml.a -o sampler


// from sampling.py 
/*
    here is a piece of python code. impl the same logic in c++ need:
    "logit_bias" argument in sample_logits & sample_probs, can be regarded as
    NULL always. we dont need it. use common libs as you see fit. the "out"
    argument to sample_logits is a float* array in C
*/

#include "ggml.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
#include <cstring>

// Class to handle ggml context, graph building, and sampling
class GGMLSampler {
public:
    GGMLSampler(size_t size) : size_(size) {
        // Estimate buffer size needed for the computation graph
        // For softmax, operations are limited, but adjust based on actual usage
        size_t ctx_size = size_ * sizeof(float) * 4; // Adjust as needed

        // Initialize ggml context with the calculated buffer size
        params_.mem_size = ctx_size + 1024 * 1024; // Add extra buffer to be safe
        params_.mem_buffer = malloc(params_.mem_size);
        params_.no_alloc = false;

        if (params_.mem_buffer == NULL) {
            throw std::runtime_error("Failed to allocate memory for ggml context.");
        }

        ctx_ = ggml_init(params_);
        if (!ctx_) {
            free(params_.mem_buffer);
            throw std::runtime_error("ggml_init() failed.");
        }

        // Create input tensor
        input_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, size_);
        if (!input_) {
            ggml_free(ctx_);
            free(params_.mem_buffer);
            throw std::runtime_error("Failed to create input tensor.");
        }
        ggml_set_name(input_, "input");

        // Create softmax tensor
        softmax_tensor_ = ggml_soft_max(ctx_, input_);
        if (!softmax_tensor_) {
            ggml_free(ctx_);
            free(params_.mem_buffer);
            throw std::runtime_error("Failed to create softmax tensor.");
        }
        ggml_set_name(softmax_tensor_, "softmax");

        // Initialize the computation graph
        ggml_cgraph gf = {};
        ggml_build_forward_expand(&gf, softmax_tensor_);

        // Store the computation graph for later use
        graph_ = gf;
    }

    ~GGMLSampler() {
        // Free ggml context and buffer
        if (ctx_) {
            ggml_free(ctx_);
        }
        if (params_.mem_buffer) {
            free(params_.mem_buffer);
        }
    }

    // Function to sample from probabilities
    int sample_probs(const std::vector<float>& probs, float temperature = 1.0f, float top_p = 0.8f) const {
        // Copy probs to modify
        std::vector<float> adjusted_probs = probs;

        // Validate temperature and top_p
        if (temperature < 0.0f) {
            throw std::invalid_argument("temperature must be non-negative");
        }
        if (top_p < 0.0f || top_p > 1.0f) {
            throw std::invalid_argument("top_p must be between 0 and 1");
        }

        // If top_p is 0.0, set it to 1.0 to include all probabilities
        if (top_p == 0.0f) {
            top_p = 1.0f;
        }

        // If temperature is 0.0, return the index with the highest probability
        if (temperature == 0.0f) {
            return std::distance(adjusted_probs.begin(), std::max_element(adjusted_probs.begin(), adjusted_probs.end()));
        }

        // Apply top_p (nucleus) filtering
        if (top_p < 1.0f) {
            // Create a sorted copy of probabilities in descending order
            std::vector<float> sorted_probs = adjusted_probs;
            std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<float>());

            // Compute cumulative probabilities
            float cumulative = 0.0f;
            float cutoff = 0.0f;
            for (const auto& p : sorted_probs) {
                cumulative += p;
                if (cumulative > top_p) {
                    cutoff = p;
                    break;
                }
            }

            // Zero out probabilities below the cutoff
            for (auto& p : adjusted_probs) {
                if (p < cutoff) {
                    p = 0.0f;
                }
            }
        }

        // Apply temperature scaling
        if (temperature != 1.0f) {
            for (auto& p : adjusted_probs) {
                p = std::pow(p, 1.0f / temperature);
            }
        }

        // Normalize the probabilities
        float sum = 0.0f;
        for (const auto& p : adjusted_probs) {
            sum += p;
        }

        if (sum == 0.0f) {
            throw std::runtime_error("Sum of probabilities is zero after processing.");
        }

        for (auto& p : adjusted_probs) {
            p /= sum;
        }

        // Set up random number generation
        static std::random_device rd;
        static std::mt19937 gen(rd());

        // Create a discrete distribution based on the probabilities
        std::discrete_distribution<> dist(adjusted_probs.begin(), adjusted_probs.end());

        // Sample and return the index
        return dist(gen);
    }

    // Function to process logits and sample an index using the pre-built graph
    int sample_logits(const float* logits, float temperature = 1.0f, float top_p = 0.8f) {
        if (size_ == 0) {
            throw std::invalid_argument("Logits size must be greater than zero.");
        }

        // Update the input tensor with new logits
        memcpy(input_->data, logits, sizeof(float) * size_);

        // Execute the computation graph
        ggml_graph_compute_with_ctx(ctx_, &graph_, 1);

        // Retrieve the softmax result
        const float* probs_data = ggml_get_data_f32(softmax_tensor_);
        if (!probs_data) {
            throw std::runtime_error("Failed to retrieve softmax data.");
        }

        // Copy probabilities to a vector
        std::vector<float> probs(probs_data, probs_data + size_);

        // Sample an index based on the probabilities
        return sample_probs(probs, temperature, top_p);
    }

private:
    size_t size_;
    struct ggml_init_params params_;
    struct ggml_context* ctx_;
    struct ggml_tensor* input_;
    struct ggml_tensor* softmax_tensor_;
    ggml_cgraph graph_; // Computation graph
};

// Example usage
int main() {
    try {
        // Define the size of the logits vector
        const size_t logits_size = 3;

        // Initialize the sampler with the size
        GGMLSampler sampler(logits_size);

        // Example logits
        float logits_array1[] = {2.0f, 1.0f, 0.1f};
        float logits_array2[] = {1.5f, 2.5f, 0.3f};
        float logits_array3[] = {0.5f, 0.2f, 3.0f};

        // Sample indices for different logits
        int sampled_index1 = sampler.sample_logits(logits_array1, 1.0f, 0.8f);
        int sampled_index2 = sampler.sample_logits(logits_array2, 1.0f, 0.8f);
        int sampled_index3 = sampler.sample_logits(logits_array3, 1.0f, 0.8f);

        std::cout << "Sampled Index 1: " << sampled_index1 << std::endl;
        std::cout << "Sampled Index 2: " << sampled_index2 << std::endl;
        std::cout << "Sampled Index 3: " << sampled_index3 << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
