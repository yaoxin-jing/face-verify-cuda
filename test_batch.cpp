#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "common.h"


typedef unsigned char uchar;


// Macro for CUDA driver error checking
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        CUresult err = call;                                            \
        if (err != CUDA_SUCCESS) {                                      \
            const char *errStr;                                         \
            cuGetErrorString(err, &errStr);                             \
            std::cerr << "CUDA Error: " << errStr << std::endl;        \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// Image parameters
#define IMG_SIZE (IMG_DIMENSION * IMG_DIMENSION)
const uint64_t batch_size = 8;


std::vector<uchar> load_dat_image(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open: " + path);

    std::vector<uchar> buffer(IMG_SIZE);
    file.read(reinterpret_cast<char*>(buffer.data()), IMG_SIZE);
    if (file.gcount() != IMG_SIZE)
        throw std::runtime_error("Incorrect image size in: " + path);
    return buffer;
}

std::vector<std::string> get_all_valid_dat_images(const std::string& folder) {
    std::vector<std::string> paths;
    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        perror("opendir failed");
        return paths;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".dat") {
            std::string full_path = folder + "/" + name;
            struct stat st;
            if (stat(full_path.c_str(), &st) == 0 && st.st_size == IMG_SIZE) {
                paths.push_back(full_path);
            }
        }
    }

    closedir(dir);
    return paths;
}

std::vector<std::string> select_random_images(const std::vector<std::string>& all, size_t count) {
    if (count > all.size()) throw std::runtime_error("Not enough images to sample from.");

    std::vector<std::string> shuffled = all;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled.begin(), shuffled.end(), g);

    return std::vector<std::string>(shuffled.begin(), shuffled.begin() + count);
}


// Generate `batch_size` fake grayscale images
std::vector<uchar> generate_reference_images(int batch_size) {
    std::vector<uchar> images(batch_size * IMG_SIZE);

    for (int i = 0; i < batch_size * IMG_SIZE; ++i) {
        images[i] = rand() % 256;  // Random grayscale pixel
    }

    return images;
}

void init_gpu_face_verfication(CUmodule module, uchar* images, uint64_t batch_size) {
    CUfunction hist_cal;
    CHECK_CUDA(cuModuleGetFunction(&hist_cal, module, "hist_cal"));  // Load from PTX

    // Step 1: Allocate memory for all images
    CUdeviceptr d_images;
    size_t image_bytes = batch_size * IMG_SIZE * sizeof(uchar);
    CHECK_CUDA(cuMemAlloc(&d_images, image_bytes));

    // Step 2: Copy image data to device
    CHECK_CUDA(cuMemcpyHtoD(d_images, images, image_bytes));

    // Step 3: Launch hist_cal once per image
    for (uint64_t i = 0; i < batch_size; i++) {
        CUdeviceptr img_ptr = d_images + i * IMG_SIZE;

        void* args[] = {
            &img_ptr,     // uchar* image
            &i            // int index
        };

        CHECK_CUDA(cuLaunchKernel(
            hist_cal,
            1, 1, 1,       // Grid
            1024, 1, 1,    // Block
            0, 0,          // Shared mem, stream
            args, 0
        ));
    }

    CHECK_CUDA(cuCtxSynchronize());
    cuMemFree(d_images);
}



int main() {

    std::vector<std::string> all_images = get_all_valid_dat_images("images");
    std::vector<std::string> selected_refs = select_random_images(all_images, batch_size);

    std::vector<uchar> reference_images;
    for (const auto& path : selected_refs) {
        auto img = load_dat_image(path);
        reference_images.insert(reference_images.end(), img.begin(), img.end());
    }

    // === Step 1: Generate fake reference faces (gallery) ===
    // std::vector<uchar> reference_images = generate_reference_images(batch_size);

    // === Step 2: Initialize CUDA driver API ===
    CHECK_CUDA(cuInit(0));
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "face_verify.ptx"));

    // === Step 3: Build hist_dataset on GPU using init_gpu_face_verfication ===
    init_gpu_face_verfication(module, reference_images.data(), batch_size); 
    CHECK_CUDA(cuCtxSynchronize());

    // === Step 4: Get pointer to face_verfication_batch kernel ===
    CUfunction face_verification_kernel;
    CHECK_CUDA(cuModuleGetFunction(&face_verification_kernel, module, "face_verfication_batch"));

    // === Step 5: Prepare query images — identical to reference images ===
    std::vector<uchar> h_imges = reference_images;
    std::vector<int> h_idxes(batch_size);
    std::vector<double> h_total_dis(batch_size);

    for (auto& px : h_imges) px = rand() % 256;

    for (int i = 0; i < batch_size; ++i)
        h_idxes[i] = i;  // Compare image[i] to hist_dataset[i]

    // === Step 6: Allocate and copy data to device ===
    CUdeviceptr d_imges, d_idxes, d_total_dis;
    CHECK_CUDA(cuMemAlloc(&d_imges, h_imges.size()));
    CHECK_CUDA(cuMemAlloc(&d_idxes, h_idxes.size() * sizeof(int)));
    CHECK_CUDA(cuMemAlloc(&d_total_dis, h_total_dis.size() * sizeof(double)));

    CHECK_CUDA(cuMemcpyHtoD(d_imges, h_imges.data(), h_imges.size()));
    CHECK_CUDA(cuMemcpyHtoD(d_idxes, h_idxes.data(), h_idxes.size() * sizeof(int)));



    // === Step 7: Launch the kernel ===
    void* args[] = {
        &d_imges,
        &d_idxes,
        &d_total_dis,
        (void*)&batch_size
    };

    CHECK_CUDA(cuLaunchKernel(
        face_verification_kernel,
        1, 1, 1,       // gridDim
        256, 1, 1,     // blockDim
        0, 0,          // shared memory, stream
        args, 0
    ));

    CHECK_CUDA(cuCtxSynchronize());

    // === Step 8: Copy and print results ===
    CHECK_CUDA(cuMemcpyDtoH(h_total_dis.data(), d_total_dis, h_total_dis.size() * sizeof(double)));

    std::cout << "✅ Results from face_verfication_batch:\n";
    for (int i = 0; i < batch_size; ++i)
        std::cout << "Pair " << i << ": Distance = " << h_total_dis[i] << "\n";

    // === Cleanup ===
    cuMemFree(d_imges);
    cuMemFree(d_idxes);
    cuMemFree(d_total_dis);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
