#pragma once

#include <initializer_list>
#include <iostream>
#include <string>

// Basic device pointer creation with optional initialization
template <class T> T *create_device_ptr(const size_t n, const T &value) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);

    // Create temporary host array and initialize
    T *h_ptr = new T[n];
    for (size_t i = 0; i < n; i++) {
        h_ptr[i] = value;
    }

    cudaMemcpy(d_ptr, h_ptr, sizeof(T) * n, cudaMemcpyHostToDevice);
    delete[] h_ptr;
    return d_ptr;
}

// Device pointer creation without initialization
template <class T> T *create_device_ptr(const size_t n) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);
    return d_ptr;
}

// Device pointer creation from initializer list
template <class T> T *create_device_ptr(std::initializer_list<T> init_list) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * init_list.size());
    cudaMemcpy(
        d_ptr, init_list.begin(), sizeof(T) * init_list.size(), cudaMemcpyHostToDevice
    );
    return d_ptr;
}

// Host to device pointer conversion
template <class T> T *host_ptr_to_device_ptr(const T *h_ptr, const size_t n) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);
    cudaMemcpy(d_ptr, h_ptr, sizeof(T) * n, cudaMemcpyHostToDevice);
    return d_ptr;
}

// Device to host pointer conversion
template <class T> T *device_ptr_to_host_ptr(const T *d_ptr, const size_t n) {
    T *h_ptr = new T[n];
    cudaMemcpy(h_ptr, d_ptr, sizeof(T) * n, cudaMemcpyDeviceToHost);
    return h_ptr;
}

// CUDA error checking
void check_cuda_set_device() {
    cudaError_t err;
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[check_cuda_set_device failed] Error: %s\n", cudaGetErrorString(err));
    }
}

void check_cuda_get_last_error() {
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[check_cuda_get_last_error failed] Error: %s\n", cudaGetErrorString(err)
        );
    }
}

void check_cuda_device_synchronize() {
    cudaError_t err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf(
            "[check_cuda_device_synchronize failed] Error: %s\n",
            cudaGetErrorString(err)
        );
    }
}