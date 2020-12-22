#include <cstdio>
#include <omp.h>
#include <CL/cl.h>

#define CHK(expr)                                               \
    if (!(expr)) {                                              \
        fprintf(stderr, "Failed at %s:%d\n", __FILE__, __LINE__); \
        abort();                                                \
    }

constexpr int BLOCK_SIZE = 16;

struct Matrix {
    int width, height;
    int *data;
};


template <typename Func, typename... Args>
void bench(const char *name, int times, Func f, Args... args) {
    for (int i = 0; i < times; ++i) {
        printf("Started %s:%d\n", name, i + 1);
        double start = omp_get_wtime();
        f(args...);
        double finish = omp_get_wtime();
        printf("%s:%d execution time: %lf\n", name, i + 1, finish - start);
    }
}

void matrix_multiply_seq(const Matrix &a, const Matrix &b, Matrix &res) {
    for (size_t i = 0; i < a.height; ++i) {
        for (size_t j = 0; j < b.width; ++j) {
            for (size_t k = 0; k < a.width; ++k) {
                res.data[i * res.width + j] += a.data[i * a.width + k] * b.data[k * b.width + j];
            }
        }
    }
}

void matrix_multiply_omp(const Matrix &a, const Matrix &b, Matrix &res) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.height; ++i) {
        for (size_t j = 0; j < b.width; ++j) {
            for (size_t k = 0; k < a.width; ++k) {
                res.data[i * res.width + j] += a.data[i * a.width + k] * b.data[k * b.width + j];
            }
        }
    }
}

void matrix_multiply_gpu_buffers(const Matrix &a, const Matrix &b, Matrix &res, const char *program_name) {
    cl_int ret = CL_SUCCESS;
    cl_uint platform_count = 0;
    CHK(!clGetPlatformIDs(0, nullptr, &platform_count));

    cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platform_count);
    CHK(!clGetPlatformIDs(platform_count, platforms, nullptr));
    for (cl_uint i = 0; i < platform_count; ++i) {
        char platformName[128];
        CHK(!clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr));
    }
    CHK(platform_count);

    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0
    };
    cl_uint device_count = 0;
    CHK(!clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count));
    CHK(device_count);
    cl_device_id *devices = (cl_device_id *) malloc(sizeof(cl_device_id) * device_count);
    CHK(devices);
    CHK(!clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, device_count, devices, nullptr));
    cl_context context = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &ret);
    CHK(context);

    cl_mem a_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * a.width * a.height, a.data, nullptr);
    CHK(a_buff);

    cl_mem b_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * b.width * b.height, b.data, nullptr);
    CHK(b_buff);

    cl_mem res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * res.width * res.height, res.data, nullptr);
    CHK(res_buff);

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, nullptr);

    cl_event buff_events[3];
    CHK(!clEnqueueWriteBuffer(queue, a_buff, CL_FALSE, 0, sizeof(int) * a.width * a.height, a.data, 0, nullptr, &buff_events[0]));
    CHK(!clEnqueueWriteBuffer(queue, b_buff, CL_FALSE, 0, sizeof(int) * b.width * b.height, b.data, 0, nullptr, &buff_events[1]));
    CHK(!clEnqueueWriteBuffer(queue, res_buff, CL_FALSE, 0, sizeof(int) * res.width * res.height, res.data, 0, nullptr, &buff_events[2]));

    FILE *kernel_file = fopen("lab3.cl", "r");
    fseek(kernel_file, 0, SEEK_END);
    size_t source_len = ftell(kernel_file);
    fseek(kernel_file, 0, SEEK_SET);
    char *source = (char *) calloc(1, source_len + 1);
    fread(source, 1, source_len, kernel_file);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_len, nullptr);
    if (clBuildProgram(program, 1, &devices[0], nullptr, nullptr, nullptr) == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);

        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "%s\n", log);
        abort();
    }

    cl_kernel kernel = clCreateKernel(program, program_name, nullptr);

    CHK(!clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buff));
    CHK(!clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buff));
    CHK(!clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff));
    CHK(!clSetKernelArg(kernel, 3, sizeof(int), &a.width));
    CHK(!clSetKernelArg(kernel, 4, sizeof(int), &b.width));
    CHK(!clSetKernelArg(kernel, 5, sizeof(int), &b.height));

    const size_t global_work_size[2] = {size_t(a.width), size_t(b.height)};
    const size_t local_work_size[2] = {BLOCK_SIZE, BLOCK_SIZE};

    printf("Started kernel\n");
    double start = omp_get_wtime();
    CHK(!clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, local_work_size, 3, buff_events, nullptr));
    CHK(!clFinish(queue));
    double finish = omp_get_wtime();
    printf("Kernel execution time: %lf\n", finish - start);

    CHK(!clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(int) * res.width * res.height, res.data, 0, nullptr, nullptr));

    free(source);
    CHK(!clReleaseMemObject(a_buff));
    CHK(!clReleaseMemObject(b_buff));
    CHK(!clReleaseMemObject(res_buff));
    CHK(!clReleaseProgram(program));
    CHK(!clReleaseKernel(kernel));
    CHK(!clReleaseCommandQueue(queue));
    CHK(!clReleaseContext(context));
}

void matrix_multiply_gpu_images(const Matrix &a, const Matrix &b, Matrix &res, const char *program_name) {
    cl_int ret = CL_SUCCESS;
    cl_uint platform_count = 0;
    CHK(!clGetPlatformIDs(0, nullptr, &platform_count));

    cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platform_count);
    CHK(!clGetPlatformIDs(platform_count, platforms, nullptr));
    for (cl_uint i = 0; i < platform_count; ++i) {
        char platformName[128];
        CHK(!clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr));
    }
    CHK(platform_count);

    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0
    };
    cl_uint device_count = 0;
    CHK(!clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count));
    CHK(device_count);
    cl_device_id *devices = (cl_device_id *) malloc(sizeof(cl_device_id) * device_count);
    CHK(devices);
    CHK(!clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, device_count, devices, nullptr));
    cl_context context = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &ret);
    CHK(context);

    cl_image_format form;
    form.image_channel_order = CL_R;
    form.image_channel_data_type = CL_SIGNED_INT32;

    cl_mem a_buff = clCreateImage2D(context, CL_MEM_READ_ONLY, &form, a.width, a.height, 0, a.data, nullptr);
    CHK(a_buff);

    cl_mem b_buff = clCreateImage2D(context, CL_MEM_READ_ONLY, &form, b.width, b.height, 0, b.data, nullptr);
    CHK(b_buff);

    cl_mem res_buff = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &form, res.width, res.height, 0, res.data, nullptr);
    CHK(res_buff);

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, nullptr);

    cl_event img_events[3];
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {0, 0, 1};
    region[0] = a.width; region[1] = a.height;
    CHK(!clEnqueueWriteImage(queue, a_buff, CL_FALSE, origin, region, 0, 0, a.data, 0, nullptr, &img_events[0]));
    region[0] = b.width; region[1] = b.height;
    CHK(!clEnqueueWriteImage(queue, b_buff, CL_FALSE, origin, region, 0, 0, b.data, 0, nullptr, &img_events[1]));
    region[0] = res.width; region[1] = res.height;
    CHK(!clEnqueueWriteImage(queue, res_buff, CL_FALSE, origin, region, 0, 0, res.data, 0, nullptr, &img_events[2]));

    FILE *kernel_file = fopen("lab3.cl", "r");
    fseek(kernel_file, 0, SEEK_END);
    size_t source_len = ftell(kernel_file);
    fseek(kernel_file, 0, SEEK_SET);
    char *source = (char *) calloc(1, source_len + 1);
    fread(source, 1, source_len, kernel_file);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_len, nullptr);
    if (clBuildProgram(program, 1, &devices[0], nullptr, nullptr, nullptr) == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);

        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "%s\n", log);
        abort();
    }

    cl_kernel kernel = clCreateKernel(program, program_name, nullptr);

    CHK(!clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buff));
    CHK(!clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buff));
    CHK(!clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff));
    CHK(!clSetKernelArg(kernel, 3, sizeof(int), &a.width));
    CHK(!clSetKernelArg(kernel, 4, sizeof(int), &b.width));
    CHK(!clSetKernelArg(kernel, 5, sizeof(int), &b.height));

    const size_t global_work_size[2] = {size_t(a.width), size_t(b.height)};
    const size_t local_work_size[2] = {BLOCK_SIZE, BLOCK_SIZE};

    printf("Started kernel\n");
    double start = omp_get_wtime();
    CHK(!clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, local_work_size, 3, img_events, nullptr));
    CHK(!clFinish(queue));
    double finish = omp_get_wtime();
    printf("Kernel execution time: %lf\n", finish - start);

    CHK(!clEnqueueReadImage(queue, res_buff, CL_TRUE, origin, region, 0, 0, res.data, 0, nullptr, nullptr));

    free(source);
    CHK(!clReleaseMemObject(a_buff));
    CHK(!clReleaseMemObject(b_buff));
    CHK(!clReleaseMemObject(res_buff));
    CHK(!clReleaseProgram(program));
    CHK(!clReleaseKernel(kernel));
    CHK(!clReleaseCommandQueue(queue));
    CHK(!clReleaseContext(context));
}

void validate_results(const char *name, Matrix &actual, Matrix &reference) {
    if (actual.width * actual.height != reference.width * reference.height) {
        printf("ERROR: '%s' wrong result!!!\n", name);
        return;
    }
    bool f = true;
    #pragma omp parallel for
    for (int i = 0; i < actual.width * actual.height; ++i) {
        f &= (actual.data[i] == reference.data[i]);
    }
    if (!f) {
        printf("ERROR: '%s' wrong result!!!\n", name);
    }
}

void matrix_fill_random(Matrix a) {
    for (int i = 0; i < a.height; ++i) {
        for (int j = 0; j < a.width; ++j) {
            a.data[i * a.width + j] = rand() % 100;
        }
    }
}

#define NEW_MAT(w, h) {                         \
    .width = w,                                 \
    .height = h,                                \
    .data = (int *)calloc(w * h, sizeof(int)),  \
}

void matrix_test() {
    // constexpr int n = 640, m = 640, l = 640;
    constexpr int n = 800, m = 640, l = 800;
    // constexpr int n = 960, m = 960, l = 960;
    // constexpr int n = 128, m = 128, l = 128;
    static_assert(n % BLOCK_SIZE == 0 && m % BLOCK_SIZE == 0 && l % BLOCK_SIZE == 0);
    Matrix mat1 = NEW_MAT(n, m);
    Matrix mat2 = NEW_MAT(m, l);
    Matrix mat3 = NEW_MAT(n, l);
    Matrix mat4 = NEW_MAT(n, l);
    Matrix mat5 = NEW_MAT(n, l);
    Matrix mat6 = NEW_MAT(n, l);
    Matrix mat7 = NEW_MAT(n, l);
    printf("------------------------------------------------\n");
    bench("seq", 3, matrix_multiply_seq, mat1, mat2, mat3);
    printf("------------------------------------------------\n");
    bench("omp", 3, matrix_multiply_omp, mat1, mat2, mat4);
    validate_results("omp", mat3, mat4);
    printf("------------------------------------------------\n");
    bench("gpu_naive", 3, matrix_multiply_gpu_buffers, mat1, mat2, mat5, "matrix_multiply_naive");
    validate_results("gpu_naive", mat3, mat5);
    printf("------------------------------------------------\n");
    bench("gpu_optimized", 3, matrix_multiply_gpu_buffers, mat1, mat2, mat6, "matrix_multiply_optimized");
    validate_results("gpu_optimized", mat3, mat6);
    printf("------------------------------------------------\n");
    bench("gpu_images", 3, matrix_multiply_gpu_images, mat1, mat2, mat7, "matrix_multiply_images");
    validate_results("gpu_images", mat3, mat7);
}

int main(int argc, char *argv[]) {
    matrix_test();
    return 0;
}
