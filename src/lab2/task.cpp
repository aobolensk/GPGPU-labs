#include <cstdio>
#include <cmath>
#include <omp.h>
#include <CL/cl.h>

#define CHK(expr)                                               \
    if (!(expr)) {                                              \
        fprintf(stderr, "Failed at %s:%d\n", __FILE__, __LINE__); \
        abort();                                                \
    }

void saxpy(size_t n, float a, float *x, int incx, float *y, int incy) {
    for (int i = 0; i < n; ++i) {
        y[i * incy] += a * x[i * incx];
    }
}

void daxpy(size_t n, double a, double *x, int incx, double *y, int incy) {
    for (int i = 0; i < n; ++i) {
        y[i * incy] += a * x[i * incx];
    }
}

void saxpy_omp(size_t n, float a, float *x, int incx, float *y, int incy) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i * incy] += a * x[i * incx];
    }
}

void daxpy_omp(size_t n, double a, double *x, int incx, double *y, int incy) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i * incy] += a * x[i * incx];
    }
}

long closest_bigger_degree_of_two(long x) {
    long d2 = 1;
    while (true) {
        if (d2 >= x) return d2;
        d2 *= 2;
    }
    return 0;
}

void saxpy_gpu(size_t n, float a, float *x, int incx, float *y, int incy) {
    cl_int ret = CL_SUCCESS;
    cl_uint platform_count = 0;
    size_t global_work_size = closest_bigger_degree_of_two(n * incy);
    CHK(!clGetPlatformIDs(0, nullptr, &platform_count));

    cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platform_count);
    CHK(!clGetPlatformIDs(platform_count, platforms, nullptr));
    for (cl_uint i = 0; i < platform_count; ++i) {
        char platformName[128];
        CHK(!clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr));
    }
    if (platform_count == 0) {
        fprintf(stderr, "No platform found!\n");
        abort();
    }

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

    cl_mem xs_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * incx, x, nullptr);
    CHK(xs_buff);

    cl_mem ys_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * incy, y, nullptr);
    CHK(ys_buff);

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, nullptr);

    cl_event buff_events[2];
    CHK(!clEnqueueWriteBuffer(queue, xs_buff, CL_FALSE, 0, sizeof(float) * n * incx, x, 0, nullptr, &buff_events[0]));
    CHK(!clEnqueueWriteBuffer(queue, ys_buff, CL_FALSE, 0, sizeof(float) * n * incy, y, 0, nullptr, &buff_events[1]));

    FILE *kernel_file = fopen("lab2.cl", "r");
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

    cl_kernel kernel = clCreateKernel(program, "saxpy_gpu", nullptr);

    size_t workgroup_size = 256;

    CHK(!clSetKernelArg(kernel, 0, sizeof(int), &n));
    CHK(!clSetKernelArg(kernel, 1, sizeof(float), &a));
    CHK(!clSetKernelArg(kernel, 2, sizeof(cl_mem), &xs_buff));
    CHK(!clSetKernelArg(kernel, 3, sizeof(int), &incx));
    CHK(!clSetKernelArg(kernel, 4, sizeof(cl_mem), &ys_buff));
    CHK(!clSetKernelArg(kernel, 5, sizeof(int), &incy));

    CHK(!clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workgroup_size, 2, buff_events, nullptr));
    CHK(!clFinish(queue));

    CHK(!clEnqueueReadBuffer(queue, ys_buff, CL_TRUE, 0, sizeof(float) * n * incy, y, 0, nullptr, nullptr));

    free(source);
    CHK(!clReleaseMemObject(xs_buff));
    CHK(!clReleaseMemObject(ys_buff));
    CHK(!clReleaseProgram(program));
    CHK(!clReleaseKernel(kernel));
    CHK(!clReleaseCommandQueue(queue));
    CHK(!clReleaseContext(context));
}

void daxpy_gpu(size_t n, double a, double *x, int incx, double *y, int incy) {
    cl_int ret = CL_SUCCESS;
    cl_uint platform_count = 0;
    size_t global_work_size = closest_bigger_degree_of_two(n * incy);
    CHK(!clGetPlatformIDs(0, nullptr, &platform_count));

    cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platform_count);
    CHK(!clGetPlatformIDs(platform_count, platforms, nullptr));
    for (cl_uint i = 0; i < platform_count; ++i) {
        char platformName[128];
        CHK(!clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr));
    }
    if (platform_count == 0) {
        fprintf(stderr, "No platform found!\n");
        abort();
    }

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

    cl_mem xs_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * n * incx, x, nullptr);
    CHK(xs_buff);

    cl_mem ys_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * n * incy, y, nullptr);
    CHK(ys_buff);

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, nullptr);

    cl_event buff_events[2];
    CHK(!clEnqueueWriteBuffer(queue, xs_buff, CL_FALSE, 0, sizeof(double) * n * incx, x, 0, nullptr, &buff_events[0]));
    CHK(!clEnqueueWriteBuffer(queue, ys_buff, CL_FALSE, 0, sizeof(double) * n * incy, y, 0, nullptr, &buff_events[1]));

    FILE *kernel_file = fopen("lab2.cl", "r");
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

    cl_kernel kernel = clCreateKernel(program, "daxpy_gpu", nullptr);

    size_t workgroup_size = 256;

    CHK(!clSetKernelArg(kernel, 0, sizeof(int), &n));
    CHK(!clSetKernelArg(kernel, 1, sizeof(double), &a));
    CHK(!clSetKernelArg(kernel, 2, sizeof(cl_mem), &xs_buff));
    CHK(!clSetKernelArg(kernel, 3, sizeof(int), &incx));
    CHK(!clSetKernelArg(kernel, 4, sizeof(cl_mem), &ys_buff));
    CHK(!clSetKernelArg(kernel, 5, sizeof(int), &incy));

    CHK(!clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workgroup_size, 2, buff_events, nullptr));
    CHK(!clFinish(queue));

    CHK(!clEnqueueReadBuffer(queue, ys_buff, CL_TRUE, 0, sizeof(double) * n * incy, y, 0, nullptr, nullptr));

    free(source);
    CHK(!clReleaseMemObject(xs_buff));
    CHK(!clReleaseMemObject(ys_buff));
    CHK(!clReleaseProgram(program));
    CHK(!clReleaseKernel(kernel));
    CHK(!clReleaseCommandQueue(queue));
    CHK(!clReleaseContext(context));
}

template <typename Func, typename... Args>
void bench(const char *name, Func f, Args... args) {
    printf("Started %s\n", name);
    double start = omp_get_wtime();
    f(args...);
    double finish = omp_get_wtime();
    printf("%s execution time: %lf\n", name, finish - start);
}

const double eps = 1e-5;

template <typename T>
bool validate_results(T *actual, T *reference, int n) {
    bool f = true;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        f &= !(std::abs(actual[i] - reference[i]) > eps);
    }
    return f;
}

void float_test() {
    size_t n;
    int incx, incy;
    float *x = nullptr, *y = nullptr, a;
    auto reset = [&]() {
        n = 52'000'000;
        incx = 3;
        incy = 2;
        a = .3f;
        free(x);
        free(y);
        x = (float *) malloc(n * incx * sizeof(float));
        y = (float *) malloc(n * incy * sizeof(float));
        for (int i = 0; i < n * incx; ++i) {
            x[i] = .1f * (i % 10);
        }
        for (int i = 0; i < n * incy; ++i) {
            y[i] = .1f * (i % 10);
        }
    };
    reset();
    bench("saxpy", saxpy, n, a, x, incx, y, incy);
    float *ref_y = (float *) malloc(n * incy * sizeof(float));
    for (int i = 0; i < n * incy; ++i)
        ref_y[i] = y[i];
    reset();
    bench("saxpy_omp", saxpy_omp, n, a, x, incx, y, incy);
    CHK(validate_results(y, ref_y, n * incy));
    reset();
    bench("saxpy_gpu", saxpy_gpu, n, a, x, incx, y, incy);
    CHK(validate_results(y, ref_y, n * incy));
    free(x);
    free(y);
}

void double_test() {
    size_t n;
    int incx, incy;
    double *x = nullptr, *y = nullptr, a;
    auto reset = [&]() {
        n = 20'000'000;
        incx = 3;
        incy = 2;
        a = .3;
        free(x);
        free(y);
        x = (double *) malloc(n * incx * sizeof(double));
        y = (double *) malloc(n * incy * sizeof(double));
        for (int i = 0; i < n * incx; ++i) {
            x[i] = .1 * (i % 10);
        }
        for (int i = 0; i < n * incy; ++i) {
            y[i] = .1 * (i % 10);
        }
    };
    reset();
    bench("daxpy", daxpy, n, a, x, incx, y, incy);
    double *ref_y = (double *) malloc(n * incy * sizeof(double));
    for (int i = 0; i < n * incy; ++i)
        ref_y[i] = y[i];
    reset();
    bench("daxpy_omp", daxpy_omp, n, a, x, incx, y, incy);
    CHK(validate_results(y, ref_y, n * incy));
    reset();
    bench("daxpy_gpu", daxpy_gpu, n, a, x, incx, y, incy);
    CHK(validate_results(y, ref_y, n * incy));
    free(x);
    free(y);
}

int main(int argc, char *argv[]) {
    float_test();
    double_test();
    return 0;
}
