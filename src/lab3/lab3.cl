__kernel void matrix_multiply_naive(__global int* a, __global int* b, __global int* c, int n, int m, int l) {
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);

    __private int res = 0;

    for (size_t i = 0; i < m; ++i) {
        res += a[global_id1 * m + i] * b[i * l + global_id0];
    }
    c[l * global_id1 + global_id0] = res;
}

#define BLOCK_SIZE 16

__kernel void matrix_multiply_optimized(__global int* a, __global int* b, __global int* c, int n, int m, int l) {
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);
    size_t local_id0 = get_local_id(0);
    size_t local_id1 = get_local_id(1);

    __local int a_coord[BLOCK_SIZE][BLOCK_SIZE];
    __local int b_coord[BLOCK_SIZE][BLOCK_SIZE];
    __private int res = 0;

    for (size_t i = 0; i < m / BLOCK_SIZE; ++i) {
        a_coord[local_id1][local_id0] = a[global_id1 * m + i * BLOCK_SIZE + local_id0];
        b_coord[local_id1][local_id0] = b[(i * BLOCK_SIZE + local_id1) * l + global_id0];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t j = 0; j < BLOCK_SIZE; ++j)
            res += a_coord[local_id1][j] * b_coord[j][local_id0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[l * global_id1 + global_id0] = res;
}

__kernel void matrix_multiply_images(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int n, int m, int l) {
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);
    size_t local_id0 = get_local_id(0);
    size_t local_id1 = get_local_id(1);

    __local int sub_arr_a[BLOCK_SIZE][BLOCK_SIZE];
    __local int sub_arr_b[BLOCK_SIZE][BLOCK_SIZE];
    __private int res = 0;

    for (size_t i = 0; i < m / BLOCK_SIZE; ++i) {
        int2 a_coord = (int2) (global_id0, i * BLOCK_SIZE + local_id1);
        int2 b_coord = (int2) (i * BLOCK_SIZE + local_id0, global_id1);

        sub_arr_a[local_id1][local_id0] = read_imagei(a, a_coord).x;
        sub_arr_b[local_id1][local_id0] = read_imagei(b, b_coord).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t j = 0; j < BLOCK_SIZE; ++j)
            res += sub_arr_a[local_id1][j] * sub_arr_b[j][local_id1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int2 c_coord = (int2) (global_id0, global_id1);
    write_imagei(c, c_coord, (int4)(res, 0, 0, 1));
}
