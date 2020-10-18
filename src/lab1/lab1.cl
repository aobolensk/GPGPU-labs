__kernel void print_info() {
    size_t block_id = get_group_id(0);
    size_t thread_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    printf("I am from %lu block, %lu thread (global index: %lu)\n", block_id, thread_id, global_id);
}

__kernel void inc_buffer(__global int* array) {
    ulong id = get_global_id(0);
    array[id] += id;
}
