// Tests that eval_sequence_in_chunks gives results equivalent to serial eval.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <rwkv.h>
#include <assert.h> // xzl
#include "assertions.inc"

#define NUM_THREADS 1   // xzl

#ifdef GGML_USE_VTUNE
#include "ittnotify.h"
    enum{VT_CONV=0, VT_ENCODE, VT_CROSS, VT_SAMPLE1, VT_SAMPLE2, VT_DECODE, VT_MAX};
	__itt_domain * itt_domain = NULL;
	// __itt_string_handle * sh_sort = NULL; // the overall task name
    __itt_string_handle * sh_parts[VT_MAX] = {NULL}; // per part task name
	#define vtune_task_begin(X) __itt_task_begin(itt_domain, __itt_null, __itt_null, sh_parts[X])
	#define vtune_task_end() __itt_task_end(itt_domain)
#else 
	#define vtune_task_begin(X)
	#define vtune_task_end()
#endif

void test_on_prompt(const char * prompt, const size_t prompt_length) {

#ifdef GGML_USE_VTUNE
	// __itt_domain * itt_domain = NULL;
	// __itt_string_handle * sh_sort = NULL; // the overall task name
    itt_domain = __itt_domain_create("my domain");
	__itt_thread_set_name("my main");
    sh_parts[VT_CONV] = __itt_string_handle_create("conv"); 
    sh_parts[VT_ENCODE] = __itt_string_handle_create("encode"); 
    sh_parts[VT_CROSS] = __itt_string_handle_create("cross"); 
    sh_parts[VT_SAMPLE1] = __itt_string_handle_create("sample1"); 
    sh_parts[VT_SAMPLE2 ] = __itt_string_handle_create("sample2");
    sh_parts[VT_DECODE ] = __itt_string_handle_create("decode"); assert(sh_parts[VT_DECODE]);    
#endif


    fprintf(stderr, "Calculating expected state and logits for prompt of size %zd\n", prompt_length);

    // struct rwkv_context * ctx = rwkv_init_from_file("tiny-rwkv-5v2-730K-FP32.bin", 2);
    // struct rwkv_context * ctx = rwkv_init_from_file("d:\\workspace-rwkv\\RWKV-5-World-0.1B-v1-20230803-ctx4096.bin", NUM_THREADS);
    struct rwkv_context * ctx = rwkv_init_from_file("d:\\workspace-rwkv\\RWKV-5-World-3B-v2-20231113-ctx4096.bin", NUM_THREADS);

    printf("xzl: model loading done\n");

    ASSERT(ctx != NULL, "Unexpected error 0x%.8X", rwkv_get_last_error(NULL));

    float * expected_state = calloc(rwkv_get_state_len(ctx), sizeof(float));
    float * expected_logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(expected_state != NULL, "Failed to allocate state");
    ASSERT(expected_logits != NULL, "Failed to allocate logits");

    rwkv_eval(ctx, prompt[0], NULL, expected_state, expected_logits);

    for (size_t i = 1; prompt[i] != 0; i++) {           // xzl: eval on prompt sequentially 
        vtune_task_begin(VT_CONV); 
        rwkv_eval(ctx, prompt[i], expected_state /* xzl: state in*/, expected_state /*xzl: state out*/, expected_logits);
        vtune_task_end(); 
    }

    printf("xzl: eval done\n");
    getchar(); 
    exit(1); 

    // ---

    uint32_t * prompt_tokens = calloc(prompt_length, sizeof(uint32_t));

    for (int i = 0; i < prompt_length; i++) {
        prompt_tokens[i] = prompt[i];
    }

    // ---

    float * state = calloc(rwkv_get_state_len(ctx), sizeof(float));
    float * logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");
    ASSERT(logits != NULL, "Failed to allocate logits");

    const size_t chunk_sizes[4] = {1, 2, 8, 10};

    for (int i = 0; i < 4; i++) {           // xzl: eval prompts in batch
        size_t chunk_size = chunk_sizes[i];

        fprintf(stderr, "Testing chunk_size = %zd\n", chunk_size);

        rwkv_eval_sequence_in_chunks(ctx, prompt_tokens, prompt_length, chunk_size, NULL, state, logits);

        ASSERT(memcmp(expected_state, state, rwkv_get_state_len(ctx) * sizeof(float)) == 0, "Results are not identical");
    }

    // ---

    rwkv_free(ctx);

    free(logits);
    free(state);
    free(expected_logits);
    free(expected_state);
    free(prompt_tokens);
}

int main(void) {
    const char prompt1[70 + 1] = "This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM";
    test_on_prompt(prompt1, 70);

    const char prompt2[1 + 1] = "T";
    test_on_prompt(prompt2, 1);

    return 0;
}
