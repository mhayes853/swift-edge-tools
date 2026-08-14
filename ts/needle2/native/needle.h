#ifndef NEEDLE_H
#define NEEDLE_H

#define NEEDLE_API

NEEDLE_API int needle_init(
    const char* system_prompt,
    const char* tools_json,
    const char* tool_index_path
);

NEEDLE_API int needle_complete(
    const char* input,
    int max_new_tokens,
    char* out,
    int out_capacity
);

NEEDLE_API void needle_reset(void);

NEEDLE_API int needle_load(
    const unsigned char* cact,
    unsigned long long n
);

#endif
