#include <node/node_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "needle.h"

static napi_value fail(napi_env env, const char* code, const char* message) {
    napi_throw_error(env, code, message);
    return NULL;
}

static int read_string(napi_env env, napi_value value, char** result, int optional) {
    napi_valuetype type;
    if (napi_typeof(env, value, &type) != napi_ok) return 0;
    if (optional && (type == napi_undefined || type == napi_null)) {
        *result = NULL;
        return 1;
    }
    if (type != napi_string) return 0;

    size_t length = 0;
    if (napi_get_value_string_utf8(env, value, NULL, 0, &length) != napi_ok) {
        return 0;
    }
    char* string = (char*)malloc(length + 1);
    if (!string) return 0;
    if (napi_get_value_string_utf8(env, value, string, length + 1, &length) != napi_ok) {
        free(string);
        return 0;
    }
    *result = string;
    return 1;
}

static napi_value load_weights(napi_env env, napi_callback_info info) {
    napi_value arguments[1];
    size_t argument_count = 1;
    if (napi_get_cb_info(env, info, &argument_count, arguments, NULL, NULL) != napi_ok ||
        argument_count != 1) {
        return fail(env, "ERR_ARGUMENTS", "needleLoad expects one Uint8Array.");
    }

    bool is_typed_array = false;
    napi_is_typedarray(env, arguments[0], &is_typed_array);
    if (!is_typed_array) {
        return fail(env, "ERR_ARGUMENTS", "needleLoad expects one Uint8Array.");
    }

    napi_typedarray_type type;
    size_t length;
    void* data;
    napi_value array_buffer;
    size_t offset;
    if (napi_get_typedarray_info(
            env,
            arguments[0],
            &type,
            &length,
            &data,
            &array_buffer,
            &offset
        ) != napi_ok || type != napi_uint8_array) {
        return fail(env, "ERR_ARGUMENTS", "needleLoad expects one Uint8Array.");
    }

    int result = needle_load((const unsigned char*)data, length);
    napi_value value;
    napi_create_int32(env, result, &value);
    return value;
}

static napi_value initialize(napi_env env, napi_callback_info info) {
    napi_value arguments[3];
    size_t argument_count = 3;
    if (napi_get_cb_info(env, info, &argument_count, arguments, NULL, NULL) != napi_ok ||
        argument_count < 2 || argument_count > 3) {
        return fail(env, "ERR_ARGUMENTS", "needleInit expects two or three strings.");
    }

    char* system_prompt = NULL;
    char* tools_json = NULL;
    char* tool_index_path = NULL;
    if (!read_string(env, arguments[0], &system_prompt, 0) ||
        !read_string(env, arguments[1], &tools_json, 0) ||
        (argument_count == 3 && !read_string(env, arguments[2], &tool_index_path, 1))) {
        free(system_prompt);
        free(tools_json);
        free(tool_index_path);
        return fail(env, "ERR_ARGUMENTS", "needleInit expects strings.");
    }

    int result = needle_init(system_prompt, tools_json, tool_index_path);
    free(system_prompt);
    free(tools_json);
    free(tool_index_path);

    napi_value value;
    napi_create_int32(env, result, &value);
    return value;
}

static napi_value complete(napi_env env, napi_callback_info info) {
    napi_value arguments[3];
    size_t argument_count = 3;
    if (napi_get_cb_info(env, info, &argument_count, arguments, NULL, NULL) != napi_ok ||
        argument_count != 3) {
        return fail(env, "ERR_ARGUMENTS", "needleComplete expects prompt, maxTokens, and outputCapacity.");
    }

    char* prompt = NULL;
    if (!read_string(env, arguments[0], &prompt, 0)) {
        return fail(env, "ERR_ARGUMENTS", "needleComplete expects a string prompt.");
    }

    int32_t max_tokens;
    int32_t output_capacity;
    if (napi_get_value_int32(env, arguments[1], &max_tokens) != napi_ok ||
        napi_get_value_int32(env, arguments[2], &output_capacity) != napi_ok ||
        max_tokens <= 0 || output_capacity <= 0) {
        free(prompt);
        return fail(env, "ERR_ARGUMENTS", "needleComplete expects positive integer limits.");
    }

    char* output = (char*)malloc((size_t)output_capacity);
    if (!output) {
        free(prompt);
        return fail(env, "ERR_ALLOCATION", "Could not allocate the Needle 2 output buffer.");
    }
    int result = needle_complete(prompt, max_tokens, output, output_capacity);
    free(prompt);
    if (result < 0) {
        free(output);
        napi_value value;
        napi_create_int32(env, result, &value);
        return value;
    }

    napi_value object;
    napi_value json;
    napi_value token_count;
    napi_create_object(env, &object);
    napi_create_string_utf8(env, output, NAPI_AUTO_LENGTH, &json);
    napi_create_int32(env, result, &token_count);
    napi_set_named_property(env, object, "json", json);
    napi_set_named_property(env, object, "tokenCount", token_count);
    free(output);
    return object;
}

static napi_value reset(napi_env env, napi_callback_info info) {
    needle_reset();
    napi_value value;
    napi_get_undefined(env, &value);
    return value;
}

NAPI_MODULE_INIT() {
    napi_property_descriptor properties[] = {
        {"needleLoad", NULL, load_weights, NULL, NULL, NULL, napi_default, NULL},
        {"needleInit", NULL, initialize, NULL, NULL, NULL, napi_default, NULL},
        {"needleComplete", NULL, complete, NULL, NULL, NULL, napi_default, NULL},
        {"needleReset", NULL, reset, NULL, NULL, NULL, napi_default, NULL}
    };
    napi_define_properties(env, exports, 4, properties);
    return exports;
}
