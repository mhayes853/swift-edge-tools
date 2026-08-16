#ifndef EDGE_TEMPLATE_H
#define EDGE_TEMPLATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  EDGE_TEMPLATE_SUCCESS = 0,
  EDGE_TEMPLATE_FAILURE = 1,
  EDGE_TEMPLATE_INVALID_ARGUMENT = 2,
  EDGE_TEMPLATE_BUFFER_TOO_SMALL = 3,
};

const char *edge_template_last_error_message(void);

int32_t edge_template_render(
  const uint8_t *source,
  size_t source_count,
  const uint8_t *context_json,
  size_t context_json_count,
  uint8_t *text,
  size_t text_capacity,
  size_t *text_count
);

#ifdef __cplusplus
}
#endif

#endif
