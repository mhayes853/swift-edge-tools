#ifndef MINJA_BRIDGING_H
#define MINJA_BRIDGING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *edge_template_last_error_message(void);

size_t edge_template_render(
  const char *source,
  const char *context_json,
  char *text,
  size_t text_capacity
);

#ifdef __cplusplus
}
#endif

#endif
