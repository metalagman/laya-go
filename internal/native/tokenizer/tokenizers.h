#ifndef LAYA_GO_TOKENIZERS_H_
#define LAYA_GO_TOKENIZERS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct tokenizers_encode_options {
  bool add_special_tokens;
  bool return_type_ids;
  bool return_tokens;
  bool return_special_tokens_mask;
  bool return_attention_mask;
  bool return_offsets;
};

struct tokenizers_options {
  bool encode_special_tokens;
};

struct tokenizers_buffer {
  uint32_t *ids;
  uint32_t *type_ids;
  uint32_t *special_tokens_mask;
  uint32_t *attention_mask;
  char **tokens;
  size_t *offsets;
  size_t len;
};

const char *tokenizers_version(void);
void tokenizers_version_1_26_0(void);

void *tokenizers_from_bytes(const uint8_t *config, uint32_t len,
                            const struct tokenizers_options *options,
                            char **error);
void *tokenizers_from_file(const char *config, char **error);
struct tokenizers_buffer tokenizers_encode(
    void *ptr, const char *message,
    const struct tokenizers_encode_options *options);
void tokenizers_free_tokenizer(void *ptr);
void tokenizers_free_buffer(struct tokenizers_buffer buffer);
void tokenizers_free_string(char *string);

#endif  // LAYA_GO_TOKENIZERS_H_
