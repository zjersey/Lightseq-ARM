#pragma once

#include "../proto/quant_transformer_weight.h"
#include <string>
#include <vector>

namespace lightseq {

class QuantEncoder {
private:
  // private member function
  void self_attention();
  void ffn_add_norm();

  float
      *_p_d_output; // encoder output, [batch_size, batch_seq_len, hidden_size]
  const QuantTransformerWeight &_tw;

  // buffer pointer
  float *_p_d_c;
  int8_t *_int8_ffn_in_buf;
  int8_t *_int8_ffn_out_buf;
  int32_t *_int32_ffn_out_buf;

  std::vector<float *> _scaled_ffn2_colsum;

  int _layer_id;
  int _weight_offset;

public:
  QuantEncoder(const QuantTransformerWeight &tw, float *p_d_output);
  void init_buffer();
  std::string check();
  void run_one_infer();

  int _seq_len;
  int *_p_d_token_id;
};

} // namespace lightseq
