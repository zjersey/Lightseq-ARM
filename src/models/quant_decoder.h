#pragma once

#include "../proto/quant_transformer_weight.h"
#include "../utils/profiler.h"
#include <string>
#include <vector>

namespace lightseq {

class QuantDecoder {
private:
  // private member function
  void self_attention();
  void encdec_attention();
  void ffn_add_norm();
  void project_encoder_output();
  bool run_step();

  const QuantTransformerWeight &_tw;
  const float *_p_d_encoder_output;

  // buffer pointer
  int8_t *_p_d_encoder_out_buf;
  float *_p_d_cur_step_query;
  float *_p_d_c;
  std::vector<int8_t *> _p_d_self_k_cache;
  std::vector<int8_t *> _p_d_self_v_cache;
  std::vector<int8_t *> _p_d_encdec_k_cache;
  std::vector<int8_t *> _p_d_encdec_v_cache;

  int8_t *_int8_ffn_in_buf;
  int8_t *_int8_ffn_out_buf;
  int32_t *_int32_ffn_out_buf;

  std::vector<float *> _scaled_ffn2_colsum;

  int _layer_id;
  int _weight_offset;

  Profiler profiler;

public:
  QuantDecoder(const QuantTransformerWeight &tw,
               const float *p_d_encoder_output);
  void init_buffer();
  std::string check();
  void run_one_infer();

  int _seq_len;
  int _cur_step;
  int *_p_d_result;
  int *_p_d_alive_seq;
};

} // namespace lightseq
