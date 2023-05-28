#include "quant_encoder.h"
#include "../../tests/util.h"
#include "../ops/ops.h"

namespace lightseq {

QuantEncoder::QuantEncoder(const QuantTransformerWeight &tw, float *p_d_output)
    : _p_d_output(p_d_output), _tw(tw) {}

void QuantEncoder::init_buffer() {
  int max_batch_dim =
      _tw._max_step * std::max(_tw._inner_size, _tw._hidden_size * 3);
  size_t buffer_size =
      (_tw._max_step * _tw._hidden_size + max_batch_dim) * sizeof(int8_t) +
      _tw._head_num * _tw._max_step * _tw._max_step * sizeof(float) +
      max_batch_dim * sizeof(int32_t);

  void *ptr = malloc(buffer_size);
  int8_t *p_int8 = (int8_t *)ptr;

  _int8_ffn_in_buf = p_int8;
  p_int8 += _tw._max_step * _tw._hidden_size;

  _int8_ffn_out_buf = p_int8;
  p_int8 += max_batch_dim;

  float *p_fp = (float *)p_int8;
  _p_d_c = p_fp;
  p_fp += _tw._head_num * _tw._max_step * _tw._max_step;

  int32_t *p_i32 = (int32_t *)p_fp;
  _int32_ffn_out_buf = p_i32;
  p_i32 += max_batch_dim;

  printf("encoder buffer size: %.1f MB\n", (float)buffer_size / 1000000);
  printf("encoder buffer init succeed\n");
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
std::string QuantEncoder::check() { return ""; }

void QuantEncoder::run_one_infer() {
  if (_tw._share_all_embedding) {
    decoder_embedding_int8(_p_d_token_id, _tw._p_d_src_emb_wei_i8[0],
                           _tw._p_d_src_emb_wei_fp[0], _p_d_output, _seq_len,
                           _tw._src_vocab_size, _tw._hidden_size,
                           _tw._src_emb_clip_max / _tw._quant_range, true);
  } else {
    embedding_int8(_p_d_token_id, _tw._p_d_src_emb_wei_i8[0],
                   _tw._p_d_src_emb_wei_fp[0], _p_d_output, _seq_len,
                   _tw._hidden_size, _tw._src_emb_clip_max / _tw._quant_range,
                   true);
  }

#ifdef DEBUG
  print_vec(_p_d_output, "emb out, head", 10);
  print_vec(_p_d_output + _seq_len * _tw._hidden_size - 10, "emb out, tail",
            10);
  print_vec(_tw._p_d_src_emb_wei_i8[0], "token embedding weight", 10);
  print_sum(_tw._p_d_src_emb_wei_i8[0], "token embedding weight",
            _tw._src_vocab_size * _tw._hidden_size);
  print_vec(_tw._p_d_src_emb_wei_fp[0], "position embedding weight", 10);
#endif

  layernorm_residual_i8O(
      _p_d_output, _int8_ffn_in_buf, _tw._p_d_enc_wei_fp[0],
      _tw._p_d_enc_wei_fp[1], _tw._p_d_enc_wei_fp[3], _tw._hidden_size,
      _seq_len, _tw._quant_range / _tw._enc_clip_max[4], _tw._is_post_ln);
#ifdef DEBUG
  print_vec(_p_d_output, "first layernorm, float out, head", 10);
  print_vec(_p_d_output + _seq_len * _tw._hidden_size - 10,
            "first layernorm, float out, tail", 10);
  print_vec(_int8_ffn_in_buf, "first layernorm, int8 out, head", 10);
  print_vec(_int8_ffn_in_buf + _seq_len * _tw._hidden_size - 10,
            "first layernorm, int8 out, tail", 10);
#endif

  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; ++_layer_id) {
    self_attention();
    ffn_add_norm();
  }
}

void QuantEncoder::self_attention() {

#ifdef DEBUG
  print_vec(_int8_ffn_in_buf, "encoder::self attn in(head)", 10);
  print_vec(_int8_ffn_in_buf + _seq_len * _tw._hidden_size - 10,
            "encoder::self attn in(tail)", 10);
  print_sum(_int8_ffn_in_buf, "encoder::self attn in",
            _seq_len * _tw._hidden_size);
#endif

  gemm_int8_arm(_int8_ffn_in_buf, _tw._p_d_enc_wei_i8[4 * _layer_id],
                _int8_ffn_out_buf, _seq_len, _tw._hidden_size,
                _tw._hidden_size * 3,
                _tw._enc_clip_max[_layer_id * 12] *
                    _tw._enc_clip_max[_layer_id * 12 + 4] /
                    (_tw._enc_clip_max[_layer_id * 12 + 8] * _tw._quant_range),
                _int32_ffn_out_buf);

  multi_head_attention(
      _int8_ffn_in_buf, _tw._p_d_enc_wei_fp[8 * _layer_id + 2],
      _int8_ffn_out_buf, _p_d_c, _seq_len, _tw._hidden_size, _tw._head_num,
      _tw._enc_clip_max[_layer_id * 12 + 8] / _tw._quant_range,
      _tw._quant_range / _tw._enc_clip_max[_layer_id * 12 + 5]);

  gemm_int8_arm(_int8_ffn_in_buf, _tw._p_d_enc_wei_i8[4 * _layer_id + 1],
                _int8_ffn_out_buf, _seq_len, _tw._hidden_size, _tw._hidden_size,
                _tw._enc_clip_max[_layer_id * 12 + 1] *
                    _tw._enc_clip_max[_layer_id * 12 + 5] /
                    (_tw._enc_clip_max[_layer_id * 12 + 9] * _tw._quant_range),
                _int32_ffn_out_buf);

  float scale = _tw._enc_clip_max[_layer_id * 12 + 9] / _tw._quant_range;
#pragma omp parallel for
  for (int i = 0; i < _seq_len * _tw._hidden_size; ++i) {
    _p_d_output[i] += (float)_int8_ffn_out_buf[i] * scale;
  }

  layernorm_residual_i8O(
      _p_d_output, _int8_ffn_in_buf, _tw._p_d_enc_wei_fp[8 * _layer_id + 4],
      _tw._p_d_enc_wei_fp[8 * _layer_id + 5],
      _tw._p_d_enc_wei_fp[8 * _layer_id + 7], _tw._hidden_size, _seq_len,
      _tw._quant_range / _tw._enc_clip_max[_layer_id * 12 + 6],
      _tw._is_post_ln);
}

void QuantEncoder::ffn_add_norm() {
#ifdef DEBUG
  print_vec(_int8_ffn_in_buf, "encoder::ffn in(head)", 10);
  print_vec(_int8_ffn_in_buf + _seq_len * _tw._hidden_size - 10,
            "encoder::ffn in(tail)", 10);
  print_sum(_int8_ffn_in_buf, "encoder::ffn in", _seq_len * _tw._hidden_size);
#endif

  gemm_int8_arm(_int8_ffn_in_buf, _tw._p_d_enc_wei_i8[4 * _layer_id + 2],
                _int8_ffn_out_buf, _seq_len, _tw._hidden_size, _tw._inner_size,
                _tw._enc_clip_max[_layer_id * 12 + 2] *
                    _tw._enc_clip_max[_layer_id * 12 + 6] /
                    (_tw._enc_clip_max[_layer_id * 12 + 10] * _tw._quant_range),
                _int32_ffn_out_buf);

#ifdef DEBUG
  print_vec(_int8_ffn_out_buf, "encoder fc1 kernel out int8, head", 10);
  print_vec(_int8_ffn_out_buf + _seq_len * _tw._inner_size - 10,
            "encoder fc1 kernel out int8, tail", 10);
  print_sum(_int8_ffn_out_buf, "encoder fc1 kernel out int8",
            _seq_len * _tw._inner_size);
#endif

  if (_tw._use_gelu) {
    bias_gelu_i8IO(_int8_ffn_out_buf, _tw._p_d_enc_wei_fp[8 * _layer_id + 6],
                   _seq_len, _tw._inner_size,
                   _tw._enc_clip_max[_layer_id * 12 + 10] / _tw._quant_range,
                   _tw._quant_range / _tw._enc_clip_max[_layer_id * 12 + 7]);
  } else {
    bias_relu_i8IO(_int8_ffn_out_buf, _tw._p_d_enc_wei_fp[8 * _layer_id + 6],
                   _seq_len, _tw._inner_size,
                   _tw._enc_clip_max[_layer_id * 12 + 10] / _tw._quant_range,
                   _tw._quant_range / _tw._enc_clip_max[_layer_id * 12 + 7],
                   _tw._enc_clip_max[_layer_id * 12 + 7]);
  }

#ifdef DEBUG
  print_vec(_int8_ffn_out_buf, "encoder fc2 kernel in int8, head", 10);
  print_vec(_int8_ffn_out_buf + _seq_len * _tw._inner_size - 10,
            "encoder fc2 kernel in int8, tail", 10);
  print_sum(_int8_ffn_out_buf, "encoder fc2 kernel in int8",
            _seq_len * _tw._inner_size);
#endif

  gemm_int8_arm(_int8_ffn_out_buf, _tw._p_d_enc_wei_i8[4 * _layer_id + 3],
                nullptr, _seq_len, _tw._inner_size, _tw._hidden_size, 0.f,
                _int32_ffn_out_buf);

#ifdef DEBUG
  print_vec(_int32_ffn_out_buf, "encoder fc2 kernel out int32, head", 10);
  print_vec(_int32_ffn_out_buf + _seq_len * _tw._hidden_size - 10,
            "encoder fc2 kernel out int32, tail", 10);
#endif

  float dequant_scale;
  dequant_scale = _tw._enc_clip_max[_layer_id * 12 + 3] *
                  _tw._enc_clip_max[_layer_id * 12 + 7] /
                  (_tw._quant_range * _tw._quant_range);

  if (_layer_id == _tw._n_enc_layer - 1) {
    residual_bias_ln_i32I(
        _int32_ffn_out_buf, nullptr, _tw._p_d_src_emb_wei_fp[1],
        _tw._p_d_src_emb_wei_fp[2], _p_d_output, nullptr, _tw._hidden_size,
        _seq_len, dequant_scale, 0.f, _tw._is_post_ln, nullptr);
  } else {
    residual_bias_ln_i32I(
        _int32_ffn_out_buf, _tw._p_d_enc_wei_fp[8 * (_layer_id + 1) + 3],
        _tw._p_d_enc_wei_fp[8 * (_layer_id + 1)],
        _tw._p_d_enc_wei_fp[8 * (_layer_id + 1) + 1], _p_d_output,
        _int8_ffn_in_buf, _tw._hidden_size, _seq_len, dequant_scale,
        _tw._quant_range / _tw._enc_clip_max[(_layer_id + 1) * 12 + 4],
        _tw._is_post_ln, nullptr);
  }

#ifdef DEBUG
  print_vec(_p_d_output, "encoder after ffn:ln+residual, head", 10);
  print_vec(_p_d_output + _seq_len * _tw._hidden_size - 10,
            "encoder after ffn:ln+residual, tail", 10);
  print_vec(_int8_ffn_in_buf, "encoder attn input int8, head", 10);
  print_vec(_int8_ffn_in_buf + _seq_len * _tw._hidden_size - 10,
            "encoder attn input int8, tail", 10);
#endif
}

} // namespace lightseq
