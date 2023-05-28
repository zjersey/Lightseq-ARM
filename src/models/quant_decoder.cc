#include "quant_decoder.h"
#include "../../tests/util.h"
#include "../ops/ops.h"

namespace lightseq {

QuantDecoder::QuantDecoder(const QuantTransformerWeight &tw,
                           const float *p_d_encoder_output)
    : _p_d_encoder_output(p_d_encoder_output), _tw(tw) {}

void QuantDecoder::init_buffer() {
  int max_batch_dim = std::max(_tw._inner_size, _tw._hidden_size * 3);

  size_t buffer_size =
      (_tw._hidden_size + _tw._head_num * _tw._max_greedy_step) *
          sizeof(float) +
      (_tw._max_greedy_step +
       std::max(std::max(max_batch_dim, _tw._trg_vocab_size),
                _tw._max_step * _tw._hidden_size * 2 * _tw._n_dec_layer)) *
          sizeof(int) +
      (max_batch_dim + std::max(max_batch_dim, _tw._trg_vocab_size) +
       _tw._n_dec_layer * 2 * _tw._max_greedy_step * _tw._hidden_size +
       _tw._max_step * _tw._hidden_size * 4 * _tw._n_dec_layer) *
          sizeof(int8_t);

  void *ptr = malloc(buffer_size);
  float *p_fp = (float *)ptr;

  _p_d_cur_step_query = p_fp;
  p_fp += _tw._hidden_size;

  _p_d_c = p_fp;
  p_fp += _tw._head_num * _tw._max_greedy_step;

  int *p_int32 = (int *)p_fp;
  _p_d_alive_seq = p_int32;
  p_int32 += _tw._max_greedy_step;

  _int32_ffn_out_buf = (int32_t *)p_int32;
  p_int32 += std::max(std::max(max_batch_dim, _tw._trg_vocab_size),
                      _tw._max_step * _tw._hidden_size * 2 * _tw._n_dec_layer);

  int8_t *p_int8 = (int8_t *)p_int32;
  _int8_ffn_in_buf = p_int8;
  p_int8 += max_batch_dim;

  _int8_ffn_out_buf = p_int8;
  p_int8 += std::max(max_batch_dim, _tw._trg_vocab_size);

  int8_t *sliding_p = p_int8;
  p_int8 += _tw._n_dec_layer * 2 * _tw._max_greedy_step * _tw._hidden_size;

  for (int i = 0; i < _tw._n_dec_layer; ++i) {
    _p_d_self_k_cache.push_back(sliding_p);
    sliding_p += _tw._max_greedy_step * _tw._hidden_size;
  }
  for (int i = 0; i < _tw._n_dec_layer; ++i) {
    _p_d_self_v_cache.push_back(sliding_p);
    sliding_p += _tw._max_greedy_step * _tw._hidden_size;
  }

  _p_d_encoder_out_buf = p_int8;
  p_int8 += _tw._max_step * _tw._hidden_size * 2 * _tw._n_dec_layer;
  for (int i = 0; i < _tw._n_dec_layer; ++i) {
    _p_d_encdec_k_cache.push_back(p_int8);
    _p_d_encdec_v_cache.push_back(p_int8 + _tw._max_step * _tw._hidden_size);
    p_int8 += _tw._max_step * _tw._hidden_size * 2;
  }

  _p_d_alive_seq[0] = _tw._start_id;

  printf("decoder buffer size: %.1f MB\n", (float)buffer_size / 1000000);
  printf("decoder buffer init succeed\n");
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
std::string QuantDecoder::check() { return ""; }

void QuantDecoder::run_one_infer() {
#ifdef PROFILE
  profiler.set_start("decoder-project_encoder_output");
#endif
  project_encoder_output();
#ifdef PROFILE
  profiler.set_end("decoder-project_encoder_output");
  profiler.set_start("decoder-run_step");
#endif

  int greedy_max_len = (int)(_tw._greedy_len_a * _seq_len) + _tw._greedy_len_b;
  for (_cur_step = 0; _cur_step < greedy_max_len - 1; ++_cur_step) {
#ifdef DEBUG
    printf("*** run step %d ***\n", _cur_step);
#endif
    if (run_step())
      break;
  }
#ifdef PROFILE
  double total_time = profiler.set_end("decoder-run_step");
  printf("average %lf ms per step, total %d steps\n",
         total_time / (_cur_step + 1), _cur_step + 1);
#endif
}

void QuantDecoder::project_encoder_output() {
  quantize(_p_d_encoder_output, _p_d_encdec_k_cache[0],
           _tw._quant_range / _tw._encoder_output_clip_max,
           _seq_len * _tw._hidden_size);
  gemm_int8_arm(_p_d_encdec_k_cache[0], _tw._p_d_trg_emb_wei_i8[1], nullptr,
                _seq_len, _tw._hidden_size,
                _tw._hidden_size * 2 * _tw._n_dec_layer, 0.f,
                _int32_ffn_out_buf);

#ifdef DEBUG
  print_vec(_p_d_encdec_k_cache[0], "encoder out quantize, head", 10);
  print_vec(_p_d_encdec_k_cache[0] + _seq_len * _tw._hidden_size - 10,
            "encoder out quantize, head", 10);
  print_sum(_p_d_encdec_k_cache[0], "encoder out quantize",
            _seq_len * _tw._hidden_size);
  print_vec(_p_d_encoder_out_buf, "encoder out proj kv, head", 10);
  print_vec(_p_d_encoder_out_buf +
                _seq_len * _tw._hidden_size * 2 * _tw._n_dec_layer - 10,
            "encoder out proj kv, tail", 10);
  print_sum(_p_d_encoder_out_buf, "encoder out proj kv",
            _seq_len * _tw._hidden_size * 2 * _tw._n_dec_layer);
#endif

#pragma omp parallel for
  for (int i = 0; i < _seq_len; ++i) {
    for (int j = 0; j < _tw._n_dec_layer; ++j) {
      float quant_scale =
          _tw._quant_range /
          _tw._dec_clip_max[j * _tw._clip_max_per_dec_layer + 20];
      float dequant_scale = _tw._encoder_output_clip_max *
                            _tw._encode_output_project_kernel_kv_clip_max[j] /
                            (_tw._quant_range * _tw._quant_range);
      for (int k = 0; k < _tw._hidden_size; ++k) {
        _p_d_encdec_k_cache[j][flat_2dim(i, k, _tw._hidden_size)] = float2int8(
            (float)_int32_ffn_out_buf[flat_3dim(i, j, k, _tw._n_dec_layer,
                                                _tw._hidden_size * 2)] *
                    dequant_scale +
                _tw._p_d_trg_emb_wei_fp[3]
                                       [flat_2dim(j, k, _tw._hidden_size * 2)],
            quant_scale);
        _p_d_encdec_v_cache[j][flat_2dim(i, k, _tw._hidden_size)] = float2int8(
            (float)_int32_ffn_out_buf[flat_3dim(i, j, k + _tw._hidden_size,
                                                _tw._n_dec_layer,
                                                _tw._hidden_size * 2)] *
                    dequant_scale +
                _tw._p_d_trg_emb_wei_fp[3][flat_2dim(j, k + _tw._hidden_size,
                                                     _tw._hidden_size * 2)],
            quant_scale);
      }
    }
  }

#ifdef DEBUG
  print_vec(_p_d_encdec_k_cache[0], "reshape enc_out k (L0), head", 10);
  print_sum(_p_d_encdec_k_cache[0], "reshape enc_out k (L0)",
            _seq_len * _tw._hidden_size);
  print_vec(_p_d_encdec_v_cache[_tw._n_dec_layer - 1] +
                _seq_len * _tw._hidden_size - 10,
            "reshape enc_out v (L-1), tail", 10);
  print_sum(_p_d_encdec_v_cache[_tw._n_dec_layer - 1],
            "reshape enc_out v (L-1)", _seq_len * _tw._hidden_size);
#endif
}

bool QuantDecoder::run_step() {
  decoder_embedding_int8(
      _p_d_alive_seq + _cur_step, _tw._p_d_trg_emb_wei_i8[0],
      _tw._p_d_trg_emb_wei_fp[0] + _cur_step * _tw._hidden_size,
      _p_d_cur_step_query, 1, _tw._trg_vocab_size, _tw._hidden_size,
      _tw._trg_emb_clip_max / _tw._quant_range, true);
#ifdef DEBUG
  print_vec(_p_d_cur_step_query, "decoder emb out", 10);
#endif

  layernorm_residual_i8O(
      _p_d_cur_step_query, _int8_ffn_in_buf, _tw._p_d_dec_wei_fp[0],
      _tw._p_d_dec_wei_fp[1], _tw._p_d_dec_wei_fp[3], _tw._hidden_size, 1,
      _tw._quant_range / _tw._dec_clip_max[6], _tw._is_post_ln);

  for (_layer_id = 0; _layer_id < _tw._n_dec_layer; _layer_id++) {
    self_attention();
    encdec_attention();
    ffn_add_norm();
  }

#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_start("step-" + std::to_string(_cur_step) + "-out_proj");
  }
#endif
  gemm_int8_arm(_int8_ffn_in_buf, _tw._p_d_trg_emb_wei_i8[0], _int8_ffn_out_buf,
                1, _tw._hidden_size, _tw._trg_vocab_size,
                _tw._output_ln_clip_max * _tw._trg_emb_clip_max /
                    (_tw._logits_clip_max * _tw._quant_range),
                _int32_ffn_out_buf);
#ifdef DEBUG
  print_vec(_int8_ffn_in_buf, "hidden, head", 10);
  print_vec(_int8_ffn_in_buf + _tw._hidden_size - 10, "hidden, tail", 10);
  print_sum(_int8_ffn_in_buf, "hidden", _tw._hidden_size);
  print_vec(_int8_ffn_out_buf, "logits", 10);
#endif
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_end("step-" + std::to_string(_cur_step) + "-out_proj");
    profiler.set_start("step-" + std::to_string(_cur_step) + "-greedy");
  }
#endif

  float dequant_scale = _tw._logits_clip_max / _tw._quant_range,
        smax = -FLT_MAX, score;
  int max_id = -1;
  for (int i = 0; i < _tw._trg_vocab_size; ++i) {
    score = (float)_int8_ffn_out_buf[i] * dequant_scale +
            _tw._p_d_trg_emb_wei_fp[4][i];
    if (score > smax) {
      smax = score;
      max_id = i;
    }
  }
  if (max_id != _tw._end_id)
    _p_d_alive_seq[_cur_step + 1] = max_id;

#ifdef DEBUG
  printf("step token id: %d\nstep logits: %f\n", max_id, smax);
  print_vec(_p_d_alive_seq, "current output id sequence", _cur_step + 2);
#endif
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_end("step-" + std::to_string(_cur_step) + "-greedy");
  }
#endif
  return max_id == _tw._end_id;
}

void QuantDecoder::self_attention() {
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_start("step-" + std::to_string(_cur_step) + "-self_attention");
  }
#endif
  gemm_int8_arm(
      _int8_ffn_in_buf, _tw._p_d_dec_wei_i8[6 * _layer_id], _int8_ffn_out_buf,
      1, _tw._hidden_size, _tw._hidden_size * 3,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer] *
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 6] /
          (_tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 12] *
           _tw._quant_range),
      _int32_ffn_out_buf);
#ifdef DEBUG
  print_vec(_int8_ffn_in_buf, "self attn input(head): ", 5);
  print_vec(_int8_ffn_in_buf + _tw._hidden_size - 5,
            "self attn input(tail): ", 5);
  print_sum(_int8_ffn_in_buf, "self attn input", _tw._hidden_size);
  print_vec(_int8_ffn_out_buf, "self qkv(head): ", 10);
  print_vec(_int8_ffn_out_buf + _tw._hidden_size * 3 - 10,
            "self qkv(tail): ", 10);
#endif

  float dequant_scale =
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 12] /
      _tw._quant_range;
  float quant_scale =
      _tw._quant_range /
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 18];
  const float *p_bias = _tw._p_d_dec_wei_fp[12 * _layer_id + 2];

#pragma omp parallel for
  for (int i = 0; i < _tw._hidden_size * 3; ++i) {
    _int8_ffn_out_buf[i] = float2int8(
        (float)_int8_ffn_out_buf[i] * dequant_scale + p_bias[i], quant_scale);
  }

  memcpy(_p_d_self_k_cache[_layer_id] + _cur_step * _tw._hidden_size,
         _int8_ffn_out_buf + _tw._hidden_size,
         _tw._hidden_size * sizeof(int8_t));
  memcpy(_p_d_self_v_cache[_layer_id] + _cur_step * _tw._hidden_size,
         _int8_ffn_out_buf + _tw._hidden_size * 2,
         _tw._hidden_size * sizeof(int8_t));

#ifdef DEBUG
  print_vec(_int8_ffn_out_buf, "rearanged q(head): ", 10);
  print_vec(_int8_ffn_out_buf + _tw._hidden_size - 10,
            "rearanged q(tail): ", 10);
  print_sum(_int8_ffn_out_buf, "rearanged q", _tw._hidden_size);
  print_vec(_p_d_self_k_cache[_layer_id], "rearanged k(head): ", 10);
  print_vec(_p_d_self_k_cache[_layer_id] + _cur_step * _tw._hidden_size,
            "rearanged k (cur_step)(head): ", 10);
  print_sum(_p_d_self_k_cache[_layer_id] + _cur_step * _tw._hidden_size,
            "rearanged k", _tw._hidden_size);
  print_vec(_p_d_self_v_cache[_layer_id] + _cur_step * _tw._hidden_size,
            "rearanged v (cur_step)(head): ", 10);
  print_sum(_p_d_self_v_cache[_layer_id] + _cur_step * _tw._hidden_size,
            "rearanged v", _tw._hidden_size);
#endif

  dec_self_attention(
      _int8_ffn_out_buf, _p_d_self_k_cache[_layer_id],
      _p_d_self_v_cache[_layer_id], _int8_ffn_in_buf, _p_d_c, _tw._hidden_size,
      _tw._head_num, _cur_step + 1,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 18] /
          _tw._quant_range,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 18] /
          _tw._quant_range,
      _tw._quant_range /
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 7]);
#ifdef DEBUG
  print_vec(_int8_ffn_in_buf, "self attn ffn in(head): ", 40);
  print_vec(_int8_ffn_in_buf + _tw._hidden_size - 40,
            "self attn ffn in(tail): ", 40);
#endif

  gemm_int8_arm(
      _int8_ffn_in_buf, _tw._p_d_dec_wei_i8[6 * _layer_id + 1],
      _int8_ffn_out_buf, 1, _tw._hidden_size, _tw._hidden_size,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 1] *
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 7] /
          (_tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 13] *
           _tw._quant_range),
      _int32_ffn_out_buf);
#ifdef DEBUG
  print_vec(_int8_ffn_out_buf, "self attn ffn out w/o bias(head): ", 40);
  print_vec(_int8_ffn_out_buf + _tw._hidden_size - 40,
            "self attn ffn out w/o bias(tail): ", 40);
#endif

  float dequant_scale_output =
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 13] /
      _tw._quant_range;
  for (int j = 0; j < _tw._hidden_size; ++j)
    _p_d_cur_step_query[j] +=
        (float)_int8_ffn_out_buf[j] * dequant_scale_output;

  layernorm_residual_i8O(
      _p_d_cur_step_query, _int8_ffn_in_buf,
      _tw._p_d_dec_wei_fp[_layer_id * 12 + 4],
      _tw._p_d_dec_wei_fp[_layer_id * 12 + 5],
      _tw._p_d_dec_wei_fp[_layer_id * 12 + 7], _tw._hidden_size, 1,
      _tw._quant_range /
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 8],
      _tw._is_post_ln);
#ifdef DEBUG
  print_vec(_int8_ffn_in_buf, "encdec attn input(head): ", 5);
  print_vec(_int8_ffn_in_buf + _tw._hidden_size - 5,
            "encdec attn input(tail): ", 5);
  print_sum(_int8_ffn_in_buf, "encdec attn input", _tw._hidden_size);
#endif
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_end("step-" + std::to_string(_cur_step) + "-self_attention");
  }
#endif
}

void QuantDecoder::encdec_attention() {
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_start("step-" + std::to_string(_cur_step) +
                       "-cross_attention");
  }
#endif
  gemm_int8_arm(
      _int8_ffn_in_buf, _tw._p_d_dec_wei_i8[6 * _layer_id + 2],
      _int8_ffn_out_buf, 1, _tw._hidden_size, _tw._hidden_size,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 2] *
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 8] /
          (_tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 14] *
           _tw._quant_range),
      _int32_ffn_out_buf);

  float dequant_scale =
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 14] /
      _tw._quant_range;
  float quant_scale =
      _tw._quant_range /
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 19];
  const float *p_bias = _tw._p_d_dec_wei_fp[12 * _layer_id + 6];

#pragma omp parallel for
  for (int i = 0; i < _tw._hidden_size; ++i) {
    _int8_ffn_out_buf[i] = float2int8(
        (float)_int8_ffn_out_buf[i] * dequant_scale + p_bias[i], quant_scale);
  }

#ifdef DEBUG
  print_vec(_int8_ffn_out_buf, "rearanged q(head): ", 5);
  print_vec(_int8_ffn_out_buf + _tw._hidden_size - 5, "rearanged q(tail): ", 5);
#endif

  dec_self_attention(
      _int8_ffn_out_buf, _p_d_encdec_k_cache[_layer_id],
      _p_d_encdec_v_cache[_layer_id], _int8_ffn_in_buf, _p_d_c,
      _tw._hidden_size, _tw._head_num, _seq_len,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 19] /
          _tw._quant_range,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 20] /
          _tw._quant_range,
      _tw._quant_range /
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 9]);
#ifdef DEBUG
  print_vec(_int8_ffn_in_buf, "encdec attn ffn in(head): ", 5);
  print_vec(_int8_ffn_in_buf + _tw._hidden_size - 5,
            "encdec attn ffn in(tail): ", 5);
#endif

  gemm_int8_arm(
      _int8_ffn_in_buf, _tw._p_d_dec_wei_i8[6 * _layer_id + 3],
      _int8_ffn_out_buf, 1, _tw._hidden_size, _tw._hidden_size,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 3] *
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 9] /
          (_tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 15] *
           _tw._quant_range),
      _int32_ffn_out_buf);
#ifdef DEBUG
  print_vec(_int8_ffn_out_buf, "encdec attn ffn out w/o bias(head): ", 5);
  print_vec(_int8_ffn_out_buf + _tw._hidden_size - 5,
            "encdec attn ffn out w/o bias(tail): ", 5);
#endif

  float dequant_scale_output =
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 15] /
      _tw._quant_range;
  for (int j = 0; j < _tw._hidden_size; ++j)
    _p_d_cur_step_query[j] +=
        (float)_int8_ffn_out_buf[j] * dequant_scale_output;

  layernorm_residual_i8O(
      _p_d_cur_step_query, _int8_ffn_in_buf,
      _tw._p_d_dec_wei_fp[_layer_id * 12 + 8],
      _tw._p_d_dec_wei_fp[_layer_id * 12 + 9],
      _tw._p_d_dec_wei_fp[_layer_id * 12 + 11], _tw._hidden_size, 1,
      _tw._quant_range /
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 10],
      _tw._is_post_ln);
#ifdef DEBUG
  print_vec(_p_d_cur_step_query, "encdec attn ffn out(head): ", 5);
  print_vec(_p_d_cur_step_query + _tw._hidden_size - 5,
            "encdec attn ffn out(tail): ", 5);
  print_vec(_int8_ffn_in_buf, "ffn input (head): ", 5);
  print_vec(_int8_ffn_in_buf + _tw._hidden_size - 5, "ffn input(tail): ", 5);
  print_sum(_int8_ffn_in_buf, "ffn input", _tw._hidden_size);
#endif
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_end("step-" + std::to_string(_cur_step) + "-cross_attention");
  }
#endif
}

void QuantDecoder::ffn_add_norm() {
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_start("step-" + std::to_string(_cur_step) + "-ffn");
  }
#endif
  gemm_int8_arm(
      _int8_ffn_in_buf, _tw._p_d_dec_wei_i8[6 * _layer_id + 4],
      _int8_ffn_out_buf, 1, _tw._hidden_size, _tw._inner_size,
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 4] *
          _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 10] /
          (_tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 16] *
           _tw._quant_range),
      _int32_ffn_out_buf);

  if (_tw._use_gelu) {
    bias_gelu_i8IO(
        _int8_ffn_out_buf, _tw._p_d_dec_wei_fp[12 * _layer_id + 10], 1,
        _tw._inner_size,
        _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 16] /
            _tw._quant_range,
        _tw._quant_range /
            _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 11]);
  } else {
    bias_relu_i8IO(
        _int8_ffn_out_buf, _tw._p_d_dec_wei_fp[12 * _layer_id + 10], 1,
        _tw._inner_size,
        _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 16] /
            _tw._quant_range,
        _tw._quant_range /
            _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 11],
        _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 11]);
  }

  gemm_int8_arm(_int8_ffn_out_buf, _tw._p_d_dec_wei_i8[6 * _layer_id + 5],
                nullptr, 1, _tw._inner_size, _tw._hidden_size, 0.f,
                _int32_ffn_out_buf);
#ifdef DEBUG
  print_vec(_int32_ffn_out_buf, "ffn2 kernel out(head): ", 10);
  print_vec(_int32_ffn_out_buf + _tw._hidden_size - 10,
            "ffn2 kernel out(tail): ", 10);
#endif

  const float *scale_ptr, *bias_ptr, *res_bias_ptr;
  float clip_max, dequant_scale;
  dequant_scale =
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 5] *
      _tw._dec_clip_max[_layer_id * _tw._clip_max_per_dec_layer + 11] /
      (_tw._quant_range * _tw._quant_range);
  if (_layer_id == _tw._n_dec_layer - 1) {
    scale_ptr = _tw._p_d_trg_emb_wei_fp[1];
    bias_ptr = _tw._p_d_trg_emb_wei_fp[2];
    res_bias_ptr = nullptr;
    clip_max = _tw._output_ln_clip_max;
  } else {
    scale_ptr = _tw._p_d_dec_wei_fp[12 * (_layer_id + 1)];
    bias_ptr = _tw._p_d_dec_wei_fp[12 * (_layer_id + 1) + 1];
    res_bias_ptr = _tw._p_d_dec_wei_fp[12 * (_layer_id + 1) + 3];
    clip_max =
        _tw._dec_clip_max[(_layer_id + 1) * _tw._clip_max_per_dec_layer + 6];
  }

  residual_bias_ln_i32I(_int32_ffn_out_buf, res_bias_ptr, scale_ptr, bias_ptr,
                        _p_d_cur_step_query, _int8_ffn_in_buf, _tw._hidden_size,
                        1, dequant_scale, _tw._quant_range / clip_max,
                        _tw._is_post_ln, nullptr);
#ifdef DEBUG
  print_vec(_p_d_cur_step_query, "ffn ln(head): ", 5);
  print_vec(_p_d_cur_step_query + _tw._hidden_size - 5, "ffn ln(tail): ", 5);
#endif
#ifdef PROFILE
  if (_cur_step % 20 == 0) {
    profiler.set_end("step-" + std::to_string(_cur_step) + "-ffn");
  }
#endif
}

} // namespace lightseq
