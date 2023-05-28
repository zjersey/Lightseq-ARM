#pragma once

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "quant_transformer.pb.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace lightseq {
/*
Load the model weights which stored in custom proto file into GPU memory.
*/
class QuantTransformerWeight {
private:
  // parsing function for protobuffer
  void proto_get_model_config(const QuantTransformer &transformer,
                              int max_seq_len, bool only_decoder = false);
  std::string proto_parse_emb_wei(const QuantEmbeddingLayer &layer,
                                  std::string source);
  std::string proto_parse_enc_wei(const QuantTransformer &transformer);
  std::string proto_parse_dec_wei(const QuantTransformer &transformer);

  // store the weights on cpu memory
  std::vector<int8_t> _d_enc_wei_i8;
  std::vector<int8_t> _d_dec_wei_i8;
  std::vector<int8_t> _d_trg_emb_wei_i8;
  std::vector<int8_t> _d_src_emb_wei_i8;
  std::vector<float> _d_enc_wei_fp;
  std::vector<float> _d_dec_wei_fp;
  std::vector<float> _d_trg_emb_wei_fp;
  std::vector<float> _d_src_emb_wei_fp;

public:
  std::string initializing(std::string proto_path, int max_seq_len,
                           bool only_decoder = false);

  int _hidden_size;
  int _inner_size;
  int _max_step;
  int _max_position;
  int _max_greedy_step;
  int _src_vocab_size;
  int _trg_vocab_size;
  int _n_enc_layer; // number of encoder layer
  int _n_dec_layer; // number of decoder layer
  int _n_enc_weight_layer;
  int _n_dec_weight_layer;
  int _dim_per_head;
  int _weight_i8_per_enc_layer; // 4
  int _weight_i8_per_dec_layer; // 6
  int _weight_fp_per_enc_layer; // 8
  int _weight_fp_per_dec_layer; // 12
  int _clip_max_per_enc_layer;  // 12
  int _clip_max_per_dec_layer;  // 19

  int _head_num;
  int _extra_decode_length;
  float _length_penalty;
  float _greedy_len_a;
  int _greedy_len_b;
  int _padding_id; // for src
  int _start_id;   // for trg
  int _end_id;
  bool _is_post_ln;
  bool _no_scale_embedding;
  bool _use_gelu;
  bool _share_all_embedding;

  const float _quant_range = 127;

  // store the weights pointer
  std::vector<const int8_t *> _p_d_src_emb_wei_i8; // size: 1
  std::vector<const int8_t *> _p_d_trg_emb_wei_i8; // size: 1
  std::vector<const int8_t *> _p_d_enc_wei_i8;     // size: 4 * enc_layer_num
  std::vector<const int8_t *> _p_d_dec_wei_i8;     // size: 6 * enc_layer_num
  std::vector<const float *> _p_d_src_emb_wei_fp;  // size: 3
  std::vector<const float *> _p_d_trg_emb_wei_fp;  // size: 3
  std::vector<const float *> _p_d_enc_wei_fp;      // size: 8 * enc_layer_num
  std::vector<const float *> _p_d_dec_wei_fp;      // size: 12 * dec_layer_num

  // store the clip_max of weights and activations
  float _src_emb_clip_max;
  float _trg_emb_clip_max;
  float _output_ln_clip_max;
  float _logits_clip_max;
  float _encoder_output_clip_max;
  std::vector<float> _encode_output_project_kernel_kv_clip_max;
  float _encode_output_project_out_clip_max;
  std::vector<float> _enc_clip_max; // size: 12 * enc_layer_num
  std::vector<float> _dec_clip_max; // size: 19 * dec_layer_num

  void print_model_config() {
    std::cout << "***model config***" << std::endl;
    std::cout << "encoder layers: " << _n_enc_layer << std::endl;
    std::cout << "decoder layers: " << _n_dec_layer << std::endl;
    std::cout << "hidden size: " << _hidden_size << std::endl;
    std::cout << "inner size: " << _inner_size << std::endl;
    std::cout << "head number: " << _head_num << std::endl;
    std::cout << "dim per head: " << _dim_per_head << std::endl;
    std::cout << "src vocab size: " << _src_vocab_size << std::endl;
    std::cout << "trg vocab size: " << _trg_vocab_size << std::endl;
    std::cout << "is_post_ln: " << _is_post_ln << std::endl;
    std::cout << "no_scale_embedding: " << _no_scale_embedding << std::endl;
    std::cout << "share_all_embedding: " << _share_all_embedding << std::endl;
    std::cout << "use_gelu: " << _use_gelu << std::endl;
    std::cout << "start_id: " << _start_id << std::endl;
    std::cout << "end_id: " << _end_id << std::endl;
    std::cout << "padding_id: " << _padding_id << std::endl;
    std::cout << std::endl;
    std::cout << "***generator config***" << std::endl;
    std::cout << "max step: " << _max_step << std::endl;
    std::cout << "extra decode length(max decode length - src input length): "
              << _extra_decode_length << std::endl;
    std::cout << "length penalty: " << _length_penalty << std::endl;
  }
};

} // namespace lightseq
