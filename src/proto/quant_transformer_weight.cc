#include "quant_transformer_weight.h"

namespace lightseq {
/**
Read model config stored in custom proto file.
*/
void QuantTransformerWeight::proto_get_model_config(
    const QuantTransformer &transformer, int max_seq_len, bool only_decoder) {
  _hidden_size = transformer.trg_embedding().norm_scale_size();

  _max_position =
      transformer.trg_embedding().position_embedding_size() / _hidden_size;

  if (max_seq_len > _max_position) {
    _max_step = _max_position;
    throw std::runtime_error("max_seq_len should <= " +
                             std::to_string(_max_position));
  }
  _max_step = max_seq_len ? max_seq_len : _max_position;

  _inner_size =
      transformer.decoder_stack()[0].ffn_first_kernel().size() / _hidden_size;

  _share_all_embedding = false;
  _trg_vocab_size =
      transformer.trg_embedding().token_embedding().size() / _hidden_size;
  if (!only_decoder) {
    if (transformer.src_embedding().token_embedding().size() == 0)
      _share_all_embedding = true;
    if (_share_all_embedding) {
      _src_vocab_size = _trg_vocab_size;
    } else {
      _src_vocab_size =
          transformer.src_embedding().token_embedding().size() / _hidden_size;
    }
  }

  if (!only_decoder) {
    _n_enc_layer = transformer.encoder_stack_size();
  }
  _n_dec_layer = transformer.decoder_stack_size();

  _weight_i8_per_enc_layer = 4;
  _weight_fp_per_enc_layer = 8;
  _clip_max_per_enc_layer = 12;
  _weight_i8_per_dec_layer = 6;
  _weight_fp_per_dec_layer = 12;
  _clip_max_per_dec_layer = 21;

  _head_num = transformer.model_conf().head_num();
  if (_hidden_size % _head_num != 0) {
    throw std::runtime_error("Wrong head_num: hidden_size " +
                             std::to_string(_hidden_size) + " % head_num " +
                             std::to_string(_head_num) + " != 0.");
  }
  _dim_per_head = _hidden_size / _head_num;
  _extra_decode_length = transformer.model_conf().extra_decode_length();
  _length_penalty = transformer.model_conf().length_penalty();
  _greedy_len_a = transformer.model_conf().greedy_len_a();
  _greedy_len_b = transformer.model_conf().greedy_len_b();
  _max_greedy_step = (int)(_max_step * _greedy_len_a) + _greedy_len_b;

  _padding_id = transformer.model_conf().src_padding_id();
  _start_id = transformer.model_conf().trg_start_id();
  _end_id = transformer.model_conf().trg_end_id();
  if (_end_id == 0) {
    _end_id = _trg_vocab_size - 1;
  }
  _is_post_ln = transformer.model_conf().is_post_ln();
  _no_scale_embedding = transformer.model_conf().no_scale_embedding();
  _use_gelu = transformer.model_conf().use_gelu();
}

/**
Load the weights of embedding layer into GPU memory.
Compared with the encoder, the decoder has more
  encoder output project weights, encoder output project bias,
  logits bias. So we need an "source" parameter to
  distinguish between encoder and decoder
*/
std::string
QuantTransformerWeight::proto_parse_emb_wei(const QuantEmbeddingLayer &layer,
                                            std::string source) {
  int vocab_size = (source == "src") ? _src_vocab_size : _trg_vocab_size;
  std::vector<int8_t> &d_emb_wei_i8 =
      source == "src" ? _d_src_emb_wei_i8 : _d_trg_emb_wei_i8;
  std::vector<float> &d_emb_wei_fp =
      source == "src" ? _d_src_emb_wei_fp : _d_trg_emb_wei_fp;

  if (source == "src") {
    _src_emb_clip_max = layer.emb_clip_max();
    _encoder_output_clip_max = layer.encoder_output_clip_max();
  } else {
    _trg_emb_clip_max = layer.emb_clip_max();
    _output_ln_clip_max = layer.output_ln_clip_max();
    _logits_clip_max = layer.logits_clip_max();
    for (float v : layer.encode_output_project_kernel_kv_clip_max()) {
      _encode_output_project_kernel_kv_clip_max.push_back(v);
    }
    // _encode_output_project_out_clip_max =
    // layer.encode_output_project_out_clip_max();
  }

  std::vector<int> offset, offset_i8;
  int idx = 0, idx_i8 = 0;

  if (source == "trg") {
    _d_trg_emb_wei_i8.reserve((vocab_size + _hidden_size * 2 * _n_dec_layer) *
                              _hidden_size);
    _d_trg_emb_wei_fp.reserve(
        (_max_position + 2 * _n_dec_layer + 2) * _hidden_size + vocab_size);
  } else {
    size_t src_emb_size_i8 = 0;
    size_t src_emb_size_fp = _hidden_size * 2;
    if (!_share_all_embedding) {
      src_emb_size_i8 += vocab_size * _hidden_size;
      src_emb_size_fp += _max_position * _hidden_size;
    }
    _d_src_emb_wei_i8.reserve(src_emb_size_i8);
    _d_src_emb_wei_fp.reserve(src_emb_size_fp);
  }

  if (source == "src" && _share_all_embedding) {
    _src_emb_clip_max = _trg_emb_clip_max;
    _p_d_src_emb_wei_i8.push_back(_p_d_trg_emb_wei_i8[0]);
    _p_d_src_emb_wei_fp.push_back(_p_d_trg_emb_wei_fp[0]);
  } else {
    offset_i8.push_back(idx_i8);
    if (layer.token_embedding().size() != vocab_size * _hidden_size)
      return "Wrong token_embedding_size !";
    for (unsigned char ele : layer.token_embedding())
      d_emb_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (layer.position_embedding_size() != _max_position * _hidden_size)
      return "Wrong position_embedding_size !";
    for (float ele : layer.position_embedding())
      d_emb_wei_fp[idx++] = ele;
  }

  offset.push_back(idx);
  if (layer.norm_scale_size() != _hidden_size)
    return "Wrong norm_scale_size !";
  for (float ele : layer.norm_scale())
    d_emb_wei_fp[idx++] = ele;

  offset.push_back(idx);
  if (layer.norm_bias_size() != _hidden_size)
    return "Wrong norm_bias_size !";
  for (float ele : layer.norm_bias())
    d_emb_wei_fp[idx++] = ele;

  if (source == "src") {
    for (int e : offset)
      _p_d_src_emb_wei_fp.push_back(_d_src_emb_wei_fp.data() + e);
    for (int e : offset_i8)
      _p_d_src_emb_wei_i8.push_back(_d_src_emb_wei_i8.data() + e);
  } else {
    // for trg, encdec_kv_kernel, encdec_kv_bias, logit_bias

    offset_i8.push_back(idx_i8);

    int per_layer_ele_num = _hidden_size * _hidden_size * 2;
    if (layer.encode_output_project_kernel_kv().size() !=
        per_layer_ele_num * _n_dec_layer)
      return "Wrong encode_output_project_kernel_kv_size !";
    for (int layer_id = 0; layer_id < _n_dec_layer; ++layer_id) {
      for (int ele_id = 0; ele_id < per_layer_ele_num; ++ele_id) {
        unsigned char ele =
            layer.encode_output_project_kernel_kv()[layer_id *
                                                        per_layer_ele_num +
                                                    ele_id];
        _d_trg_emb_wei_i8[idx_i8++] = (int8_t)(ele - 127);
      }
    }

    offset.push_back(idx);
    if (layer.encode_output_project_bias_kv_size() !=
        _hidden_size * 2 * _n_dec_layer)
      return "Wrong encode_output_project_bias_kv_size !";
    for (float ele : layer.encode_output_project_bias_kv())
      _d_trg_emb_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (layer.shared_bias_size() != vocab_size)
      return "Wrong shared_bias_size !";
    for (float ele : layer.shared_bias())
      _d_trg_emb_wei_fp[idx++] = ele;

    for (int e : offset) {
      _p_d_trg_emb_wei_fp.push_back(_d_trg_emb_wei_fp.data() + e);
    }
    for (int e : offset_i8) {
      _p_d_trg_emb_wei_i8.push_back(_d_trg_emb_wei_i8.data() + e);
    }
  } // trg

  std::cout << "Finish loading " << source << "_emb_wei" << std::endl;
  return "";
}

/**
Load the weights of encoder into GPU memory.
*/
std::string QuantTransformerWeight::proto_parse_enc_wei(
    const QuantTransformer &transformer) {
  std::vector<int> offset, offset_i8;
  int idx = 0, idx_i8 = 0;

  size_t size_fp = _hidden_size * 9 + _inner_size;
  size_t size_i8 =
      _hidden_size * _hidden_size * 4 + _hidden_size * _inner_size * 2;
  _d_enc_wei_fp.reserve(size_fp * _n_enc_layer);
  _d_enc_wei_i8.reserve(size_i8 * _n_enc_layer);

  for (auto enc_layer : transformer.encoder_stack()) {
    offset.push_back(idx);
    if (enc_layer.multihead_norm_scale_size() != _hidden_size)
      return "Wrong multihead_norm_scale_size !";
    for (float ele : enc_layer.multihead_norm_scale())
      _d_enc_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (enc_layer.multihead_norm_bias_size() != _hidden_size)
      return "Wrong multihead_norm_bias_size !";
    for (float ele : enc_layer.multihead_norm_bias())
      _d_enc_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (enc_layer.multihead_project_kernel_qkv().size() !=
        _hidden_size * _hidden_size * 3)
      return "Wrong multihead_project_kernel_qkv_size !";
    for (unsigned char ele : enc_layer.multihead_project_kernel_qkv())
      _d_enc_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_qkv_size() != _hidden_size * 3)
      return "Wrong multihead_project_bias_qkv_size !";
    for (float ele : enc_layer.multihead_project_bias_qkv())
      _d_enc_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (enc_layer.multihead_project_kernel_output().size() !=
        _hidden_size * _hidden_size)
      return "Wrong multihead_project_kernel_output_size !";
    for (unsigned char ele : enc_layer.multihead_project_kernel_output())
      _d_enc_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_output_size() != _hidden_size)
      return "Wrong multihead_project_bias_output_size !";
    for (float ele : enc_layer.multihead_project_bias_output())
      _d_enc_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_scale_size() != _hidden_size)
      return "Wrong ffn_norm_scale_size !";
    for (float ele : enc_layer.ffn_norm_scale())
      _d_enc_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_bias_size() != _hidden_size)
      return "Wrong ffn_norm_bias_size !";
    for (float ele : enc_layer.ffn_norm_bias())
      _d_enc_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (enc_layer.ffn_first_kernel().size() != _hidden_size * _inner_size)
      return "Wrong ffn_first_kernel_size !";
    for (unsigned char ele : enc_layer.ffn_first_kernel())
      _d_enc_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (enc_layer.ffn_first_bias_size() != _inner_size)
      return "Wrong ffn_first_bias_size !";
    for (float ele : enc_layer.ffn_first_bias())
      _d_enc_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (enc_layer.ffn_second_kernel().size() != _hidden_size * _inner_size)
      return "Wrong ffn_second_kernel_size !";
    for (unsigned char ele : enc_layer.ffn_second_kernel())
      _d_enc_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (enc_layer.ffn_second_bias_size() != _hidden_size)
      return "Wrong ffn_second_bias_size !";
    for (float ele : enc_layer.ffn_second_bias())
      _d_enc_wei_fp[idx++] = ele;

    _enc_clip_max.push_back(enc_layer.multihead_project_kernel_qkv_clip_max());
    _enc_clip_max.push_back(
        enc_layer.multihead_project_kernel_output_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_first_kernel_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_second_kernel_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_ln_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_project_output_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_ln_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_first_act_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_qkv_dense_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_output_dense_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_first_output_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_second_output_clip_max());

  } // for

  for (int e : offset)
    _p_d_enc_wei_fp.push_back((_d_enc_wei_fp.data()) + e);
  for (int e : offset_i8)
    _p_d_enc_wei_i8.push_back((_d_enc_wei_i8.data()) + e);
  std::cout << "Finish loading enc_wei" << std::endl;

  return "";
}

/**
Load the weights of decoder into GPU memory.
*/
std::string QuantTransformerWeight::proto_parse_dec_wei(
    const QuantTransformer &transformer) {
  std::vector<int> offset, offset_i8;
  int idx = 0, idx_i8 = 0;

  size_t size_fp = _hidden_size * 13 + _inner_size;
  size_t size_i8 =
      _hidden_size * _hidden_size * 6 + _hidden_size * _inner_size * 2;
  _d_dec_wei_fp.reserve(size_fp * _n_dec_layer);
  _d_dec_wei_i8.reserve(size_i8 * _n_dec_layer);

  for (auto dec_layer : transformer.decoder_stack()) {
    offset.push_back(idx);
    if (dec_layer.self_norm_scale_size() != _hidden_size)
      return "Wrong self_norm_scale size !";
    for (float ele : dec_layer.self_norm_scale())
      _d_dec_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (dec_layer.self_norm_bias_size() != _hidden_size)
      return "Wrong self_norm_bias_size !";
    for (float ele : dec_layer.self_norm_bias())
      _d_dec_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (dec_layer.self_project_kernel_qkv().size() !=
        _hidden_size * _hidden_size * 3)
      return "Wrong self_project_kernel_qkv size !";
    for (unsigned char ele : dec_layer.self_project_kernel_qkv())
      _d_dec_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (dec_layer.self_project_bias_qkv_size() != _hidden_size * 3)
      return "Wrong self_project_bias_qkv size !";
    for (float ele : dec_layer.self_project_bias_qkv())
      _d_dec_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (dec_layer.self_project_kernel_output().size() !=
        _hidden_size * _hidden_size)
      return "Wrong self_project_kernel_output size !";
    for (unsigned char ele : dec_layer.self_project_kernel_output())
      _d_dec_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (dec_layer.self_project_bias_output_size() != _hidden_size)
      return "Wrong self_project_bias_output size !";
    for (float ele : dec_layer.self_project_bias_output())
      _d_dec_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (dec_layer.encdec_norm_scale_size() != _hidden_size)
      return "Wrong encdec_norm_scale size !";
    for (float ele : dec_layer.encdec_norm_scale())
      _d_dec_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (dec_layer.encdec_norm_bias_size() != _hidden_size)
      return "Wrong encdec_norm_bias_size !";
    for (float ele : dec_layer.encdec_norm_bias())
      _d_dec_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (dec_layer.encdec_project_kernel_q().size() !=
        _hidden_size * _hidden_size)
      return "Wrong encdec_project_kernel_q size !";
    for (unsigned char ele : dec_layer.encdec_project_kernel_q())
      _d_dec_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (dec_layer.encdec_project_bias_q_size() != _hidden_size)
      return "Wrong encdec_project_bias_q size !";
    for (float ele : dec_layer.encdec_project_bias_q())
      _d_dec_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (dec_layer.encdec_project_kernel_output().size() !=
        _hidden_size * _hidden_size)
      return "Wrong encdec_project_kernel_output size !";
    for (unsigned char ele : dec_layer.encdec_project_kernel_output())
      _d_dec_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (dec_layer.encdec_project_bias_output_size() != _hidden_size)
      return "Wrong encdec_project_bias_output size !";
    for (float ele : dec_layer.encdec_project_bias_output())
      _d_dec_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (dec_layer.ffn_norm_scale_size() != _hidden_size)
      return "Wrong ffn_norm_scale_size !";
    for (float ele : dec_layer.ffn_norm_scale())
      _d_dec_wei_fp[idx++] = ele;

    offset.push_back(idx);
    if (dec_layer.ffn_norm_bias_size() != _hidden_size)
      return "Wrong ffn_norm_bias_size !";
    for (float ele : dec_layer.ffn_norm_bias())
      _d_dec_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (dec_layer.ffn_first_kernel().size() != _hidden_size * _inner_size)
      return "Wrong ffn_first_kernel_size !";
    for (unsigned char ele : dec_layer.ffn_first_kernel())
      _d_dec_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (dec_layer.ffn_first_bias_size() != _inner_size)
      return "Wrong ffn_first_bias_size !";
    for (float ele : dec_layer.ffn_first_bias())
      _d_dec_wei_fp[idx++] = ele;

    offset_i8.push_back(idx_i8);
    if (dec_layer.ffn_second_kernel().size() != _hidden_size * _inner_size)
      return "Wrong ffn_second_kernel_size !";
    for (unsigned char ele : dec_layer.ffn_second_kernel())
      _d_dec_wei_i8[idx_i8++] = (int8_t)(ele - 127);

    offset.push_back(idx);
    if (dec_layer.ffn_second_bias_size() != _hidden_size)
      return "Wrong ffn_second_bias_size !";
    for (float ele : dec_layer.ffn_second_bias())
      _d_dec_wei_fp[idx++] = ele;

    _dec_clip_max.push_back(dec_layer.self_project_kernel_qkv_clip_max());
    _dec_clip_max.push_back(dec_layer.self_project_kernel_output_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_project_kernel_q_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_project_kernel_output_clip_max());
    _dec_clip_max.push_back(dec_layer.ffn_first_kernel_clip_max());
    _dec_clip_max.push_back(dec_layer.ffn_second_kernel_clip_max());
    _dec_clip_max.push_back(dec_layer.self_ln_clip_max());
    _dec_clip_max.push_back(dec_layer.self_project_output_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_ln_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_project_output_clip_max());
    _dec_clip_max.push_back(dec_layer.ffn_ln_clip_max());
    _dec_clip_max.push_back(dec_layer.ffn_first_act_clip_max());
    _dec_clip_max.push_back(dec_layer.self_qkv_dense_clip_max());
    _dec_clip_max.push_back(dec_layer.self_output_dense_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_q_dense_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_output_dense_clip_max());
    _dec_clip_max.push_back(dec_layer.ffn_first_output_clip_max());
    _dec_clip_max.push_back(dec_layer.ffn_second_output_clip_max());
    _dec_clip_max.push_back(dec_layer.self_qkv_bias_out_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_q_bias_out_clip_max());
    _dec_clip_max.push_back(dec_layer.encdec_kv_bias_out_clip_max());

  } // for

  for (int e : offset)
    _p_d_dec_wei_fp.push_back((_d_dec_wei_fp.data()) + e);
  for (int e : offset_i8)
    _p_d_dec_wei_i8.push_back((_d_dec_wei_i8.data()) + e);
  std::cout << "Finish loading dec_wei" << std::endl;

  return "";
}

/**
Load the proto file into CPU memory and parse it.
*/
std::string QuantTransformerWeight::initializing(std::string weight_path,
                                                 int max_seq_len,
                                                 bool only_decoder) {

  std::cout << "Parsing protobuf: " << weight_path << std::endl;
  QuantTransformer transformer;
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::fstream raw_input(weight_path, std::ios::in | std::ios::binary);
  if (!transformer.ParseFromIstream(&raw_input)) {
    return "Parse weights from [" + weight_path + "] failed.";
  }
  proto_get_model_config(transformer, max_seq_len, only_decoder);

  std::string res;
  res = proto_parse_emb_wei(transformer.trg_embedding(), "trg");
  if (!res.empty())
    return res;

  if (!only_decoder) {
    res = proto_parse_emb_wei(transformer.src_embedding(), "src");
    if (!res.empty())
      return res;
  }

  if (!only_decoder) {
    res = proto_parse_enc_wei(transformer);
    if (!res.empty())
      return res;
  }

  res = proto_parse_dec_wei(transformer);
  if (!res.empty())
    return res;

  std::cout << "Finish loading model weights" << std::endl;
  // Optional:  Delete all global objects allocated by libprotobuf.
  // google::protobuf::ShutdownProtobufLibrary();
  return "";
}

} // namespace lightseq
