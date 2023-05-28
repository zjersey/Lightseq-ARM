#include "quant_transformer.h"
#include "../utils/profiler.h"

namespace lightseq {

QuantTransformer::QuantTransformer(const std::string weight_path,
                                   int max_seq_len = 0) {

  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path, max_seq_len);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  tw_.print_model_config();

  d_encoder_output_ =
      (float *)malloc(tw_._max_step * tw_._hidden_size * sizeof(float));

  encoder_ = std::make_shared<QuantEncoder>(tw_, d_encoder_output_);
  res = encoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  decoder_ = std::make_shared<QuantDecoder>(tw_, d_encoder_output_);
  res = decoder_->check();
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  encoder_->init_buffer();
  decoder_->init_buffer();
}

QuantTransformer::~QuantTransformer() { free(d_encoder_output_); }

void QuantTransformer::Infer() {
#ifdef PROFILE
  Profiler profiler;
  profiler.set_start("e2e-encoder");
#endif
  encoder_->run_one_infer();

#ifdef PROFILE
  profiler.set_end("e2e-encoder");
  profiler.set_start("e2e-decoder");
#endif
  decoder_->run_one_infer();

#ifdef PROFILE
  profiler.set_end("e2e-decoder");
#endif
}

void QuantTransformer::set_input_ptr(int index, void *input_ptr, int seq_len) {
  encoder_->_seq_len = seq_len;
  decoder_->_seq_len = seq_len;
  switch (index) {
  case 0:
    encoder_->_p_d_token_id = static_cast<int *>(input_ptr);
    break;

  default:
    throw std::runtime_error("invalid input index");
    break;
  }
}

void QuantTransformer::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
  case 0:
    decoder_->_p_d_result = static_cast<int *>(output_ptr);
    break;

  default:
    throw std::runtime_error("invalid input index");
    break;
  }
}

const void *QuantTransformer::get_output_ptr(int index) {
  switch (index) {
  case 0:
    return static_cast<void *>(decoder_->_p_d_alive_seq);
    break;

  default:
    throw std::runtime_error("invalid output index");
    break;
  }
}

int QuantTransformer::get_output_seq_len() { return decoder_->_cur_step + 1; };

} // namespace lightseq
