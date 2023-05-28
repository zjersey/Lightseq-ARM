#include "../proto/quant_transformer_weight.h"
#include "lightseq_model.hpp"
#include "quant_decoder.h"
#include "quant_encoder.h"

namespace lightseq {

class QuantTransformer : public LSModel {
private:
  std::shared_ptr<QuantEncoder> encoder_;
  std::shared_ptr<QuantDecoder> decoder_;

  float *d_encoder_output_;
  QuantTransformerWeight tw_;

public:
  QuantTransformer(const std::string weight_path, int max_seq_len);
  ~QuantTransformer();

  void Infer();
  void set_input_ptr(int index, void *input_ptr, int seq_len);
  void set_output_ptr(int index, void *input_ptr);
  const void *get_output_ptr(int index);
  int get_output_seq_len();
};

} // namespace lightseq
