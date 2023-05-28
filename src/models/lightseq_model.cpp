#ifndef LIGHTSEQ_MODEL_CPP_
#define LIGHTSEQ_MODEL_CPP_
#include "lightseq_model.hpp"
#include "quant_transformer.h"
#include <memory>
#include <string>

namespace lightseq {

std::shared_ptr<LSModel>
LSModelFactory::CreateLSModel(const std::string weight_path, int max_seq_len,
                              const std::string resource_path,
                              const std::string name) {
  return std::shared_ptr<LSModel>(
      new QuantTransformer(weight_path, max_seq_len));
}

} // namespace lightseq
#endif
