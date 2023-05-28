#ifndef LIGHTSEQ_MODEL_HPP_
#define LIGHTSEQ_MODEL_HPP_
#include <memory>
#include <string>

namespace lightseq {

class LSModel {
public:
  virtual ~LSModel(){};

  virtual void Infer() = 0;
  virtual void set_input_ptr(int index, void *input_ptr, int seq_len) = 0;
  virtual void set_output_ptr(int index, void *output_ptr) = 0;
  virtual const void *get_output_ptr(int index) = 0;
  virtual int get_output_seq_len() = 0;
};

class LSModelFactory {
public:
  static std::shared_ptr<LSModel>
  CreateLSModel(const std::string weight_path, int max_seq_len,
                const std::string resource_path = "",
                const std::string name = "");
};

} // namespace lightseq
#endif
