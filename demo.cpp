#include "src/models/lightseq_model.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  const char *weight_file = argc > 1 ? argv[1] : "";
  
  std::shared_ptr<lightseq::LSModel> model =
      lightseq::LSModelFactory().CreateLSModel(std::string(weight_file), 64);

  std::vector<int> input_ids(
      {3438, 1628, 4});

  int seq_len = input_ids.size();

  model->set_input_ptr(0, (void *)input_ids.data(), seq_len);

  model->Infer();

  const int *output_ptr = (const int *)(model->get_output_ptr(0));
  int out_seq_len = model->get_output_seq_len();
  printf("model infer out (seqlen: %d):\t", out_seq_len);
  for (int i = 0; i < out_seq_len; ++i) {
    std::cout << output_ptr[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
