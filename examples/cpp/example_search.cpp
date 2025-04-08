#include "../../hnswlib/hnswlib.h"
#include <chrono>

template <typename T = float>
void normalize_vector(T *l2_mag, T *vec_norm, int index) {
  T sqrt_norm = std::sqrt(*l2_mag);
  if (sqrt_norm == 0) {
    throw std::invalid_argument("ERROR: norm value is 0.");
  }
  vec_norm[index] = sqrt_norm;
  *l2_mag = 0;
}

int main() {
  int dim = 1024; // Dimension of the elements
  int max_elements = 10000;
  //  int max_elements = 3000;

  int M = 32; // Tightly connected with internal dimensionality of the data
              // strongly affects the memory consumption
  int ef_construction = 200; // Controls index search speed/build speed tradeoff

  // Initing index
  hnswlib::L2Space space(dim);
  //  hnswlib::InnerProductSpace space(dim);

  hnswlib::HierarchicalNSW<float> *alg_hnsw =
      new hnswlib::HierarchicalNSW<float>(&space, max_elements, M,
                                          ef_construction);

  // std::cout << "max_elements: " << max_elements << " M: " << M
  //           << " ef_construction: " << ef_construction << " dim: " << dim
  //           << " ef: " << 100 << " l2space " << std::endl;

  std::cout << "max_elements: " << max_elements << " M: " << M
            << " ef_construction: " << ef_construction << " dim: " << dim
            << " ef: " << 100 << " L2 " << std::endl;

  //  InnerProductSpace

  /*
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float *data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
      float norm = 0.0; // 벡터의 L2 노름 계산
      for (int j = 0; j < dim; j++) {
        data[i * dim + j] = distrib_real(rng);
        norm += data[i * dim + j] * data[i * dim + j];
      }
      norm = std::sqrt(norm);

      // 벡터 정규화 (L2 노름으로 나누기)
      for (int j = 0; j < dim; j++) {
        data[i * dim + j] /= norm;
      }
    }
  */
  //
  std::mt19937 rng(static_cast<int>(47));
  std::uniform_real_distribution<float> distrib_real;
  float *data = new float[max_elements * dim];
  float *data_normed = new float[max_elements];
  float l2_mag = 0;

  for (int i = 0; i < max_elements * dim; ++i) {
    data[i] = distrib_real(rng);
    l2_mag += data[i] * data[i];
    if ((i + 1) % dim == 0) {
      normalize_vector(&l2_mag, data_normed, i / dim);
    }
  }

  int j = 0;
  for (int i = 0; i < dim * max_elements; ++i) {
    data[i] /= data_normed[j];
    if ((i + 1) % dim == 0) {
      ++j;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();

  // 종료 시간 기록

  // Add data to index
  for (int i = 0; i < max_elements; i++) {
    alg_hnsw->addPoint(data + i * dim, i);
  }

  auto end = std::chrono::high_resolution_clock::now();

  // 소요 시간 계산 (마이크로초 단위)
  //   auto duration =
  //       std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  //   std::cout << "Elapsed time: " << duration.count() << " microseconds"
  //             << std::endl;
  auto duration = std::chrono::duration<double>(end - start); // double 초 변환

  std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;

  auto startSearch = std::chrono::high_resolution_clock::now();

  // Query the elements for themselves and measure recall
  float correct = 0;
  for (int i = 0; i < max_elements; i++) {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
        alg_hnsw->searchKnn(data + i * dim, 1);
    hnswlib::labeltype label = result.top().second;
    if (label == i)
      correct++;
  }
  auto endSearch = std::chrono::high_resolution_clock::now();

  float recall = correct / max_elements;
  std::cout << "Recall: " << recall << "\n";

  //   // 소요 시간 계산 (마이크로초 단위)
  //   auto durationSearch =
  //   std::chrono::duration_cast<std::chrono::microseconds>(
  //       endSearch - startSearch);

  //   std::cout << "Elapsed time: " << durationSearch.count() << "
  //   microseconds"
  //             << std::endl;

  auto durationSearch =
      std::chrono::duration<double>(endSearch - startSearch); // double 초 변환

  std::cout << "Elapsed time: " << durationSearch.count() << " seconds"
            << std::endl;

  // Serialize index
  std::string hnsw_path = "hnsw.bin";
  alg_hnsw->saveIndex(hnsw_path);
  delete alg_hnsw;

  // Deserialize index and check recall
  alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
  correct = 0;
  for (int i = 0; i < max_elements; i++) {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
        alg_hnsw->searchKnn(data + i * dim, 1);
    hnswlib::labeltype label = result.top().second;
    if (label == i)
      correct++;
  }
  recall = (float)correct / max_elements;
  std::cout << "Recall of deserialized index: " << recall << "\n";

  delete[] data;
  delete alg_hnsw;
  return 0;
}
