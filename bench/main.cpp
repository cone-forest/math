#include "mr-math/math.hpp"

#include <benchmark/benchmark.h>

volatile float a = 1; // to disable constexpr calculations

mr::Vec3f v1 {a, 0, 0};
mr::Vec3f v2 {0, a, 0};
mr::Vec3f v3 {0, 0, a};

mr::Matr4f m1 {
  a, 2, 3, 4,
  a, 3, 4, 5,
  a, 4, 5, 6,
  a, 5, 6, 7
};
mr::Matr4f m2 {
  mr::Matr4f::RowT{a, 6, 5, 4},
  mr::Matr4f::RowT{a, 5, 4, 3},
  mr::Matr4f::RowT{a, 4, 3, 2},
  mr::Matr4f::RowT{a, 3, 2, 1}
};

mr::Camera<float> cam {};

static void BM_camera_perspective(benchmark::State& state) {
  for (auto _ : state) {
    auto m = cam.calculate_perspective();
    benchmark::DoNotOptimize(m);
  }
}
BENCHMARK(BM_camera_perspective);

static void BM_camera_ortholinear(benchmark::State& state) {
  for (auto _ : state) {
    auto m = cam.calculate_orthographic();
    benchmark::DoNotOptimize(m);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_camera_ortholinear);

static void BM_camera_frustum(benchmark::State& state) {
  for (auto _ : state) {
    auto m = cam.calculate_frustum();
    benchmark::DoNotOptimize(m);
  }
}
BENCHMARK(BM_camera_frustum);

static void BM_camera_rotation(benchmark::State& state) {
  for (auto _ : state) {
    cam += mr::Yaw(mr::pi);
  }
}
BENCHMARK(BM_camera_rotation);

static void BM_vector_rotation(benchmark::State& state) {
  for (auto _ : state) {
    auto v3 = v1 * mr::Matr4f::rotate_z(mr::Radiansf(mr::pi));
    benchmark::DoNotOptimize(v3);
  }
}
BENCHMARK(BM_vector_rotation);

static void BM_vector_scale(benchmark::State& state) {
  for (auto _ : state) {
    auto v3 = v1 * mr::ScaleMatr3f({1, 2, 3});
    benchmark::DoNotOptimize(v3);
  }
}
BENCHMARK(BM_vector_scale);

static void BM_vector_const_multiplication(benchmark::State& state) {
  for (auto _ : state) {
    auto v3 = v1 * 3;
    benchmark::DoNotOptimize(v3);
  }
}
BENCHMARK(BM_vector_const_multiplication);

static void BM_vector_matrix_multiplication(benchmark::State& state) {
  for (auto _ : state) {
    auto v3 = v1 * m1;
    benchmark::DoNotOptimize(v3);
  }
}
BENCHMARK(BM_vector_matrix_multiplication);

static void BM_normalized(benchmark::State& state) {
  for (auto _ : state) {
    auto v3 = v1.normalized();
    benchmark::DoNotOptimize(v3);
  }
}
BENCHMARK(BM_normalized);

static void BM_normalized_fast(benchmark::State& state) {
  for (auto _ : state) {
    auto v3 = v1.normalized_fast_unchecked();
    benchmark::DoNotOptimize(v3);
  }
}
BENCHMARK(BM_normalized_fast);

static void BM_dot(benchmark::State& state) {
  for (auto _ : state) {
    auto v4 = v1.dot(v3);
    benchmark::DoNotOptimize(v4);
  }
}
BENCHMARK(BM_dot);

static void BM_cross(benchmark::State& state) {
  for (auto _ : state) {
    auto v4 = v1.cross(v2);
    benchmark::DoNotOptimize(v4);
  }
}
BENCHMARK(BM_cross);

static void BM_matrix_multiplication(benchmark::State& state) {
  for (auto _ : state) {
    auto m3 = m1 * m2;
    benchmark::DoNotOptimize(m3);
  }
}

BENCHMARK(BM_matrix_multiplication);

#if 0
static void BM_matrix_inversed(benchmark::State& state) {
  for (auto _ : state) {
    auto m3 = m1.inversed();
    benchmark::DoNotOptimize(m3);
  }
}

BENCHMARK(BM_matrix_inversed);
#endif

static void BM_matrix_determinant(benchmark::State& state) {
  for (auto _ : state) {
    auto m3 = m1.determinant();
    benchmark::DoNotOptimize(m3);
  }
}

BENCHMARK(BM_matrix_determinant);

static void BM_matrix_transposed(benchmark::State& state) {
  for (auto _ : state) {
    auto v3 = m1.transposed();
    benchmark::DoNotOptimize(v3);
  }
}
BENCHMARK(BM_matrix_transposed);


[[maybe_unused]]
static void compile_test() {
  auto f = [](mr::Vec3f v) { std::cout << v; };
  mr::Vec3f v(30);
  mr::Row<float, 3> r{47.f, 47.f, 47.f};
  mr::SimdImpl<float, 3> s(102);
  v *= -2;
  v = 3.f * r;
  r -= s / 2;
  v *= mr::Vec3f(1.0);
  v = {0, 102, 0};
  v -= {0, 102, 0};
  // v = v - {0, 102, 0};
  v += r;
  f(v);
  f(r);
}

BENCHMARK_MAIN();

