#ifndef __Matr_hpp_
#define __Matr_hpp_

#include "def.hpp"
#include "row.hpp"
#include "units.hpp"

namespace mr
{
  // forward declarations
  template <ArithmeticT T, std::size_t N>
    class Matr;

  // common aliases
  template <ArithmeticT T>
    using Matr4 = Matr<T, 4>;

  using Matr4f = Matr4<float>;
  using Matr4d = Matr4<double>;
  using Matr4i = Matr4<int>;
  using Matr4u = Matr4<uint32_t>;

  template <ArithmeticT T, std::size_t N>
    class [[nodiscard]] Matr
    {
    public:
      using RowT = Row<T, N>;

      template <typename... Args>
        requires (std::is_same_v<Args, RowT> && ...) &&
                 ((sizeof...(Args) == 0) || (sizeof...(Args) == N))
        constexpr Matr(Args... args) noexcept {
          _data = std::array<RowT, N>({args...});
        }

      constexpr Matr(const std::array<RowT, N> &arr) : _data(arr) {}

      // copy semantics
      constexpr Matr(const Matr &) noexcept = default;
      constexpr Matr & operator=(const Matr &) noexcept = default;

      // move semantics
      constexpr Matr(Matr &&) noexcept = default;
      constexpr Matr & operator=(Matr &&) noexcept = default;

      constexpr ~Matr() = default;

      // basic math operations
      constexpr Matr & operator*=(const Matr &other) noexcept {
        std::array<RowT, N> tmp {};
        for (size_t i = 0; i < N; i++) {
          for (size_t j = 0; j < N; j++) {
            tmp._data[j] += other._data[i] * _data[j][i];
          }
        }
        *this = tmp;
        return *this;
      }

      constexpr Matr & operator+=(const Matr &other) noexcept {
        for (int i = 0; i < N; i++)
          _data[i] += other._data[i];
        return *this;
      }

      constexpr Matr & operator-=(const Matr &other) noexcept {
        for (int i = 0; i < N; i++)
          _data[i] -= other._data[i];
        return *this;
      }

      constexpr Matr operator*(const Matr &other) const noexcept {
        std::array<RowT, N> tmp;
        for (size_t i = 0; i < N; i++) {
          for (size_t j = 0; j < N; j++) {
            tmp[j] += other._data[i] * _data[j][i];
          }
        }
        return {tmp};
      }

      constexpr Matr operator+(const Matr &other) const noexcept {
        std::array<RowT, N> tmp;
        for (size_t i = 0; i < N; i++)
          tmp[i] = static_cast<RowT>(_data[i] + other._data[i]);
        return {tmp};
      }

      constexpr Matr operator-(const Matr &other) const noexcept {
        std::array<RowT, N> tmp;
        for (size_t i = 0; i < N; i++)
          tmp[i] = _data[i] - other._data[i];
        return {tmp};
      }

      // matrix related operations
      [[nodiscard]] constexpr const RowT & operator[](size_t i) const noexcept {
        return _data[i];
      }

      [[nodiscard]] constexpr T determinant() const noexcept {
        std::array<RowT, N> tmp = _data;

        for (size_t i = 1; i < N; i++) {
          for (size_t j = i; j < N; j++) {
            tmp[i] -= tmp[i][i - 1] == 0
              ? 0
              : tmp[i - 1] * tmp[j][i - 1] / tmp[i - 1][i - 1];
          }
        }

        constexpr auto io0 = std::ranges::iota_view((size_t)0, N);
        return std::transform_reduce(
          std::execution::par_unseq,
          io0.begin(), io0.end(), 1,
          std::multiplies {},
          [&tmp](auto i) { return tmp[i][i]; });
      }

      [[nodiscard]] constexpr T operator!() const noexcept {
        return determinant();
      }

      constexpr Matr transposed() const noexcept {
        std::array<RowT, N> tmp1;
        std::array<std::array<T, N>, N> tmp2;
        for (size_t i = 0; i < N; i++)
          for (size_t j = 0; j < N; j++)
            tmp2[i][j] = _data[j][i];
        for (size_t i = 0; i < N; i++)
          tmp1[i]._data.copy_from(tmp2[i].data(), stdx::element_aligned);
        return {tmp1};
      }

      constexpr Matr & transpose() noexcept {
        std::array<std::array<T, N>, N> tmp2;
        for (size_t i = 0; i < N; i++)
          for (size_t j = 0; j < N; j++)
            tmp2[i][j] = _data[j][i];
        for (size_t i = 0; i < N; i++)
          _data[i].copy_from(tmp2[i].data(), stdx::element_aligned);
        return *this;
      }

      constexpr Matr inversed() const noexcept {
        constexpr auto io = std::ranges::iota_view{(size_t)0, N};

        std::array<Row<T, 2 * N>, N> tmp;
        std::for_each(std::execution::par_unseq, io.begin(), io.end(),
                      [&tmp, this](auto i) { tmp[i] += stdx::simd_cast<stdx::fixed_size_simd<T, 2 * N>>(stdx::concat(_data[i]._data, identity[i]._data)); });

        // null bottom triangle
        for (size_t i = 1; i < N; i++) {
          for (size_t j = i; j < N; j++) {
            tmp[j] -= tmp[i - 1][i - 1] == 0 ? 0 : tmp[i - 1] * tmp[j][i - 1] / tmp[i - 1][i - 1];
          }
        }

        // null top triangle
        for (int i = N - 2; i >= 0; i--) {
          for (int j = i; j >= 0; j--) {
            tmp[j] -= tmp[i + 1][i + 1] == 0 ? 0 : tmp[i + 1] * tmp[j][i + 1] / tmp[i + 1][i + 1];
          }
        }

        // make main diagonal 1
        std::for_each(std::execution::par_unseq,
          io.begin(), io.end(), [&tmp](auto i) { tmp[i] /= tmp[i][i]; });

        std::array<RowT, N> res;
        std::for_each(std::execution::par_unseq, io.begin(), io.end(),
          [&tmp, &res](auto i) {
            auto [a, b] = stdx::split<N, N>(tmp[i]._data);
            res[i] += stdx::simd_cast<stdx::fixed_size_simd<T, N>>(b);
          });

        return {res};
      }

      constexpr Matr & inverse() const noexcept {
        *this = inversed();
        return *this;
      }

      static constexpr Matr4<T> scale(const Vec3<T> &vec) noexcept {
        return Matr4<T> {
          typename mr::Matr4<T>::RowT(vec[0], 0, 0, 0),
          typename mr::Matr4<T>::RowT(0, vec[1], 0, 0),
          typename mr::Matr4<T>::RowT(0, 0, vec[2], 0),
          typename mr::Matr4<T>::RowT(0, 0, 0, 1)
        };
      }

      static constexpr Matr4<T> translate(const Vec3<T> &vec) noexcept {
        return Matr4<T> {
          typename mr::Matr4<T>::RowT(1, 0, 0, 0),
          typename mr::Matr4<T>::RowT(0, 1, 0, 0),
          typename mr::Matr4<T>::RowT(0, 0, 1, 0),
          typename mr::Matr4<T>::RowT(vec[0], vec[1], vec[2], 1)
        };
      }

      static constexpr Matr4<T> rotate_x(const Radians<T> &rad) noexcept {
        T co = std::cos(rad.value);
        T si = std::sin(rad.value);

        return Matr4<T> {
          typename mr::Matr4<T>::RowT(1, 0, 0, 0),
          typename mr::Matr4<T>::RowT(0, co, si, 0),
          typename mr::Matr4<T>::RowT(0, -si, co, 0),
          typename mr::Matr4<T>::RowT(0, 0, 0, 1)
        };
      }

      static constexpr Matr4<T> rotate_y(const Radians<T> &rad) noexcept {
        T co = std::cos(rad.value);
        T si = std::sin(rad.value);

        return Matr4<T> {
          typename mr::Matr4<T>::RowT(co, 0, -si, 0),
          typename mr::Matr4<T>::RowT(0, 1, 0, 0),
          typename mr::Matr4<T>::RowT(si, 0, co, 0),
          typename mr::Matr4<T>::RowT(0, 0, 0, 1)
        };
      }

      static constexpr Matr4<T> rotate_z(const Radians<T> &rad) noexcept {
        T co = std::cos(rad.value);
        T si = std::sin(rad.value);

        return Matr4<T> {
          typename mr::Matr4<T>::RowT(co, si, 0, 0),
          typename mr::Matr4<T>::RowT(-si, co, 0, 0),
          typename mr::Matr4<T>::RowT(0, 0, 1, 0),
          typename mr::Matr4<T>::RowT(0, 0, 0, 1)
        };
      }

      static constexpr Matr4<T> rotate(const Radians<T> &rad, const Vec3<T> &vec) noexcept {
        T co = std::cos(rad.value);
        T si = std::sin(rad.value);
        T nco = 1 - co;
        Vec4<T> tmp = Vec4<T>(vec * vec * nco + co);
        Matr4<T> tmp1 = scale(tmp);
        Matr4<T> tmp2 = Matr4<T> {
          typename mr::Matr4<T>::RowT(0, vec[0] * vec[1] * nco, vec[0] * vec[2] * nco, 0),
          typename mr::Matr4<T>::RowT(vec[0] * vec[1] * nco, 0, vec[1] * vec[2] * nco, 0),
          typename mr::Matr4<T>::RowT(vec[0] * vec[2] * nco, vec[1] * vec[2] * nco, 0, 0),
          typename mr::Matr4<T>::RowT(0, 0, 0, 0)
        };
        Matr4<T> tmp3 = Matr4<T> {
          typename mr::Matr4<T>::RowT(0, vec[2] * si, -vec[1] * si, 0),
          typename mr::Matr4<T>::RowT(-vec[2] * si, 0, vec[0] * si, 0),
          typename mr::Matr4<T>::RowT(vec[1] * si, -vec[0] * si, 0, 0),
          typename mr::Matr4<T>::RowT(0, 0, 0, 0)
        };

        return tmp1 + tmp2 + tmp3;
      }

      friend std::ostream & operator<<(std::ostream &s, const Matr &m) noexcept {
        for (size_t i = 0; i < N; i++)
          std::cout << m._data[i] << std::endl;
        return s;
      }

    private:
      static Matr _identity() {
        std::array<RowT, N> id;
        constexpr auto io = std::ranges::iota_view {(size_t)0, N};

        std::transform(std::execution::par_unseq,
          io.begin(), io.end(), id.begin(),
          [&io](auto i) -> RowT {
            std::array<T, N> tmp;
            std::transform(std::execution::par_unseq,
              io.begin(), io.end(), tmp.begin(),
              [i](auto i2) -> T { return i2 == i ? 1 : 0; });
            return {tmp.data()};
          });

        return {id};
      }

    public:
      inline static const Matr identity = _identity();
      std::array<RowT, N> _data;
    };
} // namespace mr

#endif // __Matr_hpp_
