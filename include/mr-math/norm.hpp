#ifndef __MR_NORM_HPP_
#define __MR_NORM_HPP_

#include "def.hpp"
#include "vec.hpp"

namespace mr {
  template <std::floating_point T>
    struct Rotation;

  // common aliases
  template <ArithmeticT T>
    using Norm2 = Norm<T, 2>;
  template <ArithmeticT T>
    using Norm3 = Norm<T, 3>;
  template <ArithmeticT T>
    using Norm4 = Norm<T, 4>;

  using Norm2i = Norm2<int>;
  using Norm3i = Norm3<int>;
  using Norm4i = Norm4<int>;

  using Norm2u = Norm2<uint32_t>;
  using Norm3u = Norm3<uint32_t>;
  using Norm4u = Norm4<uint32_t>;

  using Norm2f = Norm2<float>;
  using Norm3f = Norm3<float>;
  using Norm4f = Norm4<float>;

  using Norm2d = Norm2<double>;
  using Norm3d = Norm3<double>;
  using Norm4d = Norm4<double>;

  template <ArithmeticT T, std::size_t N> requires (N >= 2)
    struct [[nodiscard]] Norm {
      using ValueT = T;
      using VecT = Vec<T, N>;

      // from elements constructor
      template <ArithmeticT... Args>
      requires (sizeof...(Args) >= 2) && (sizeof...(Args) <= N)
        constexpr Norm(Args... args) : _data(args...) {
          _data.normalize();
        }

      constexpr Norm(const VecT &v) noexcept : Norm(unchecked, v.normalized_unchecked()) {}

      constexpr Norm(UncheckedTag, const VecT &v) noexcept : _data(v) {
          assert(mr::equal(v.length(), 1, 0.1f));
        }

      constexpr operator VecT() const noexcept { return _data; }

      // getters
      [[nodiscard]] constexpr T x() const noexcept requires (N >= 1) { return _data[0]; }
      [[nodiscard]] constexpr T y() const noexcept requires (N >= 2) { return _data[1]; }
      [[nodiscard]] constexpr T z() const noexcept requires (N >= 3) { return _data[2]; }
      [[nodiscard]] constexpr T w() const noexcept requires (N >= 4) { return _data[3]; }
      [[nodiscard]] constexpr T operator[](std::size_t i) const { return _data[i]; }

      // structured binding support
      template <size_t I> requires (I < N) constexpr T get() const { return _data[I]; }

      // cross product
      constexpr VecT cross(const VecT &other) const noexcept requires (N == 3) {
        return RowT(_data._data.shifted(-1) * other._data._data.shifted(1)
          - _data._data.shifted(1) * other._data._data.shifted(-1));
#if 0
        std::array<T, 3> arr {
          _data[1] * other._data[2] - _data[2] * other._data[1],
          _data[2] * other._data[0] - _data[0] * other._data[2],
          _data[0] * other._data[1] - _data[1] * other._data[0]
        };

        SimdImpl<T, 3> ans;
        ans.copy_from(arr.data(), stdx::element_aligned);
        return {ans};
#endif
      }

      constexpr VecT operator%(const VecT &other) const noexcept requires (N == 3) {
        return cross(other);
      }

      // dot product
      [[nodiscard]] constexpr T dot(const VecT &other) const noexcept {
        return (_data._data._data * other._data._data).sum();
      }

      [[nodiscard]] constexpr T operator&(const VecT &other) const noexcept {
        return dot(other);
      }

      [[nodiscard]] constexpr VecT operator*(const Norm &other) const noexcept {
        return _data * other._data;
      }

      // matrix multiplication
      template<ArithmeticT R>
        constexpr VecT operator*(const Matr<R, N> &other) const noexcept {
          VecT tmp {};
          for (size_t i = 0; i < N; i++) {
            tmp._data += (other._data[i] * _data[i])._data;
          }
          return tmp;
        }

      template<ArithmeticT R>
        constexpr VecT operator*(const Matr<R, N + 1> &other) noexcept {
          Vec<T, N + 1> copy = Vec<T, N + 1>(*this);
          Vec<T, N + 1> tmp {};

          for (size_t i = 0; i < N; i++) {
            tmp._data += (other._data[i] * copy._data[i])._data;
          }
          tmp._data += other._data[N]._data;
          return {tmp};
        }

        friend std::ostream & operator<<(std::ostream &os, const Norm &n) noexcept {
          os << '(';
          for (size_t i = 0; i < N - 1; i++) {
            os << n[i] << ", ";
          }
          os << n[N - 1] << ')';
          return os;
        }

        constexpr friend Norm operator-(const Norm &n) noexcept {
          return {unchecked, -n._data};
        }

        constexpr bool operator==(const Norm &other) const noexcept {
          return _data == other._data;
        }
        constexpr bool operator==(const VecT &other) const noexcept {
          return _data == other;
        }

        constexpr bool equal(const Norm &other, ValueT eps = epsilon<ValueT>()) const noexcept {
          return _data.equal(other._data, eps);
        }
        constexpr bool equal(const VecT &other, ValueT eps = epsilon<ValueT>()) const noexcept {
          return _data.equal(other, eps);
        }

      private:
        VecT _data;
    };
} // namespace mr

#ifdef __cpp_structured_bindings
// specializations for structured binding support
namespace std
{
  template <mr::ArithmeticT T, std::size_t N>
  struct tuple_size<mr::Norm<T, N>>
      : std::integral_constant<size_t, N> {};

  template <mr::ArithmeticT T, std::size_t N, std::size_t I>
  struct tuple_element<I, mr::Norm<T, N>> {
    using type = T;
  };
}
#endif

#endif // __MR_NORM_HPP_
