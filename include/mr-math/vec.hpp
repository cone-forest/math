#ifndef __MR_VEC_HPP_
#define __MR_VEC_HPP_

#include "def.hpp"
#include "row.hpp"

namespace mr {
  // forward declarations
  template <ArithmeticT T, std::size_t N> requires (N >= 2)
    struct Vec;
  template <ArithmeticT T, std::size_t N> requires (N >= 2)
    struct Norm;
  template <ArithmeticT T, std::size_t N>
    struct Matr;
  template <ArithmeticT T>
    struct Quat;

  // common aliases
  template <ArithmeticT T>
    using Vec2 = Vec<T, 2>;
  template <ArithmeticT T>
    using Vec3 = Vec<T, 3>;
  template <ArithmeticT T>
    using Vec4 = Vec<T, 4>;

  using Vec2i = Vec2<int>;
  using Vec3i = Vec3<int>;
  using Vec4i = Vec4<int>;

  using Vec2u = Vec2<uint32_t>;
  using Vec3u = Vec3<uint32_t>;
  using Vec4u = Vec4<uint32_t>;

  using Vec2f = Vec2<float>;
  using Vec3f = Vec3<float>;
  using Vec4f = Vec4<float>;

  using Vec2d = Vec2<double>;
  using Vec3d = Vec3<double>;
  using Vec4d = Vec4<double>;

  // base vector (use aliases for full functional)
  template <ArithmeticT T, std::size_t N> requires (N >= 2)
    struct [[nodiscard]] Vec : public RowOperators<Vec<T, N>> {
      friend struct Quat<T>;

    public:
      using ValueT = T;
      using RowT = Row<T, N>;
      using NormT = Norm<T, N>;
      using SimdT = SimdImpl<T, N>;
      static constexpr size_t size = N;

      RowT _data;

      constexpr Vec() noexcept = default;

      // from simd constructor
      constexpr Vec(const RowT &row) noexcept : _data(row._data) {};

      // from elements constructor
      template <ArithmeticT... Args>
      requires (sizeof...(Args) >= 2) && (sizeof...(Args) <= N)
        constexpr Vec(Args... args) : _data(args...) {}

      // from span constructor
      // TODO: implement using Vc
      template <ArithmeticT U, size_t M>
        constexpr Vec(std::span<const U, M> span) noexcept {
          const size_t len = std::min(N, span.size());
          for (size_t i = 0; i < len; i++) {
            _data._set_ind(i, span[i]);
          }
        }

      // conversion constructor
      template <ArithmeticT R, std::size_t S>
        constexpr Vec(const Vec<R, S> &v) noexcept : _data(v._data) {}

// TODO: implement this using Vc library
#if 0
      template <ArithmeticT R, std::size_t S, ArithmeticT ... Args>
        constexpr Vec(const Vec<R, S> &v, Args ... args) noexcept : _data(v, args...) {}
#endif

      // setters
      constexpr void set(size_t i, T value) noexcept { _data._set_ind(i, value); } // for some reason `T & RowT::operator[]` doesn't compile (maybe I'm just stupid)
      constexpr void x(T x) noexcept requires (N >= 1) { set(0, x); }
      constexpr void y(T y) noexcept requires (N >= 2) { set(1, y); }
      constexpr void z(T z) noexcept requires (N >= 3) { set(2, z); }
      constexpr void w(T w) noexcept requires (N >= 4) { set(3, w); }

      // getters
      [[nodiscard]] constexpr T x() const noexcept requires (N >= 1) { return _data[0]; }
      [[nodiscard]] constexpr T y() const noexcept requires (N >= 2) { return _data[1]; }
      [[nodiscard]] constexpr T z() const noexcept requires (N >= 3) { return _data[2]; }
      [[nodiscard]] constexpr T w() const noexcept requires (N >= 4) { return _data[3]; }
      [[nodiscard]] constexpr T operator[](std::size_t i) const { return _data[i]; }

      // structured binding support
      template <size_t I> requires (I < N) constexpr T get() const { return _data[I]; }

      // cross product
      constexpr Vec cross(const Vec &other) const noexcept requires (N == 3) {
        return RowT(_data._data.rotated(1) * other._data._data.rotated(-1)
          - _data._data.rotated(-1) * other._data._data.rotated(1));

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

      constexpr Vec operator%(const Vec &other) const noexcept requires (N == 3) {
        return cross(other);
      }

      // length methods
      [[nodiscard]] constexpr T length2() const noexcept {
        return (_data._data * _data._data).sum();
      }

      [[nodiscard]] constexpr T length() const noexcept {
        return std::sqrt(length2());
      }

      // use 1 / length() for higher precision
      [[nodiscard]] constexpr T inversed_length() const {
        return fast_rsqrt(length2());
      }

      // normalize methods
      constexpr Vec & normalize() noexcept {
        auto len = length2();
        if (len <= _epsilon) [[unlikely]] return *this;
        *this /= std::sqrt(len);
        return *this;
      };

      constexpr std::optional<NormT> normalized() const noexcept {
        auto len = length2();
        if (len <= _epsilon) [[unlikely]] return std::nullopt;
        return {{mr::unchecked, *this / std::sqrt(len)}};
      };

      constexpr Vec & normalize_unchecked() noexcept {
        auto len = length2();
        *this /= std::sqrt(len);
        return *this;
      };

      constexpr NormT normalized_unchecked() const noexcept {
        auto len = length2();
        return {mr::unchecked, *this / std::sqrt(len)};
      };

      // use normalize() for higher precision
      constexpr Vec & normalize_fast() noexcept {
        auto len = length2();
        if (len <= _epsilon) [[unlikely]] return *this;
        *this *= fast_rsqrt(len);
        return *this;
      };

      // use normalized() for higher precision
      constexpr std::optional<NormT> normalized_fast() const noexcept {
        auto len = length2();
        if (len <= _epsilon) [[unlikely]] return std::nullopt;
        return {{mr::unchecked, *this * fast_rsqrt(len)}};
      };

      // use normalize_unchecked() for higher precision
      constexpr Vec & normalize_fast_unchecked() {
        auto l = length2();
        *this *= fast_rsqrt(l);
        return *this;
      };

      // use normalized_unchecked() for higher precision
      constexpr NormT normalized_fast_unchecked() const {
        auto l = length2();
        return {mr::unchecked, *this * fast_rsqrt(l)};
      };

      // dot product
      [[nodiscard]] constexpr T dot(const Vec &other) const noexcept {
        return (_data._data * other._data._data).sum();
      }

      [[nodiscard]] constexpr T operator&(const Vec &other) const noexcept {
        return dot(other);
      }

      template<ArithmeticT R>
        constexpr Vec operator*(const Matr<R, N> &other) const noexcept {
          Vec res {};
          for (size_t i = 0; i < N; i++) {
            res._data += (other._data[i] * _data[i])._data;
          }
          return res;
        }

      template<ArithmeticT R>
        constexpr Vec operator*(const Matr<R, N + 1> &other) noexcept {
          Vec<T, N + 1> copy = Vec<T, N + 1>(*this);
          Vec<T, N + 1> tmp {};

          for (size_t i = 0; i < N; i++) {
            tmp._data += (other._data[i] * copy._data[i])._data;
          }
          tmp._data += other._data[N]._data;
          return {tmp};
        }

      template<ArithmeticT R>
        constexpr Vec & operator*=(const Matr<R, N> &other) noexcept {
          Vec tmp {};
          for (size_t i = 0; i < N; i++) {
            tmp._data += (other._data[i] * _data[i])._data;
          }
          *this = tmp;
          return *this;
        }

      template<ArithmeticT R>
        constexpr Vec & operator*=(const Matr<R, N + 1> &other) noexcept {
          Vec<T, N + 1> copy = Vec<T, N + 1>(*this);
          Vec<T, N + 1> tmp {};
          for (size_t i = 0; i < N; i++) {
            tmp._data += (other._data[i] * copy._data[i])._data;
          }
          tmp._data += other._data[N]._data;

          *this = Vec<T, N>(tmp);
          return *this;
        }

        // reflect from other vector
        constexpr Vec reflected(const NormT &n) const noexcept {
          return -(*this - (2 * dot(n) * (Vec)n));
        }

        constexpr Vec & reflect(const NormT &n) noexcept {
          *this = reflected(n);
          return *this;
        }

        constexpr Vec absed() const noexcept {
          return {stdx::abs(_data._data)};
        }

        constexpr Vec & abs() noexcept {
          *this = absed();
          return *this;
        }

        constexpr Vec clamped(T low, T high) const noexcept {
          assert(low < high);
          const auto &data = _data._data;
          return {stdx::iif(data <= low, SimdT(low), stdx::iif(data >= high, SimdT(high), data))};
        }

        constexpr Vec & clamp(T low, T high) noexcept {
          *this = clamped(low, high);
          return *this;
        }

        constexpr bool operator==(const Vec &other) const noexcept {
          return _data == other._data;
        }

        constexpr bool equal(const Vec &other, ValueT eps = epsilon<ValueT>()) const noexcept {
          return _data.equal(other._data, eps);
        }

      private:
        static constexpr T _epsilon = std::numeric_limits<T>::epsilon();
    };
} // namespace mr

#ifdef __cpp_structured_bindings
// specializations for structured binding support
namespace std
{
  template <mr::ArithmeticT T, std::size_t N>
  struct tuple_size<mr::Vec<T, N>>
      : std::integral_constant<size_t, N> {};

  template <mr::ArithmeticT T, std::size_t N, std::size_t I>
  struct tuple_element<I, mr::Vec<T, N>> {
    using type = T;
  };
}
#endif

#endif // __MR_VEC_HPP_
