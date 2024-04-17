#ifndef __Row_hpp_
#define __Row_hpp_

#include "def.hpp"

namespace mr {
  template <typename T, std::size_t N>
    using SimdImpl = stdx::fixed_size_simd<T, N>;

  template <typename DerivedT>
    struct UnitOperators {
      friend constexpr DerivedT
      operator+(const DerivedT &lhs, const DerivedT &rhs) noexcept {
        return DerivedT{lhs._data + rhs._data};
      }

      friend constexpr DerivedT
      operator-(const DerivedT &lhs, const DerivedT &rhs) noexcept {
        return DerivedT{lhs._data - rhs._data};
      }

      friend constexpr DerivedT &
      operator+=(DerivedT &lhs, const DerivedT &rhs) noexcept {
        lhs._data += rhs._data;
        return lhs;
      }

      friend constexpr DerivedT &
      operator-=(DerivedT &lhs, const DerivedT &rhs) noexcept {
        lhs._data -= rhs._data;
        return lhs;
      }

      friend constexpr DerivedT
      operator*(const DerivedT &lhs, const ArithmeticT auto x) noexcept {
        return DerivedT{lhs._data * x};
      }

      friend constexpr DerivedT
      operator/(const DerivedT &lhs, const ArithmeticT auto x) noexcept {
        return DerivedT{lhs._data / x};
      }

      friend constexpr DerivedT
      operator*(const ArithmeticT auto x, const DerivedT &lhs) noexcept {
        return DerivedT{lhs._data * (float)x};
      }

      friend constexpr DerivedT &
      operator*=(DerivedT &lhs, const ArithmeticT auto x) noexcept {
        lhs._data *= x;
        return lhs;
      }

      friend constexpr DerivedT &
      operator/=(DerivedT &lhs, const ArithmeticT auto x) noexcept {
        lhs._data /= x;
        return lhs;
      }

      friend constexpr DerivedT
      operator-(const DerivedT &rhs) noexcept {
        return -rhs._data;
      }
    };

  template <typename DerivedT>
    struct RowOperators : UnitOperators<DerivedT> {
      friend constexpr DerivedT
      operator*(const DerivedT &lhs, const DerivedT &rhs) noexcept {
        return DerivedT{lhs._data * rhs._data};
      }

      friend constexpr DerivedT
      operator/(const DerivedT &lhs, const DerivedT &rhs) noexcept {
        return DerivedT{lhs._data / rhs._data};
      }

      friend constexpr DerivedT
      operator<<(const DerivedT &lhs, const DerivedT &rhs) noexcept {
        return DerivedT{lhs._data << rhs._data};
      }

      friend constexpr DerivedT
      operator>>(const DerivedT &lhs, const DerivedT &rhs) noexcept {
        return DerivedT{lhs._data >> rhs._data};
      }

      friend constexpr DerivedT &
      operator*=(DerivedT &lhs, const DerivedT &rhs) noexcept {
        lhs._data *= rhs._data;
        return lhs;
      }

      friend constexpr DerivedT &
      operator/=(DerivedT &lhs, const DerivedT &rhs) noexcept {
        lhs._data /= rhs._data;
        return lhs;
      }

      friend constexpr DerivedT &
      operator<<=(DerivedT &lhs, const DerivedT &rhs) noexcept {
        lhs._data <<= rhs._data;
        return lhs;
      }

      friend constexpr DerivedT &
      operator>>=(DerivedT &lhs, const DerivedT &rhs) noexcept {
        lhs._data >>= rhs._data;
        return lhs;
      }

      friend std::ostream &
      operator<<(std::ostream &s, const DerivedT &v) noexcept {
        s << '(';
        for (size_t i = 0; i < DerivedT::size - 1; i++) {
          s << v[i] << ", ";
        }
        s << v[DerivedT::size - 1] << ')';
        return s;
      }
    };

  template <ArithmeticT T, std::size_t N>
    struct Row : RowOperators<Row<T, N>> {
    public:
      using SimdT = SimdImpl<T, N>;
      static constexpr size_t size = N;

      SimdT _data{};

      constexpr Row() noexcept = default;

      // simd-like constructors
      constexpr Row(const SimdT &rhs) noexcept : _data(rhs) {}

      constexpr Row(const T data) : _data(data) {}

      constexpr Row(const T *data) {
        _data.copy_from(data, stdx::element_aligned);
      }

      // from elements constructor
      template <std::convertible_to<T>... Args>
      requires (sizeof...(Args) >= 2) && (sizeof...(Args) <= N)
        constexpr Row(Args... args) {
          _set(args...);
        }

      // copy constructor from different row type
      template <ArithmeticT R, std::size_t S>
        requires std::is_convertible_v<R, T>
        constexpr Row(const Row<R, S> &rhs) noexcept {
          // size conversion
          std::array<R, std::max(S, N)> tmp1;
          rhs._data.copy_to(tmp1.data(), stdx::element_aligned);

          stdx::fixed_size_simd<R, N> tmp2;
          tmp2.copy_from(tmp1.data(), stdx::element_aligned);

          // type conversion
          _data = tmp2;
        }

      template <ArithmeticT R, std::size_t S, std::convertible_to<R>... Args>
        requires (sizeof...(Args) + S == N)
        constexpr Row(const Row<R, S> &rhs, Args ...args) noexcept {
          // size conversion
          std::array<R, N> tmp1;
          rhs._data.copy_to(tmp1.data(), stdx::element_aligned);

          Row tmp3 {static_cast<R>(args)...};
          tmp3._data.copy_to(tmp1.data() + S, stdx::element_aligned);

          stdx::fixed_size_simd<R, N> tmp2;
          tmp2.copy_from(tmp1.data(), stdx::element_aligned);

          // type conversion
          _data = tmp2;
        }

      // fixed_size_simd -> intrinsics
      [[nodiscard]]
      constexpr auto to_underlying() const noexcept {
        auto tmp2 =
          stdx::__vector_convert<stdx::fixed_size_simd<float, 8>>(_data,
              std::index_sequence<_data.size()>{});

        stdx::native_simd<T> tmp = stdx::simd_cast<stdx::native_simd<T>>(tmp2);
        auto tmp3 = stdx::__as_vector(tmp);

        if constexpr (stdx::native_simd<T>::size == 8) {
#if defined(__SSE__)
          return _mm256_castps256_ps128(tmp3);
#elif __arm__ || __aarch64__
#endif
        }
        return tmp3;
      };

      // intrinsics -> fixed_size_simd
      constexpr void from_underlying(auto vec) noexcept {
#if defined(__SSE__)
#elif __arm__ || __aarch64__
#endif
      }

      template <size_t ...Indices>
        [[nodiscard]] constexpr Row shuffled(std::index_sequence<Indices...>) const noexcept
        requires (sizeof...(Indices) == size) {

          auto underlying = to_underlying();

#if defined(__SSE__)
          underlying = _mm_shuffle_ps(underlying, underlying, _MM_SHUFFLE(Indices...));
#elif __arm__ || __aarch64__
#endif

          return Row{}.from_underlying(underlying);
        }

      template <typename ...Args> requires (std::is_same_v<Args, std::size_t> && ...)
      [[nodiscard]] constexpr Row shuffled(Args... indices) const noexcept {
        return shuffled<resized_index_sequence<stdx::native_simd<T>::size, indices...>>();
      }

      [[nodiscard]] constexpr T operator[](std::size_t i) const {
        return _data[i];
      }

    protected:
      template <std::convertible_to<T>... Args>
        requires (sizeof...(Args) >= 1) && (sizeof...(Args) <= N)
        constexpr void _set(Args... args) noexcept {
          std::array<T, N> arr {static_cast<T>(args)...};
          _data.copy_from(arr.data(), stdx::element_aligned);
        }

      constexpr void _set_ind(std::size_t ind, T value) noexcept {
        std::array<T, N> arr;
        _data.copy_to(arr.data(), stdx::element_aligned);
        arr[ind] = value;
        _data.copy_from(arr.data(), stdx::element_aligned);
      }
    };
} // namespace mr

#ifdef __cpp_lib_format
// std::format support
namespace std {
  template <mr::ArithmeticT T, size_t N>
    struct formatter<mr::Row<T, N>> {
      template<typename ParseContext>
        constexpr auto parse(ParseContext& ctx) {
          // skip all format specifiers
          return ctx.end();
        }

      template<typename FmtContext>
        auto format(const mr::Row<T, N> &r, FmtContext& ctx) const {
          ostringstream out;
          out << r;
          return ranges::copy(move(out).str(), ctx.out()).out;
        }
    };
} // namespace std
#endif

#endif // __Row_hpp_
