#ifndef __MR_MATR_HPP_
#define __MR_MATR_HPP_

#include "def.hpp"
#include "row.hpp"
#include "vec.hpp"
#include "norm.hpp"
#include "units.hpp"

namespace mr
{
  // forward declarations
  template <ArithmeticT T, std::size_t N>
    struct Matr;
  template <ArithmeticT T, std::size_t N>
    struct ScaleMatr;
  template <ArithmeticT T, std::size_t N>
    struct TranslateMatr;
  template <ArithmeticT T>
    struct RotateMatr;

  // common aliases
  template <ArithmeticT T>
    using Matr3 = Matr<T, 3>;
  template <ArithmeticT T>
    using Matr4 = Matr<T, 4>;

  template <ArithmeticT T>
    using ScaleMatr3 = ScaleMatr<T, 3>;
  template <ArithmeticT T>
    using ScaleMatr4 = ScaleMatr<T, 4>;

  template <ArithmeticT T>
    using TranslateMatr3 = TranslateMatr<T, 3>;
  template <ArithmeticT T>
    using TranslateMatr4 = TranslateMatr<T, 4>;

  using Matr3f = Matr3<float>;
  using Matr3d = Matr3<double>;
  using Matr3i = Matr3<int>;
  using Matr3u = Matr3<uint32_t>;
  using Matr4f = Matr4<float>;
  using Matr4d = Matr4<double>;
  using Matr4i = Matr4<int>;
  using Matr4u = Matr4<uint32_t>;

  using ScaleMatr3f = ScaleMatr3<float>;
  using ScaleMatr3d = ScaleMatr3<double>;
  using ScaleMatr3i = ScaleMatr3<int>;
  using ScaleMatr3u = ScaleMatr3<uint32_t>;
  using ScaleMatr4f = ScaleMatr4<float>;
  using ScaleMatr4d = ScaleMatr4<double>;
  using ScaleMatr4i = ScaleMatr4<int>;
  using ScaleMatr4u = ScaleMatr4<uint32_t>;

  using TranslateMatr3f = TranslateMatr3<float>;
  using TranslateMatr3d = TranslateMatr3<double>;
  using TranslateMatr3i = TranslateMatr3<int>;
  using TranslateMatr3u = TranslateMatr3<uint32_t>;
  using TranslateMatr4f = TranslateMatr4<float>;
  using TranslateMatr4d = TranslateMatr4<double>;
  using TranslateMatr4i = TranslateMatr4<int>;
  using TranslateMatr4u = TranslateMatr4<uint32_t>;

  using RotateMatr4f = RotateMatr<float>;
  using RotateMatr4d = RotateMatr<double>;
  using RotateMatr4i = RotateMatr<int>;
  using RotateMatr4u = RotateMatr<uint32_t>;

  template <ArithmeticT T, std::size_t N>
    struct [[nodiscard]] Matr
    {
    public:
      using ValueT = T;
      using RowT = Row<T, N>;

      constexpr Matr() noexcept = default;

      template <typename... Args>
        requires (std::is_same_v<Args, RowT> && ...) &&
                 (sizeof...(Args) == N)
        constexpr Matr(Args... args) noexcept {
          _data = std::array<RowT, N>({args...});
        }

      constexpr Matr(std::array<RowT, N> rows) : _data(std::move(rows)) {}

      template <typename... Args>
        requires (std::is_convertible_v<Args, T> && ...) &&
                 (sizeof...(Args) == N * N)
        constexpr Matr(Args... args) noexcept {
          std::array<T, N * N> tmp {static_cast<T>(args)...};
          for (size_t i = 0; i < N; i++) {
            _data[i] = RowT {
              tmp[N * i + 0],
              tmp[N * i + 1],
              tmp[N * i + 2],
              tmp[N * i + 3]
            };
          }
        }

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
            tmp[j] += other._data[i] * _data[j][i];
          }
        }
        *this = tmp;
        return *this;
      }

      constexpr Matr & operator+=(const Matr &other) noexcept {
        for (int i = 0; i < N; i++) {
          _data[i] += other._data[i];
        }
        return *this;
      }

      constexpr Matr & operator-=(const Matr &other) noexcept {
        for (int i = 0; i < N; i++) {
          _data[i] -= other._data[i];
        }
        return *this;
      }

      constexpr Matr operator*(const Matr &other) const noexcept {
        std::array<RowT, N> tmp{};
        for (size_t i = 0; i < N; i++) {
          for (size_t j = 0; j < N; j++) {
            tmp[j] += other._data[i] * _data[j][i];
          }
        }
        return {tmp};
      }

      constexpr Matr operator+(const Matr &other) const noexcept {
        std::array<RowT, N> tmp;
        for (size_t i = 0; i < N; i++) {
          tmp[i] = static_cast<RowT>(_data[i] + other._data[i]);
        }
        return {tmp};
      }

      constexpr Matr operator-(const Matr &other) const noexcept {
        std::array<RowT, N> tmp;
        for (size_t i = 0; i < N; i++) {
          tmp[i] = _data[i] - other._data[i];
        }
        return {tmp};
      }

      // matrix related operations
      [[nodiscard]] constexpr const RowT & operator[](size_t i) const noexcept {
        return _data[i];
      }

      [[nodiscard]] constexpr RowT & operator[](size_t i) noexcept {
        return _data[i];
      }

      // TODO: fix (or remove as it's neved used)
      [[nodiscard]] constexpr T determinant() const noexcept {
        std::array<RowT, N> tmp = _data;
        T det = 1;

        tmp[N - 1] /= tmp[N - 1][N - 1];
        for (size_t i = 0; i < N - 1; i++) {
          det *= tmp[i][i];
          tmp[i] /= mr::equal(tmp[i][i], 0) ? static_cast<T>(1) : tmp[i][i];
          for (size_t j = i + 1; j < N; j++) {
            tmp[i + 1] -= tmp[i] * tmp[j][i];
          }
        }
        return det;
      }

      constexpr Matr transposed() const noexcept {
        Matr transposed;
        for (size_t i = 0; i < N; i++) {
          transposed[i] = SimdImpl<T, N>([this, i](size_t j) { return _data[j][i]; });
        }
        return transposed;
      }

      constexpr Matr & transpose() noexcept {
        *this = transposed();
        return *this;
      }

// TODO: implement this using Vc library
#if 0
      constexpr Matr inversed() const noexcept {
        constexpr auto io = std::ranges::iota_view{(size_t)0, N};

        std::array<Row<T, 2 * N>, N> tmp;
        std::for_each(io.begin(), io.end(),
                      [&tmp, this](auto i) {
                        // adding temporary variable here brings performance down 2.5 times (reason unknown)
                        tmp[i] += stdx::simd_cast<SimdImpl<T, 2 * N>>(stdx::concat(_data[i]._data, identity[i]._data));
                      });

        // null bottom triangle
        for (size_t i = 1; i < N; i++) {
          tmp[i - 1] /= tmp[i - 1][i - 1];
          for (size_t j = i; j < N; j++) {
            tmp[j] -= std::abs(tmp[i - 1][i - 1]) <= _epsilon ? static_cast<T>(0) : tmp[i - 1] * tmp[j][i - 1];
          }
        }
        tmp[N - 1] /= tmp[N - 1][N - 1];

        // null top triangle
        for (int i = N - 2; i >= 0; i--) {
          for (int j = i; j >= 0; j--) {
            tmp[j] -= std::abs(tmp[i + 1][i + 1]) <= _epsilon ? static_cast<T>(0) : tmp[i + 1] * tmp[j][i + 1];
          }
        }

        std::array<RowT, N> res;
        std::for_each(io.begin(), io.end(),
          [&tmp, &res](auto i) {
            auto [a, b] = stdx::split<N, N>(tmp[i]._data);
            res[i] = stdx::simd_cast<SimdImpl<T, N>>(b);
          });

        return res;
      }
#else
      // dummy
      constexpr Matr inversed() const noexcept {
        return {};
      }
#endif

      constexpr Matr & inverse() noexcept {
        *this = inversed();
        return *this;
      }

      static constexpr Matr identity() noexcept {
        return _identity;
      }

      static constexpr Matr4<T> rotate_x(const Radians<T> &rad) noexcept {;
        T co = std::cos(rad._data);
        T si = std::sin(rad._data);

        if (axis::x.x() < 0) {
          return Matr4<T> {
            1,  0,   0, 0,
            0, co, -si, 0,
            0, si,  co, 0,
            0,  0,   0, 1
          };
        }

        return Matr4<T> {
          1,   0,  0, 0,
          0,  co, si, 0,
          0, -si, co, 0,
          0,   0,  0, 1
        };
      }

      static constexpr Matr4<T> rotate_y(const Radians<T> &rad) noexcept {
        T co = std::cos(rad._data);
        T si = std::sin(rad._data);
        if (axis::y.y() < 0) {
          return Matr4<T> {
             co, 0, si, 0,
              0, 1,  0, 0,
            -si, 0, co, 0,
              0, 0,  0, 1
          };
        }

        return Matr4<T> {
          co, 0, -si, 0,
           0, 1,   0, 0,
          si, 0,  co, 0,
           0, 0,   0, 1
        };
      }

      static constexpr Matr4<T> rotate_z(const Radians<T> &rad) noexcept {
        T co = std::cos(rad._data);
        T si = std::sin(rad._data);

        // this is required for passing a test with our basis (z = {0, 0, -1})
        // TODO: consider possibility of swapping axes (maybe we should change these matrices when basis is set or just use quaternions)
        if (axis::z.z() < 0) {
          return Matr4<T> {
            co, -si, 0, 0,
            si,  co, 0, 0,
             0,   0, 1, 0,
             0,   0, 0, 1
          };
        }

        return Matr4<T> {
           co, si, 0, 0,
          -si, co, 0, 0,
            0,  0, 1, 0,
            0,  0, 0, 1
        };
      }

      constexpr bool operator==(const Matr &other) const noexcept {
        for (size_t i = 0; i < N; i++) {
          if (_data[i] != other._data[i]) {
            return false;
          }
        }
        return true;
      }

      constexpr bool equal(const Matr &other, ValueT eps = epsilon<ValueT>()) const noexcept {
        for (size_t i = 0; i < N; i++) {
          if (not _data[i].equal(other._data[i], eps)) {
            return false;
          }
        }
        return true;
      }

      friend std::ostream & operator<<(std::ostream &os, const Matr &m) noexcept {
        os << "\n(" << m[0] << ",\n";
          for (size_t i = 1; i < N - 1; i++)
            os << ' ' << m[i] << ",\n";
          os << ' ' << m[N - 1] << ')';
        return os;
      }

    private:
      static Matr get_identity() {
        std::array<RowT, N> id;
        constexpr auto io = std::ranges::iota_view {(size_t)0, N};

        std::transform(
          io.begin(), io.end(), id.begin(),
          [&io](size_t i) -> RowT {
            return SimdImpl<T, N>([i](size_t j) { return j == i ? 1 : 0; });
          });

        return id;
      }

    public:
      std::array<RowT, N> _data;

    private:
      static const Matr _identity;
      inline static const T _epsilon = std::numeric_limits<T>::epsilon();
    };

    // this is required to initialize 'Matr::_identity' on MSVC
    template <ArithmeticT T, std::size_t N>
      const Matr<T, N> Matr<T, N>::_identity = Matr<T, N>::get_identity();

    template <ArithmeticT T, std::size_t N>
      struct ScaleMatr {
        private:
          mr::Vec<T, N> _data = 1;

        public:
          constexpr ScaleMatr() = default;
          constexpr ScaleMatr(mr::Vec<T, N> scale) : _data(scale) { }

          constexpr ScaleMatr &inverse() noexcept {
            _data = 1 / _data;
            return *this;
          }
          constexpr ScaleMatr inversed() const noexcept {
            return ScaleMatr(1 / _data);
          }

          friend constexpr Matr<T, N> operator*(const Matr<T, N> &lhs, const ScaleMatr &rhs) noexcept {
            Matr<T, N> res = lhs;
            for (size_t i = 0; i < N; i++) {
              res[i] *= rhs._data[i];
            }
            return res;
          }
          friend constexpr Matr<T, N> & operator*=(Matr<T, N> &lhs, const ScaleMatr &rhs) noexcept {
            for (size_t i = 0; i < N; i++) {
              lhs[i] *= rhs._data[i];
            }
            return lhs;
          }

          friend constexpr ScaleMatr operator*(const ScaleMatr &lhs, const ScaleMatr &rhs) noexcept {
            ScaleMatr res = lhs;
            lhs._data *= rhs._data;
            return res;
          }
          friend constexpr ScaleMatr & operator*=(ScaleMatr &lhs, const ScaleMatr &rhs) noexcept {
            lhs._data *= rhs._data;
            return lhs;
          }

          friend constexpr Vec<T, N> operator*(const Vec<T, N> &lhs, const ScaleMatr &rhs) noexcept {
            Vec<T, N> res = lhs * rhs._data;
            return res;
          }
          friend constexpr Vec<T, N> & operator*=(Vec<T, N> &lhs, const ScaleMatr &rhs) noexcept {
            lhs *= rhs._data;
            return lhs;
          }

          operator Matr<T, N>() noexcept {
            auto res = Matr<T, N>::identity() * *this;
            return res;
          }
      };

    template <ArithmeticT T, std::size_t N>
      struct TranslateMatr {
        private:
          mr::Vec<T, N> _data {};

        public:
          constexpr TranslateMatr() = default;
          constexpr TranslateMatr(mr::Vec<T, N> translate) : _data(translate) { }

          operator Matr<T, N>() {
            Matr<T, N> res = Matr<T, N>::identity();
            res[N - 1] += _data;
            return res;
          }

          constexpr TranslateMatr &inverse() noexcept {
            _data = -_data;
            return *this;
          }
          constexpr TranslateMatr inversed() const noexcept {
            return TranslateMatr(-_data);
          }

          friend constexpr TranslateMatr operator*(const TranslateMatr &lhs, const TranslateMatr &rhs) noexcept {
            TranslateMatr res = lhs;
            res._data += rhs._data;
            return res;
          }
          friend constexpr TranslateMatr & operator*=(TranslateMatr &lhs, const TranslateMatr &rhs) noexcept {
            lhs._data += rhs._data;
            return lhs;
          }

          friend constexpr Matr<T, N> operator*(const Matr<T, N> &lhs, const TranslateMatr &rhs) noexcept {
            Matr<T, N> res = lhs;
            res[N - 1] += rhs._data;
            return res;
          }
          friend constexpr Matr<T, N> & operator*=(Matr<T, N> &lhs, const TranslateMatr &rhs) noexcept {
            lhs = lhs * rhs;
            return lhs;
          }

          friend constexpr Vec<T, N> operator*(const Vec<T, N> &lhs, const TranslateMatr &rhs) noexcept {
            Vec<T, N> res = lhs + rhs._data;
            return res;
          }
          friend constexpr Vec<T, N> & operator*=(Vec<T, N> &lhs, const TranslateMatr &rhs) noexcept {
            lhs += rhs._data;
            return lhs;
          }

          friend constexpr TranslateMatr operator*(const TranslateMatr &lhs, const ScaleMatr<T, N> &rhs) noexcept {
            TranslateMatr res = lhs;
            res._data *= rhs._data;
            return res;
          }
          friend constexpr TranslateMatr & operator*=(TranslateMatr &lhs, const ScaleMatr<T, N> &rhs) noexcept {
            lhs._data *= rhs._data;
            return lhs;
          }
      };

    template <ArithmeticT T>
      struct RotateMatr {
        private:
          mr::Quat<T> _data {};

        public:
          constexpr RotateMatr() = default;
          constexpr RotateMatr(mr::Quat<T> rotate) : _data(rotate) {}

          constexpr RotateMatr(mr::Radians<T> rad, mr::Norm3<T> v) noexcept
            : _data(rad, (mr::Vec3f)v) { }

          constexpr RotateMatr(mr::Radians<T> rad, mr::Vec3<T> v) noexcept
            : RotateMatr(rad, v.normalized_unchecked()) {}

          constexpr RotateMatr(mr::Yaw<T> angle) noexcept
            : RotateMatr(angle.value, mr::Norm3<T>(mr::unchecked, mr::axis::y)) { }

          constexpr RotateMatr(mr::Pitch<T> angle) noexcept
            : RotateMatr(angle.value, mr::Norm3<T>(mr::unchecked, mr::axis::x)) { }

          constexpr RotateMatr(mr::Roll<T> angle) noexcept
            : RotateMatr(angle.value, mr::Norm3<T>(mr::unchecked, mr::axis::z)) { }

          constexpr RotateMatr &inverse() noexcept {
            _data.inverse();
            return *this;
          }
          constexpr RotateMatr inversed() const noexcept {
            return RotateMatr(_data.inversed());
          }

          friend constexpr RotateMatr operator*(const RotateMatr &lhs, const RotateMatr &rhs) noexcept {
            RotateMatr res = lhs;
            res._data *= rhs._data;
            return res;
          }
          friend constexpr RotateMatr & operator*=(RotateMatr &lhs, const RotateMatr &rhs) noexcept {
            lhs._data *= rhs._data;
            return lhs;
          }

          template <std::size_t N>
            friend constexpr Matr<T, N> operator*(const Matr<T, N> &lhs, const RotateMatr &rhs) noexcept {
              return lhs * rhs._data;
            }
          template <std::size_t N>
            friend constexpr Matr<T, N> & operator*=(Matr<T, N> &lhs, const RotateMatr &rhs) noexcept {
              lhs = lhs * rhs;
              return lhs;
            }

          friend constexpr Vec3<T> operator*(const Vec3<T> &lhs, const RotateMatr &rhs) noexcept {
            T alpha = (mr::pi._data + rhs._data.w()) / 2;
            mr::Vec3<T> vec = rhs._data.vec().normalized_unchecked();

            auto vq = vec * std::cos(alpha);
            auto t = vq % lhs;
            auto u = std::sin(alpha) * t + vq % t;
            return {lhs + u + u};
          }

          friend constexpr Vec3<T> & operator*=(Vec3<T> &lhs, const RotateMatr &rhs) noexcept {
            lhs = lhs * rhs;
            return lhs;
          }
      };

    template <typename T, std::size_t N>
      constexpr ScaleMatr<T, N> scale(mr::Vec<T, N> v) noexcept {
        return ScaleMatr<T, N>(v);
      }

    template <typename T, std::size_t N>
      constexpr ScaleMatr<T, N> translate(mr::Vec<T, N> v) noexcept {
        return TranslateMatr<T, N>(v);
      }

    template <typename T>
      constexpr RotateMatr<T> rotate(mr::Quat<T> v) noexcept {
        return RotateMatr<T>(v);
      }

    template <typename T>
      constexpr RotateMatr<T> rotate(mr::Radians<T> rad, mr::Vec3<T> v) noexcept {
        return RotateMatr(rad, v);
      }

    template <typename T>
      constexpr RotateMatr<T> rotate(mr::Radians<T> rad, mr::Norm3<T> v) noexcept {
        return RotateMatr(rad, v);
      }

    template <typename T>
      constexpr RotateMatr<T> rotate(mr::Yaw<T> rad) noexcept {
        return RotateMatr(rad);
      }

    template <typename T>
      constexpr RotateMatr<T> rotate(mr::Pitch<T> rad) noexcept {
        return RotateMatr(rad);
      }

    template <typename T>
      constexpr RotateMatr<T> rotate(mr::Roll<T> rad) noexcept {
        return RotateMatr(rad);
      }
} // namespace mr

#ifdef __cpp_lib_format
// std::format support
namespace std {
  template <mr::ArithmeticT T, size_t N>
    struct formatter<mr::Matr<T, N>> {
      template<typename ParseContext>
        constexpr auto parse(ParseContext& ctx) {
          // skip all format specifiers
          auto it = ctx.begin();
          while (*it != '}')
              ++it;
          return it;
        }

      template<typename FmtContext>
        auto format(const mr::Matr<T, N> &m, FmtContext& ctx) const {
          ostringstream out;
          out << m;
          return ranges::copy(std::move(out).str(), ctx.out()).out;
        }
    };
} // namespace std
#endif

#endif // __MR_MATR_HPP_
