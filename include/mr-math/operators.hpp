#ifndef __operators_units_hpp_
#define __operators_units_hpp_

#include "../def.hpp"

namespace mr {
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
      operator*(const DerivedT &lhs, const ArithmeticT auto rhs) noexcept {
        return DerivedT{lhs._data * static_cast<DerivedT::ValueT>(rhs)};
      }

      friend constexpr DerivedT
      operator/(const DerivedT &lhs, const ArithmeticT auto rhs) noexcept {
        return DerivedT{lhs._data / static_cast<DerivedT::ValueT>(rhs)};
      }

      friend constexpr DerivedT
      operator*(const ArithmeticT auto lhs, const DerivedT &rhs) noexcept {
        return DerivedT{rhs._data * static_cast<DerivedT::ValueT>(lhs)};
      }

      friend constexpr DerivedT &
      operator*=(DerivedT &lhs, const ArithmeticT auto x) noexcept {
        lhs._data *= static_cast<DerivedT::ValueT>(x);
        return lhs;
      }

      friend constexpr DerivedT &
      operator/=(DerivedT &lhs, const ArithmeticT auto x) noexcept {
        lhs._data /= static_cast<DerivedT::ValueT>(x);
        return lhs;
      }

      friend constexpr DerivedT
      operator-(const DerivedT &rhs) noexcept {
        return DerivedT{-rhs._data};
      }
    };
}

#endif // __operators_units_hpp_
