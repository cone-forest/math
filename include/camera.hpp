#ifndef __camera_hpp_
#define __camera_hpp_

#include "def.hpp"
#include "vec.hpp"
#include "rot.hpp"
#include "matr.hpp"

namespace mr {

  template <std::floating_point T = float>
    class [[nodiscard]] Camera {
      public:
        struct Projection {
          friend class Camera;
        public:
          T distance = 0.1f;
          T far = (1 << 10);
          T size = 0.1;

          T height = size;
          T width = size;

        private:
          // cached frustum matrix
          mutable bool frustum_calculated = false;
          mutable Matr4<T> frustum;

          // cached ortholinear matrix
          mutable bool ortholinear_calculated = false;
          mutable Matr4<T> ortholinear;
        };

        constexpr Camera() = default;
        constexpr Camera(Vec3<T> position) : _position(position) {}
        constexpr Camera(Vec3<T> position, Vec3<T> direction, Vec3<T> up = {0, 1, 0}) :
          _position(position),
          _rotation(
              direction.normalized(),
              up.normalized(),
              (direction % up).normalized()) {}

        // copy semantics
        constexpr Camera(const Camera &other) noexcept = default;
        constexpr Camera & operator=(const Camera &other) noexcept = default;

        // move semantics
        constexpr Camera(Camera &&other) noexcept = default;
        constexpr Camera & operator=(Camera &&other) noexcept = default;

        // position delta
        constexpr Camera & operator+=(Vec3<T> position_delta) noexcept {
          _perspective_calculated = false;
          _position += position_delta;
        }

        // angle in radians
        constexpr Camera & operator+=(Pitch<T> angle_rad) noexcept {
          _perspective_calculated = false;
          _rotation += angle_rad;

          return *this;
        }

        // angle in radians
        constexpr Camera & operator+=(Yaw<T> angle_rad) noexcept {
          _perspective_calculated = false;
          _rotation += angle_rad;

          return *this;
        }

        // angle in radians
        constexpr Camera & operator+=(Roll<T> angle_rad) noexcept {
          _perspective_calculated = false;
          _rotation += angle_rad;

          return *this;
        }

        constexpr Vec3<T> position() const noexcept {
          return _position;
        }

        constexpr void position(Vec3<T> pos) noexcept {
          _perspective_calculated = false;
          _position = pos;
        }

        constexpr Vec3<T> direction() const noexcept {
          return _rotation[0];
        }

        constexpr void direction(Vec3<T> dir) noexcept {
          _perspective_calculated = false;
          _rotation[0] = dir;
        }

        constexpr Projection & projection() noexcept {
          _perspective_calculated = false;
          return _projection;
        }

        constexpr const Projection & projection() const noexcept {
          return _projection;
        }

        constexpr Matr4<T> perspective() const noexcept {
          if (_perspective_calculated) [[likely]] {
            return _perspective;
          }
          std::lock_guard lg(_perspective_mutex);

          auto direction = _rotation[0];
          auto right = _rotation[1];
          auto up = _rotation[2];
          _perspective_calculated = true;
          _perspective = mr::Matr4<T>{
            typename mr::Matr4<T>::RowT{right[0], up[0], direction[0], 0},
            typename mr::Matr4<T>::RowT{right[1], up[1], direction[1], 0},
            typename mr::Matr4<T>::RowT{right[2], up[2], direction[2], 0},
            typename mr::Matr4<T>::RowT{-(_position & right), -(_position & up), (_position & direction), 1}
          };
          return _perspective;
        }

        constexpr Matr4<T> ortholinear() const noexcept {
          if (_projection.ortholinear_calculated) [[likely]] {
            return _projection.ortholinear;
          }
          std::lock_guard lg(_perspective_mutex);

          const T l = -_projection.height / 2; // left
          const T r = _projection.height / 2;  // right
          const T b = -_projection.width / 2;  // bottom
          const T t = _projection.width / 2;   // top
          const T n = _projection.distance;    // near
          const T f = _projection.far;         // far

          _projection.ortholinear_calculated = true;
          _projection.ortholinear = mr::Matr4<T>{
            typename mr::Matr4<T>::RowT(2 / (r - l), 0, 0, 0),
            typename mr::Matr4<T>::RowT(0, 2 / (t - b), 0, 0),
            typename mr::Matr4<T>::RowT(0, 0, 2 / (n - f), 0),
            typename mr::Matr4<T>::RowT((r + l) / (l - r), (t + b) / (b - t), (f + n) / (n - f), 1)
          };
          return _projection.ortholinear;
        }

        constexpr Matr4<T> frustum() const noexcept {
          if (_projection.frustum_calculated) [[likely]] {
            return _projection.frustum;
          }
          std::lock_guard lg(_perspective_mutex);

          const T l = -_projection.height / 2; // left
          const T r = _projection.height / 2;  // right
          const T b = -_projection.width / 2;  // bottom
          const T t = _projection.width / 2;   // top
          const T n = _projection.distance;    // near
          const T f = _projection.far;         // far

          _projection.frustum_calculated = true;
          _projection.frustum = mr::Matr4<T>{
            typename mr::Matr4<T>::RowT(2 * n / (r - l), 0, 0, 0),
            typename mr::Matr4<T>::RowT(0, 2 * n / (t - b), 0, 0),
            typename mr::Matr4<T>::RowT((r + l) / (r - l), (t + b) / (t - b), (f + n) / (n - f), -1),
            typename mr::Matr4<T>::RowT(0, 0, 2 * n * f / (n - f), 0)
          };
          return _projection.frustum;
        }

      private:
        Vec3<T> _position;
        Rotation<T> _rotation;
        Projection _projection;

        mutable std::mutex _perspective_mutex;
        mutable std::atomic_bool _perspective_calculated = false;
        mutable Matr4<T> _perspective;
    };
}

#endif // __camera_hpp_
