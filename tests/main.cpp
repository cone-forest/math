#include <array>

#include "gtest/gtest.h"
#include "mr-math/math.hpp"
#include "mr-math/units.hpp"

using namespace mr::literals;

class Vector3DTest : public ::testing::Test {
protected:
  mr::Vec3f v1{1.0, 2.0, 3.0};
  mr::Vec3f v2{4.0, 5.0, 6.0};
};

TEST_F(Vector3DTest, Constructors) {
  EXPECT_EQ(mr::Vec3f(), mr::Vec3f(0, 0, 0));
  EXPECT_EQ(mr::Vec3f(1), mr::Vec3f(1, 1, 1));
  EXPECT_EQ(mr::Vec3f(mr::Vec2f(1, 2)), mr::Vec3f(1, 2, 0));
  // TODO: EXPECT_EQ(mr::Vec3f(mr::Vec2f(1, 2), 3), mr::Vec3f(1, 2, 3));
  EXPECT_EQ(mr::Vec3f(mr::Vec4f(1, 2, 3, 4)), mr::Vec3f(1, 2, 3));

  EXPECT_EQ(mr::Vec3f(std::span<const int, 2>{{1, 2}}), mr::Vec3f(1, 2, 0));
  EXPECT_EQ(mr::Vec3f(std::span<const int, 3>{{1, 2, 3}}), mr::Vec3f(1, 2, 3));
  EXPECT_EQ(mr::Vec3f(std::span<const int, 4>{{1, 2, 3, 4}}), mr::Vec3f(1, 2, 3));
  EXPECT_EQ(mr::Vec3f(std::span<const int>{{1, 2, 3}}), mr::Vec3f(1, 2, 3));
}

TEST_F(Vector3DTest, Getters) {
  EXPECT_EQ(v1.x(), 1.0);
  EXPECT_EQ(v1.y(), 2.0);
  EXPECT_EQ(v1.z(), 3.0);
  EXPECT_EQ(v1[0], 1.0);
  EXPECT_EQ(v1[1], 2.0);
  EXPECT_EQ(v1[2], 3.0);
}

TEST_F(Vector3DTest, Setters) {
  mr::Vec3f v{0, 0, 3};
  v.x(1);
  v.set(1, 2);
  EXPECT_EQ(v, mr::Vec3f(1, 2, 3));
}

TEST_F(Vector3DTest, Equality) {
  auto copy = v1;
  EXPECT_EQ(v1, copy);
  EXPECT_TRUE(v1.equal(copy));
  EXPECT_TRUE(equal(v1, copy));

  EXPECT_NE(v1, v2);
  EXPECT_FALSE(v1.equal(v2));
  EXPECT_FALSE(equal(v1, v2));
}

TEST_F(Vector3DTest, Addition) {
  mr::Vec3f result = v1 + v2;
  EXPECT_EQ(result, mr::Vec3f(5, 7, 9));
}

TEST_F(Vector3DTest, Subtraction) {
  mr::Vec3f result = v1 - v2;
  EXPECT_EQ(result, mr::Vec3f(-3));
}

TEST_F(Vector3DTest, DotProduct) {
  auto result = v1.dot(v2);
  EXPECT_EQ(result, 32.0);
}

TEST_F(Vector3DTest, CrossProduct) {
  auto result1 = v1.cross(v2);
  mr::Vec3f expected{-3, 6, -3};
  EXPECT_EQ(result1, expected);

  auto result2 = v2.cross(v1);
  EXPECT_EQ(result2, -expected);
}

TEST_F(Vector3DTest, ComponentsProduct) {
  EXPECT_EQ(v1 * v2, mr::Vec3f(4, 10, 18));
}

TEST_F(Vector3DTest, Length) {
  EXPECT_NEAR(v1.length(), std::sqrt(14.0f), 0.0001);
  EXPECT_NEAR(v1.length2(), 14.0f, 0.0001);
  EXPECT_NEAR(v1.inversed_length(), 1 / std::sqrt(14.0f), 0.01);
}

TEST_F(Vector3DTest, Normalize) {
  mr::Vec3f expected{0.267261, 0.534522, 0.801784};
  EXPECT_TRUE(mr::equal(v1.normalized_unchecked(), expected, 0.000001));
  auto copy = v1;
  EXPECT_TRUE(mr::equal(copy.normalize_unchecked(), expected, 0.000001));

  EXPECT_TRUE(mr::equal(v1.normalized_fast_unchecked(), expected, 0.1));
  EXPECT_TRUE(mr::equal(copy.normalize_fast_unchecked(), expected, 0.1));

  auto zero_v = mr::Vec3f{0};
  EXPECT_EQ(zero_v.normalize(), zero_v);
  auto null = zero_v.normalized();
  EXPECT_FALSE(null.has_value());
}

TEST_F(Vector3DTest, Abs) {
  mr::Vec3f v{-30, 47, -80};
  EXPECT_EQ(v.absed(), mr::Vec3f(30, 47, 80));
  EXPECT_EQ(v.abs(), mr::Vec3f(30, 47, 80));
}

TEST_F(Vector3DTest, Clamp) {
  mr::Vec3f v{-30, 47, -80};
  EXPECT_EQ(v.clamped(-47, 0), mr::Vec3f(-30, 0, -47));
  EXPECT_EQ(v.clamp(-47, 0), mr::Vec3f(-30, 0, -47));
}

class MatrixTest : public ::testing::Test {
protected:
  mr::Matr4f m1 {
    1,  2,  3,  4 ,
    5,  6,  7,  8 ,
    9,  10, 11, 12,
    13, 14, 15, 16
  };
  mr::Matr4f m2 {
    17, 18, 19, 20,
    21, 22, 23, 24,
    25, 26, 27, 28,
    29, 30, 31, 32
  };
};

TEST_F(MatrixTest, Getters) {
  EXPECT_EQ(m1[0][0],  1.0);
  EXPECT_EQ(m1[0][1],  2.0);
  EXPECT_EQ(m1[0][2],  3.0);
  EXPECT_EQ(m1[0][3],  4.0);
  EXPECT_EQ(m1[1][0],  5.0);
  EXPECT_EQ(m1[1][1],  6.0);
  EXPECT_EQ(m1[1][2],  7.0);
  EXPECT_EQ(m1[1][3],  8.0);
  EXPECT_EQ(m1[2][0],  9.0);
  EXPECT_EQ(m1[2][1], 10.0);
  EXPECT_EQ(m1[2][2], 11.0);
  EXPECT_EQ(m1[2][3], 12.0);
  EXPECT_EQ(m1[3][0], 13.0);
  EXPECT_EQ(m1[3][1], 14.0);
  EXPECT_EQ(m1[3][2], 15.0);
  EXPECT_EQ(m1[3][3], 16.0);
}

TEST_F(MatrixTest, Equality) {
  mr::Matr4f copy = m1;
  EXPECT_EQ(m1, copy);
  EXPECT_TRUE(m1.equal(copy));
  EXPECT_TRUE(equal(m1, copy));

  EXPECT_NE(m1, m2);
  EXPECT_FALSE(m1.equal(m2));
  EXPECT_FALSE(equal(m1, m2));
}

TEST_F(MatrixTest, Addition) {
  mr::Matr4f result = m1 + m2;
  mr::Matr4f expected {
    18, 20, 22, 24,
    26, 28, 30, 32,
    34, 36, 38, 40,
    42, 44, 46, 48
  };

  EXPECT_TRUE(equal(result, expected));
}

TEST_F(MatrixTest, Multiplication) {
  mr::Matr4f result = m1 * m2;
  mr::Matr4f expected {
    250,  260,  270,  280,
    618,  644,  670,  696,
    986,  1028, 1070, 1112,
    1354, 1412, 1470, 1528
  };

   EXPECT_EQ(result, expected);
}

TEST_F(MatrixTest, Transposition) {
  mr::Matr4f expected {
    1, 5, 9,  13,
    2, 6, 10, 14,
    3, 7, 11, 15,
    4, 8, 12, 16
  };
  EXPECT_EQ(m1.transposed(), expected);

  mr::Matr4f copy = m1;
  EXPECT_EQ(copy.transpose(), expected);
}

TEST_F(MatrixTest, Identity) {
  mr::Matr4f expected {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  };
  EXPECT_EQ(mr::Matr4f::identity(), expected);
}

TEST_F(MatrixTest, ScaleVector) {
  EXPECT_EQ(mr::Vec3f(1, -1, 0) * mr::ScaleMatr3f({30, 47, 80}), mr::Vec3f(30, -47, 0));
}

TEST_F(MatrixTest, TranslateVector) {
  EXPECT_EQ(mr::Vec3f(0, 0, 22) * mr::TranslateMatr3f({30, 47, 80}), mr::Vec3f(30, 47, 102));
}

TEST_F(MatrixTest, RotateBasis) {
  mr::axis::x = {1, 0, 0};
  mr::axis::y = {0, 1, 0};
  mr::axis::z = {0, 0, 1};
  EXPECT_TRUE(mr::equal(mr::axis::x * mr::Matr4f::rotate_y(-90_deg), mr::axis::z));
  EXPECT_TRUE(mr::equal(mr::axis::y * mr::Matr4f::rotate_z(-90_deg), mr::axis::x));
  EXPECT_TRUE(mr::equal(mr::axis::z * mr::Matr4f::rotate_x(-90_deg), mr::axis::y));

  mr::axis::x = {1, 0, 0};
  mr::axis::y = {0, 1, 0};
  mr::axis::z = {0, 0, -1};
  EXPECT_TRUE(mr::equal(mr::axis::x * mr::Matr4f::rotate_y(90_deg), mr::axis::z));
  EXPECT_TRUE(mr::equal(mr::axis::y * mr::Matr4f::rotate_z(90_deg), mr::axis::x));
  EXPECT_TRUE(mr::equal(mr::axis::z * mr::Matr4f::rotate_x(90_deg), mr::axis::y));
}

TEST_F(MatrixTest, RotateVector) {
  mr::Vec3f v{30, 47, 80};
  mr::Vec3f expected{38.340427, 81.678845, 36.980571};
  EXPECT_TRUE(mr::equal(v * mr::rotate(mr::Radiansf(102_deg), mr::Norm3f(1, 1, 1)), expected, 0.0001));
}

class QuaternionTest : public ::testing::Test {
protected:
  mr::Quat<float> q1 {mr::Degreesf(90), 1, 0, 0};
};

TEST_F(QuaternionTest, DefaultConstructor) {
  mr::Quat<float> q;
  EXPECT_EQ((mr::Vec4f)q, mr::Vec4f());
}

TEST_F(QuaternionTest, ParameterizedConstructor) {
    float w = 1.0, x = 2.0, y = 3.0, z = 4.0;
    mr::Quat<float> p(mr::Radiansf(w), x, y, z);
    EXPECT_EQ(p.w(), w);
    EXPECT_EQ(p.vec(), mr::Vec3f(x, y, z));
}

TEST_F(QuaternionTest, Multiplication) {
    mr::Quat<float> a(mr::Radiansf(1), 2, 3, 4);
    mr::Quat<float> b(mr::Radiansf(5), 6, 7, 8);
    mr::Quat<float> res = a * b;
    mr::Quat<float> expected(mr::Radiansf(-60), 12, 30, 24);
    EXPECT_TRUE(mr::equal(res.vec(), expected.vec()));
    EXPECT_TRUE(mr::equal(res.w(), expected.w()));
}

TEST_F(QuaternionTest, Addition) {
    mr::Quat<float> c(mr::Radiansf(1), 2, 3, 4);
    mr::Quat<float> d(mr::Radiansf(5), 6, 7, 8);
    mr::Quat<float> res = c + d;
    mr::Quat<float> sum(mr::Radiansf(6), 8, 10, 12);
    EXPECT_TRUE(mr::equal(res.vec(), sum.vec()));
    EXPECT_TRUE(mr::equal(res.w(), sum.w()));
}

TEST_F(QuaternionTest, Subtraction) {
    mr::Quat<float> e(mr::Radiansf(1), 2, 3, 4);
    mr::Quat<float> f(mr::Radiansf(5), 6, 7, 8);
    mr::Quat<float> diff(mr::Radiansf(-4), -4, -4, -4);
    mr::Quat<float> res = e - f;
    EXPECT_TRUE(mr::equal(res.vec(), diff.vec()));
    EXPECT_TRUE(mr::equal(res.w(), diff.w()));
}

TEST_F(QuaternionTest, Normalize) {
    mr::Quat<float> g(mr::Radiansf(3), 4, 0, 0);
    g.normalize();
    EXPECT_TRUE(mr::equal(g.w(), 0.6));
    EXPECT_TRUE(mr::equal(g.vec(), mr::Vec3f(0.8, 0, 0)));
}

TEST_F(QuaternionTest, RotateMatrix) {
  mr::Vec3f v {0, 1, 0};
  mr::Vec3f expected {0, 0, -1};
  EXPECT_TRUE(mr::equal(v * mr::rotate(q1), expected));
}

// TODO: camera tests

TEST(ColorTest, Constructors) {
  EXPECT_EQ(mr::Color(), mr::Color(0, 0, 0, 0));

  const mr::Color expected1{0.3, 0.47, 0.8, 1.0};
  EXPECT_EQ(mr::Color(0.3, 0.47, 0.8), expected1);
  EXPECT_EQ(mr::Color(mr::Vec4f(0.3, 0.47, 0.8, 1)), expected1);

  const mr::Color expected2{0.2980392156862745, 0.4666666666666667, 0.8, 1.0};
  EXPECT_EQ(mr::Color(76, 119, 204, 255), expected2);
  EXPECT_EQ(mr::Color(0x4C'77'CC'FF), expected2);
  EXPECT_EQ(0x4C'77'CC'FF_rgba, expected2);
}

TEST(ColorTest, Formats) {
  const auto color = 0x4C'77'CC'FF_rgba;
  EXPECT_EQ(color.argb(), 0xFF'4C'77'CC_rgba);
  EXPECT_EQ(color.bgra(), 0xCC'77'4c'FF_rgba);
  EXPECT_EQ(color.abgr(), 0xFF'CC'77'4c_rgba);
}

TEST(ColorTest, Getters) {
  const auto color = 0x4C'77'CC'FF_rgba;
  EXPECT_FLOAT_EQ(color.r(), 0.2980392156862745f);
  EXPECT_FLOAT_EQ(color.g(), 0.4666666666666667f);
  EXPECT_FLOAT_EQ(color.b(), 0.8);
  EXPECT_FLOAT_EQ(color.a(), 1.0);

  EXPECT_EQ(color[0], color.r());
  EXPECT_EQ(color[1], color.g());
  EXPECT_EQ(color[2], color.b());
  EXPECT_EQ(color[3], color.a());

  const auto[r, g, b, a] = color;
  EXPECT_EQ(r, color.r());
  EXPECT_EQ(g, color.g());
  EXPECT_EQ(b, color.b());
  EXPECT_EQ(a, color.a());
}

TEST(ColorTest, Setters) {
  auto color = 0x4C'77'CC'FF_rgba;
  color.r(1.0);
  color.set(1, 0.5);
  EXPECT_EQ(color, mr::Color(1.0, 0.5, 0.8, 1.0));
}

TEST(ColorTest, Equality) {
  const auto color1 = 0x4C'77'CC'FF_rgba;
  const auto copy = color1;
  EXPECT_EQ(color1, copy);
  EXPECT_TRUE(color1.equal(copy));
  EXPECT_TRUE(equal(color1, copy));

  const auto color2 = 0x00'00'00'00_rgba;
  EXPECT_NE(color1, color2);
  EXPECT_FALSE(color1.equal(color2));
  EXPECT_FALSE(equal(color1, color2));
}

TEST(ColorTest, Addition) {
  // Values can exeed 1.0 (should they?)
  EXPECT_EQ(mr::Color(1.0, 0.0, 0.5, 1.0) + mr::Color(0.0, 1.0, 0.5, 1.0), mr::Color(1.0, 1.0, 1.0, 2.0));
}

TEST(UtilityTest, Within) {
  EXPECT_FALSE(mr::within(1, 10)(0));
  EXPECT_TRUE(mr::within(1, 10)(1));
  EXPECT_TRUE(mr::within(1, 10)(5));
  EXPECT_TRUE(mr::within(1, 10)(10));
  EXPECT_FALSE(mr::within(1, 10)(11));

  EXPECT_FALSE(mr::within_ex(1, 10)(0));
  EXPECT_FALSE(mr::within_ex(1, 10)(1));
  EXPECT_TRUE(mr::within_ex(1, 10)(5));
  EXPECT_FALSE(mr::within_ex(1, 10)(10));
  EXPECT_FALSE(mr::within_ex(1, 10)(11));

  EXPECT_TRUE(mr::within(1., 10.f)(5));
  EXPECT_TRUE(mr::within_ex(1., 10.f)(5));
}
