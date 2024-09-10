#include "pch.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include "nu/polynomials.h"

TEST(Poly, Eval)
{
    using namespace nu;

    Poly2d p{1.0, 0.0, 1.0}; // 1 + x^2

    ASSERT_EQ(p.eval(1.0), 2.0);
    ASSERT_EQ(p.eval(-1.0), 2.0);
    ASSERT_EQ(p.eval(0.0), 1.0);
}

TEST(Quadrature, GaussLegendre)
{
    using namespace nu;
    using namespace std::placeholders;

    Poly2d p{1.0, 0.0, 1.0}; // 1 + x^2
    Poly3d result{0.0, 1.0, 0.0, 1.0 / 3.0};

    ASSERT_NEAR(Quadrature<double>::integ_over_unit_ball(std::bind(&Poly2d::eval, &p, _1), 2),
                result.eval(1.0) - result.eval(-1.0),
                2.0 * std::numeric_limits<double>::epsilon());
}
