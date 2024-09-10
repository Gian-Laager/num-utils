#include "pch.h"
#include <cmath>
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
