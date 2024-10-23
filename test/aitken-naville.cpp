#include "nu/aitken-naville.h"
#include <gtest/gtest.h>
#include <random>

using namespace nu;

TEST(AitkenNaville, Constant)
{
    AitkenNaville<3> interp({1.0, 2.0, 3.0});
    interp.add_point(0.0, 1.0);
    ei::Vector3d res = interp.eval();

    ASSERT_NEAR((res - ei::Vector3d::Ones()).norm(), 0.0, 1e-10) << res;
}

TEST(AitkenNaville, ConstantXd)
{
    AitkenNavilleXd interp(ei::Vector3d{1.0, 2.0, 3.0});
    interp.add_point(0.0, 1.0);
    ei::Vector3d res = interp.eval();

    ASSERT_NEAR((res - ei::Vector3d::Ones()).norm(), 0.0, 1e-10) << res;
}

TEST(AitkenNaville, Line)
{
    auto f = [](double x)
    { return x; };
    ei::Vector3d xs{1.0, 2.0, 3.0};
    AitkenNaville<3> interp(xs);
    interp.add_point(0.0, f(0.0));
    interp.add_point(1.0, f(1.0));

    ei::Vector3d res = interp.eval();
    ASSERT_NEAR((res - xs.unaryExpr(f)).norm(), 0.0, 1e-10) << res;
}

TEST(AitkenNaville, LineXd)
{
    auto f = [](double x)
    { return x; };
    ei::Vector3d xs{1.0, 2.0, 3.0};
    AitkenNavilleXd interp(xs);
    interp.add_point(0.0, f(0.0));
    interp.add_point(1.0, f(1.0));

    ei::Vector3d res = interp.eval();
    ASSERT_NEAR((res - xs.unaryExpr(f)).norm(), 0.0, 1e-10) << res;
}

TEST(AitkenNaville, Ploynomial1)
{
    constexpr int NSAMPLES = 3;
    auto f = [](double x)
    { return x * x + 1; };
    ei::Vector3d xs{1.0, 2.0, 3.0};

    AitkenNaville<3> interp(xs);
    std::mt19937 gen(30531);
    std::uniform_real_distribution<double> sampleDistrib(0.0, 1.0);

    for (int i = 0; i < NSAMPLES; i++)
    {
        double t = sampleDistrib(gen);
        interp.add_point(t, f(t));
    }

    ei::Vector3d res = interp.eval();
    ASSERT_NEAR((res - xs.unaryExpr(f)).norm(), 0.0, 1e-10) << res;
}

TEST(AitkenNaville, Ploynomial2)
{
    constexpr int NSAMPLES = 5;
    auto f = [](double x)
    { return x * x + 1 - 12 * x - x * x * x + 17 * x * x * x * x; };
    ei::Vector3d xs{1.0, 2.0, 3.0};

    AitkenNaville<3> interp(xs);
    std::mt19937 gen(107);
    std::uniform_real_distribution<double> sampleDistrib(0.0, 1.0);

    for (int i = 0; i < NSAMPLES; i++)
    {
        double t = sampleDistrib(gen);
        interp.add_point(t, f(t));
    }

    ei::Vector3d res = interp.eval();
    ASSERT_NEAR((res - xs.unaryExpr(f)).norm(), 0.0, 1e-10) << res;
}
