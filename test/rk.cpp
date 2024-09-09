#include "pch.h"
#include "nu/nu.h"
#include <limits>
#include <random>

#define IMPL_RK45_INI_TEST(name, INITIAL, f, F, ABS_ERROR, N_STEPS)                                                      \
    TEST(Rk45, name)                                                                                                     \
    {                                                                                                                    \
        using namespace nu;                                                                                              \
                                                                                                                         \
        ei::VectorXd state = INITIAL;                                                                                    \
        Rk45 rk(state);                                                                                                  \
                                                                                                                         \
        for (int i = 0; i < N_STEPS; i++)                                                                                \
        {                                                                                                                \
            rk.step(f);                                                                                                  \
            ei::VectorXd expected = F(rk.t);                                                                             \
            for (int j = 0; j < state.rows(); j++)                                                                       \
            {                                                                                                            \
                const double tolerance = std::max(1.0, std::abs(expected(j)) * (i + 1)) * ABS_ERROR;                     \
                ASSERT_NEAR(state(j), expected(j), tolerance) << "t: " << rk.t << ", step: " << i << ", rk.h: " << rk.h; \
            }                                                                                                            \
        }                                                                                                                \
    }

#define IMPL_RK45_TEST(name, N, f, F, ABS_ERROR, N_STEPS) \
    IMPL_RK45_INI_TEST(name, ei::VectorXd::Zero(N, 1), f, F, ABS_ERROR, N_STEPS)

ei::VectorXd x(double x, const ei::VectorXd& _)
{
    return ei::Vector<double, 1>{x};
}

ei::VectorXd half_x_square(double x)
{
    return ei::Vector<double, 1>{0.5 * x * x};
}

IMPL_RK45_TEST(x, 1, x, half_x_square, 10.0 * std::numeric_limits<double>::epsilon(), 1000);

ei::VectorXd cos_x(double x, const ei::VectorXd& _)
{
    return ei::Vector<double, 1>{cos(x)};
}

ei::VectorXd sin_x(double x)
{
    return ei::Vector<double, 1>{sin(x)};
}

IMPL_RK45_TEST(cos, 1, cos_x, sin_x, std::numeric_limits<float>::epsilon(), 1000);

ei::VectorXd zero_f(double x, const ei::VectorXd& _)
{
    return ei::Vector<double, 1>{0.0};
}

ei::VectorXd zero_F(double x)
{
    return ei::Vector<double, 1>{0.0};
}

IMPL_RK45_TEST(zero, 1, zero_f, zero_F, std::numeric_limits<float>::epsilon(), 1000);

ei::VectorXd kf(double x, const ei::VectorXd& f)
{
    return f;
}

ei::VectorXd exp_F(double x)
{
    return ei::Vector<double, 1>{exp(x)};
}

IMPL_RK45_INI_TEST(exp, (ei::Vector<double, 1>{1.0}), kf, exp_F, std::numeric_limits<float>::epsilon(), 1000);
