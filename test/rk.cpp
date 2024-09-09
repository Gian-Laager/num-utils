#include "Eigen/Core"
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

ei::VectorXd neg_f_squared(double x, const ei::VectorXd& f)
{
    return -ei::Vector<double, 1>{f(0) * f(0)};
}

ei::VectorXd non_lin_sol(double x)
{
    return ei::Vector<double, 1>{1.0 / (1.0 + x)};
}

IMPL_RK45_INI_TEST(NonLinear, (ei::Vector<double, 1>{1.0}), neg_f_squared, non_lin_sol, std::numeric_limits<float>::epsilon(), 1000);

ei::VectorXd coupled_linear(double t, const ei::VectorXd& f)
{
    double deg45 = std::numbers::pi / 4.0;
    ei::Matrix3d rot45x;
    rot45x << 1.0, 0.0, 0.0,
        0.0, cos(deg45), -sin(deg45),
        0.0, sin(deg45), cos(deg45);
    ei::Matrix3d rot45y;
    rot45y << cos(deg45), 0.0, -sin(deg45),
        0.0, 1.0, 0.0,
        sin(deg45), 0.0, cos(deg45);
    ei::Matrix3d lambda = ei::Matrix3d{
        {-2.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, -3.0}};
    ei::Matrix3d a = rot45x * rot45y * lambda * (rot45x * rot45y).transpose();

    return a * f;
}

ei::Vector3d coupled_linear_sol(double t)
{
    double deg45 = std::numbers::pi / 4.0;
    ei::Matrix3d rot45x;
    rot45x << 1.0, 0.0, 0.0,
        0.0, cos(deg45), -sin(deg45),
        0.0, sin(deg45), cos(deg45);
    ei::Matrix3d rot45y;
    rot45y << cos(deg45), 0.0, -sin(deg45),
        0.0, 1.0, 0.0,
        sin(deg45), 0.0, cos(deg45);
    ei::Matrix3d lambda = ei::Matrix3d{
        {exp(-2.0 * t), 0.0, 0.0},
        {0.0, exp(1.0 * t), 0.0},
        {0.0, 0.0, exp(-3.0 * t)}};
    ei::Matrix3d a = rot45x * rot45y * lambda * (rot45x * rot45y).transpose();

    return a * ei::Vector3d{1.0, 1.0, 1.0};
}

IMPL_RK45_INI_TEST(CoupledLinear, (ei::Vector3d::Ones()), coupled_linear, coupled_linear_sol, std::numeric_limits<float>::epsilon(), 1000);
