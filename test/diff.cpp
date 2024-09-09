#include "pch.h"
#include "nu/nu.h"
#include <random>

#define IMPL_TEST_DERIVATIVE(name, DiffMethod, Ord, Epsilon)                                     \
    TEST(name, Order##Ord)                                                                       \
    {                                                                                            \
        constexpr size_t N_FUNCS = 6;                                                            \
        auto f = [](double x)                                                                    \
        {                                                                                        \
            return ei::Vector<double, N_FUNCS>{                                                  \
                2.0,                                                                             \
                x * x,                                                                           \
                x * x + 0.5 * x + 1,                                                             \
                x * x * x + 0.5 * x + 1,                                                         \
                sin(x), log(x)};                                                                 \
        };                                                                                       \
                                                                                                 \
        auto df = [](double x)                                                                   \
        {                                                                                        \
            return ei::Vector<double, N_FUNCS>{                                                  \
                0.0,                                                                             \
                2.0 * x,                                                                         \
                2.0 * x + 0.5,                                                                   \
                3.0 * x * x + 0.5,                                                               \
                cos(x), 1.0 / x};                                                                \
        };                                                                                       \
                                                                                                 \
        constexpr unsigned seed = 42;                                                            \
        std::mt19937 generator(seed);                                                            \
                                                                                                 \
        std::uniform_real_distribution<double> distrib{1e-2, 100};                               \
                                                                                                 \
        for (int i = 0; i < 5000; i++)                                                           \
        {                                                                                        \
            double x = distrib(generator);                                                       \
            auto expected = df(x);                                                               \
            ei::Vector<double, N_FUNCS> numeric = nu::DiffMethod<Ord>::deriv(f, x);            \
                                                                                                 \
            for (int j = 0; j < numeric.rows(); j++)                                             \
            {                                                                                    \
                const double tolerance = Epsilon * std::max<double>(1.0, std::abs(expected(j))); \
                ASSERT_NEAR(numeric(j), expected(j), tolerance);                                 \
            }                                                                                    \
        }                                                                                        \
    }

IMPL_TEST_DERIVATIVE(DiffForward, ForwardDifference, 1, 1e-3);
IMPL_TEST_DERIVATIVE(DiffForward, ForwardDifference, 2, 1e-4);
IMPL_TEST_DERIVATIVE(DiffForward, ForwardDifference, 3, 1e-5);
IMPL_TEST_DERIVATIVE(DiffForward, ForwardDifference, 4, 1e-5);
IMPL_TEST_DERIVATIVE(DiffForward, ForwardDifference, 5, 5e-6);
IMPL_TEST_DERIVATIVE(DiffForward, ForwardDifference, 6, 5e-6);

IMPL_TEST_DERIVATIVE(DiffCentral, CentralDifference, 2, 1e-5);
IMPL_TEST_DERIVATIVE(DiffCentral, CentralDifference, 4, 1e-7);
IMPL_TEST_DERIVATIVE(DiffCentral, CentralDifference, 6, std::numeric_limits<float>::epsilon());

IMPL_TEST_DERIVATIVE(DiffBackward, BackwardDifference, 1, 1e-4);
IMPL_TEST_DERIVATIVE(DiffBackward, BackwardDifference, 2, 1e-5);
IMPL_TEST_DERIVATIVE(DiffBackward, BackwardDifference, 3, 1e-6);

TEST(NewtonsMethod, Converge)
{
    constexpr unsigned seed = 23198;
    std::mt19937 generator(seed);

    std::vector<std::vector<double>> zeros{
        {1.0},
        {-2.0, 2.0},
        {1.0, 1.0},
    };

    std::uniform_real_distribution<double> zeroDistrib{-1e3, 1e3};
    std::uniform_int_distribution<unsigned> nZerosDistrib{1, 20};
    std::uniform_int_distribution<unsigned> zeroOrderDistrib{1, 5};

    for (int i = 0; i < 200; i++)
    {
        unsigned n = nZerosDistrib(generator);

        std::vector<double> x0s;
        x0s.reserve(n);

        for (unsigned j = 0; j < n; j++)
        {
            unsigned order = zeroOrderDistrib(generator);
            for (unsigned k = 0; k < order; k++)
                x0s.push_back(zeroDistrib(generator));
        }

        zeros.push_back(x0s);
    }

    std::vector<std::function<double(double)>> funcs;
    funcs.reserve(zeros.size());

    for (std::vector<double>& x0s : zeros)
    {
        auto f = [&x0s](double x)
        {
            double res = 1.0;

            for (double x0 : x0s)
                res *= (x - x0);

            return res;
        };

        funcs.push_back(f);
    }
    std::uniform_real_distribution<double> distrib{-1e4, 1e4};

    for (int nguess = 0; nguess < 10; nguess++)
    {
        int i = 0;
        for (auto f : funcs)
        {
            double x = distrib(generator);
            std::optional<double> x0 = nu::newtons_method(f, x);
            ASSERT_TRUE(x0.has_value()) << " x: " << x << ", zeros: " << ei::Map<ei::VectorXd>(zeros[i].data(), zeros[i].size());
            ASSERT_LE(std::abs(f(*x0)), nu::NewtonsMethodParams{}.zero);
            i++;
        }
    }
}

TEST(NewtonsMethod, Diverge)
{
    constexpr unsigned seed = 12983;
    std::mt19937 generator(seed);

    std::vector<std::vector<double>> params{};

    std::uniform_real_distribution<double> zeroDistrib{1.0, 1e3};
    std::uniform_int_distribution<unsigned> nZerosDistrib{1, 20};
    std::uniform_int_distribution<unsigned> zeroOrderDistrib{1, 3};

    for (int i = 0; i < 20; i++)
    {
        unsigned n = nZerosDistrib(generator);

        std::vector<double> x0s;
        x0s.reserve(n);

        for (unsigned j = 0; j < n; j++)
        {
            unsigned order = zeroOrderDistrib(generator);
            for (unsigned k = 0; k < order; k++)
            {
                double param = zeroDistrib(generator);
                x0s.push_back(param);
            }
        }

        params.push_back(x0s);
    }

    std::vector<std::function<double(double)>> funcs;
    funcs.reserve(params.size());

    for (std::vector<double>& x0s : params)
    {
        auto f = [&x0s](double x)
        {
            double res = 1.0;

            for (double x0 : x0s)
                res *= (x * x + x0);

            return res;
        };

        funcs.push_back(f);
    }
    std::uniform_real_distribution<double> distrib{-1e4, 1e4};

    for (int nguess = 0; nguess < 3; nguess++)
    {
        int i = 0;
        for (auto f : funcs)
        {
            double x = distrib(generator);
            std::optional<double> x0 = nu::newtons_method(f, x);
            ASSERT_FALSE(x0.has_value()) << " x: " << x << ", params: " << ei::Map<ei::VectorXd>(params[i].data(), params[i].size());
            i++;
        }
    }
}

TEST(Deriv, Partial)
{
    auto f = [](ei::Vector2d x) -> double
    {
        return x(0) + 0.5 * x(1) * x(1);
    };

    constexpr unsigned seed = 4386564;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distrib{-1e2, 1e2};

    for (int i = 0; i < 1000; i++)
    {
        auto x = ei::Vector2d{distrib(generator), distrib(generator)};
        const double tolerance = 1e-5 * std::max<double>(1.0, std::abs(x.maxCoeff()));
        ASSERT_NEAR(nu::partial_deriv<nu::CentralDifference<6>>(f, x, 0), 1.0, tolerance) << "x: " << x;
        ASSERT_NEAR(nu::partial_deriv<nu::CentralDifference<6>>(f, x, 1), x(1), tolerance) << "x: " << x;
    }
}

TEST(Deriv, Diffrential)
{
    auto f = [](ei::Vector2d x)
    {
        return ei::Vector3d{10.0, x(0) + 0.5 * x(1) * x(1), std::exp(x(0) + x(1))};
    };

    auto expectedDf = [](ei::Vector2d x)
    {
        return ei::Matrix<double, 3, 2>{
            {0.0, 0.0},
            {1.0, x(1)},
            {std::exp(x(0) + x(1)), std::exp(x(0) + x(1))}};
    };

    constexpr unsigned seed = 31492;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distrib{-1e2, 1e2};

    for (int i = 0; i < 1000; i++)
    {
        auto x = ei::Vector2d{distrib(generator), distrib(generator)};
        ei::Matrix<double, 3, 2> df = nu::differential(f, x);
        ASSERT_NEAR((df - expectedDf(x)).norm() / expectedDf(x).norm(), 0.0, std::numeric_limits<float>::epsilon()) << "x: " << x;
    }
}

TEST(NewtonsMethod, NdConverge)
{
    constexpr unsigned seed = 17188;
    std::mt19937 generator(seed);

    auto f = [](ei::Vector2d x)
    {
        return ei::Vector3d{
            (x(1) - 2.0) * x(0), 
            (x(1) - 2.0) * x(1), 
            (x(1) - 2.0), 
        };
    };

    std::optional<ei::Vector2d> zero = nu::newtons_method(f, static_cast<ei::Vector2d>(ei::Vector2d::Zero()));

    ASSERT_TRUE(zero.has_value());
    ASSERT_NEAR(f(*zero).norm(), nu::NewtonsMethodParams{}.zero, std::numeric_limits<float>::epsilon());
}
