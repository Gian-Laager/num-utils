#include "Eigen/Core"
#include "pch.h"
#include "nu/polynomials.h"
#include <cmath>
#include <complex>
#include <random>
#include <vector>

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

bool lex_ord_on_c(const std::complex<double>& lhs, const std::complex<double>& rhs)
{
    return rhs.real() == lhs.real() ? rhs.imag() < lhs.imag() : rhs.real() < lhs.real();
}

TEST(Roots, Deg2InR)
{
    using namespace nu;

    Poly2d p{1.0, 0.0, -1.0};
    std::vector<std::complex<double>> zerosList = roots(p);
    ei::VectorXcd expected = ei::Vector2cd{1.0, -1.0};

    ei::VectorXcd zeros = ei::Map<ei::Vector2cd>(zerosList.data(), zerosList.size());

    std::sort(zeros.begin(), zeros.end(), lex_ord_on_c);
    std::sort(expected.begin(), expected.end(), lex_ord_on_c);

    ASSERT_EQ(expected.rows(), zeros.rows());
    for (ei::Index i = 0; i < expected.rows(); i++)
    {
        ASSERT_NEAR(std::norm(expected(i) - zeros(i)), 0.0, std::numeric_limits<double>::epsilon() * std::norm(expected(i)));
    }
}

template<ei::Index NRoots>
void test_roots(ei::Vector<double, NRoots> expectedZeros)
{
    using namespace nu;

    Poly<double, NRoots> p = Poly<double, NRoots>::Zero();
    p(0) = 1.0;

    for (auto root : expectedZeros)
    {
        p = (p * Poly1d{-root, 1.0}).template block<NRoots + 1, 1>(0, 0);
    }

    std::vector<std::complex<double>> zerosList = roots(p);
    ei::VectorXcd zeros = ei::Map<ei::VectorXcd>(zerosList.data(), zerosList.size());

    std::sort(zeros.begin(), zeros.end(), lex_ord_on_c);
    std::sort(expectedZeros.begin(), expectedZeros.end(), lex_ord_on_c);

    ASSERT_EQ(expectedZeros.rows(), zeros.rows());
    for (ei::Index i = 0; i < expectedZeros.rows(); i++)
    {
        ASSERT_NEAR(std::norm(expectedZeros(i) - zeros(i)), 0.0, 2.0 * std::numeric_limits<double>::epsilon() * std::norm(expectedZeros(i)));
    }
}

template<ei::Index NRoots>
void test_roots_complex(ei::Vector<std::complex<double>, NRoots> expectedZeros)
{
    using namespace nu;

    Poly<std::complex<double>, NRoots> p = Poly<std::complex<double>, NRoots>::Zero();
    p(0) = 1.0;

    for (auto root : expectedZeros)
    {
        p = (p * Poly1cd{-root, 1.0}).template block<NRoots + 1, 1>(0, 0);
    }

    std::vector<std::complex<double>> zerosList = roots(p);
    ei::VectorXcd zeros = ei::Map<ei::VectorXcd>(zerosList.data(), zerosList.size());

    std::sort(zeros.begin(), zeros.end(), lex_ord_on_c);
    std::sort(expectedZeros.begin(), expectedZeros.end(), lex_ord_on_c);

    ASSERT_EQ(expectedZeros.rows(), zeros.rows());
    for (ei::Index i = 0; i < expectedZeros.rows(); i++)
    {
        ASSERT_NEAR(std::norm(expectedZeros(i) - zeros(i)), 0.0, 2.0 * std::numeric_limits<double>::epsilon() * std::norm(expectedZeros(i)));
    }
}

TEST(Roots, Real)
{
    using namespace std::complex_literals;
    std::mt19937 mt(3084);
    std::uniform_real_distribution<double> dist(-5.0, 5.0); // Generate real and imaginary parts between -5 and 5

    // Degree 3 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots<3>({dist(mt), dist(mt), dist(mt)});

    // Degree 4 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots<4>({dist(mt), dist(mt), dist(mt), dist(mt)});

    // Degree 5 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots<5>({dist(mt), dist(mt), dist(mt), dist(mt), dist(mt)});

    // Degree 6 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots<6>({dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt)});
}

TEST(Roots, Complex)
{
    using namespace std::complex_literals;
    std::mt19937 mt(13272);
    std::uniform_real_distribution<double> dist(-5.0, 5.0); // Generate real and imaginary parts between -5 and 5

    auto random_complex = [&mt, &dist]()
    {
        return std::complex<double>(dist(mt), dist(mt));
    };

    // Degree 3 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots_complex<3>({random_complex(), random_complex(), random_complex()});

    // Degree 4 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots_complex<4>({random_complex(), random_complex(), random_complex(), random_complex()});

    // Degree 5 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots_complex<5>({random_complex(), random_complex(), random_complex(), random_complex(), random_complex()});

    // Degree 6 polynomial tests
    for (int i = 0; i < 10; i++)
        test_roots_complex<6>({random_complex(), random_complex(), random_complex(), random_complex(), random_complex(), random_complex()});
}

TEST(Roots, Degenerate)
{
    using namespace nu;
    using namespace std::complex_literals;

    Poly2d p{0.0, 1.0, 0.0};
    std::vector<std::complex<double>> zeros = roots(p);

    ASSERT_EQ(zeros.size(), 1);
    ASSERT_EQ(zeros[0], 0.0 + 0.0i);
}
