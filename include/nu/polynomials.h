#include "nu/pch.h"

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include <cmath>
#include <vector>

namespace nu
{
    template<typename T>
    struct is_complex : public std::false_type
    {
    };

    template<typename R>
    struct is_complex<std::complex<R>> : public std::is_floating_point<R>
    {
    };

    template<typename S>
    struct RealScalar 
    {
        using type = S;
    };

    template<typename S>
    struct RealScalar<std::complex<S>> 
    {
        using type = S;
    };

    template<typename Scalar>
    struct StdPolyBase
    {
        static Scalar eval_base(const Scalar& x, size_t i)
        {
            return std::pow(x, i);
        }
    };

    template<typename Scalar>
    struct LegendrePolyBase
    {
        static Scalar eval_base(const Scalar& x, size_t i)
        {
            return std::legendre(i, x);
        }
    };

    template<typename Scalar, size_t Deg, typename Base = StdPolyBase<Scalar>>
    struct Poly : public ei::Vector<Scalar, Deg + 1>
    {
    private:
        using self_t = Poly<Scalar, Deg, Base>;

        self_t& self()
        {
            return *this;
        }

        const self_t& self() const
        {
            return *this;
        }

    public:
        using ei::Vector<Scalar, Deg + 1>::Vector;

        static ei::RowVector<Scalar, Deg + 1> eval_covector(const Scalar& x)
        {
            ei::RowVector<Scalar, Deg + 1> result;

            for (ei::Index i = 0; i < result.cols(); i++)
                result(i) = Base::eval_base(x, i);

            return result;
        }

        Scalar eval(const Scalar& x) const
        {
            return eval_covector(x) * self();
        }
    };

#define MAKE_POLY_ALIASES_FOR_DIMS(Prefix, Postfix, Scalar, Base) \
    using Prefix##1##Postfix = Poly<Scalar, 1, Base<Scalar>>;     \
    using Prefix##2##Postfix = Poly<Scalar, 2, Base<Scalar>>;     \
    using Prefix##3##Postfix = Poly<Scalar, 3, Base<Scalar>>;     \
    using Prefix##4##Postfix = Poly<Scalar, 4, Base<Scalar>>;     \
    using Prefix##5##Postfix = Poly<Scalar, 5, Base<Scalar>>;     \
    using Prefix##6##Postfix = Poly<Scalar, 6, Base<Scalar>>

#define MAKE_POLY_ALIASES_FOR_BASE(Name, Base)                        \
    MAKE_POLY_ALIASES_FOR_DIMS(Name, d, double, Base);                \
    MAKE_POLY_ALIASES_FOR_DIMS(Name, f, float, Base);                 \
    MAKE_POLY_ALIASES_FOR_DIMS(Name, cd, std::complex<double>, Base); \
    MAKE_POLY_ALIASES_FOR_DIMS(Name, cf, std::complex<float>, Base)

    MAKE_POLY_ALIASES_FOR_BASE(Poly, StdPolyBase);
    MAKE_POLY_ALIASES_FOR_BASE(Legendre, LegendrePolyBase);

    template<typename Scalar, typename = void>
    struct choose_eigen_solver;

    template<typename Scalar>
    struct choose_eigen_solver<Scalar, std::enable_if_t<is_complex<Scalar>::value>>
    {
        using Mat = ei::Matrix<Scalar, ei::Dynamic, ei::Dynamic>;
        using type = ei::ComplexEigenSolver<Mat>;
    };

    template<typename Scalar>
    struct choose_eigen_solver<Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value>>
    {
        using Mat = ei::Matrix<Scalar, ei::Dynamic, ei::Dynamic>;
        using type = ei::EigenSolver<Mat>;
    };

    template<typename Scalar>
    using RootsType = std::vector<typename choose_eigen_solver<Scalar>::type::EigenvalueType::Scalar>;

    template<typename Scalar, size_t Deg>
    RootsType<Scalar> roots(const Poly<Scalar, Deg, StdPolyBase<Scalar>>& poly, typename RealScalar<Scalar>::type h = 0.0)
    {
        using Mat = choose_eigen_solver<Scalar>::Mat;
        using Solver = choose_eigen_solver<Scalar>::type;

        ei::Index nonZeroDeg = Deg;

        for (; nonZeroDeg >= 0; nonZeroDeg--)
        {
            if (std::norm(poly(nonZeroDeg)) > h)
                break;
        }

        Mat charact = Mat::Zero(nonZeroDeg, nonZeroDeg);
        Solver solver;

        for (ei::Index i = 0; i < charact.rows(); i++)
        {
            if (i < charact.rows() - 1) [[likely]]
                charact(i, i + 1) = 1.0;
            charact(charact.rows() - 1, i) = -poly(i) / poly(nonZeroDeg);
        }
        solver.compute(charact, false);
        auto eigs = solver.eigenvalues();
        return std::vector(eigs.begin(), eigs.end());
    }

    template<typename L, typename H>
    using MultiplicationResultType = decltype(std::declval<const L&>() * std::declval<const H&>());

    template<typename ScalarL, size_t DegL, typename ScalarR, size_t DegR>
    Poly<MultiplicationResultType<ScalarL, ScalarR>, DegL + DegR, StdPolyBase<MultiplicationResultType<ScalarL, ScalarR>>>
    operator*(const Poly<ScalarL, DegL, StdPolyBase<ScalarL>>& lhs, const Poly<ScalarR, DegR, StdPolyBase<ScalarR>>& rhs)
    {
        using Res_t = Poly<MultiplicationResultType<ScalarL, ScalarR>, DegL + DegR, StdPolyBase<MultiplicationResultType<ScalarL, ScalarR>>>;
        Res_t result = Res_t::Zero();

        for (ei::Index i = 0; i < lhs.rows(); i++)
        {
            for (ei::Index j = 0; j < rhs.rows(); j++)
            {
                result(i + j) += lhs(i) * rhs(j);
            }
        }
        return result;
    }

    template<typename Scalar, unsigned n>
    struct GaussLegendreHelper
    {
        template<ei::Index Rows, ei::Index Cols>
        using Mat = ei::Matrix<Scalar, Rows, Cols>;

        template<ei::Index Dim>
        using Vec = ei::Vector<Scalar, Dim>;
        static std::pair<Vec<n>, Vec<n>> value()
        {
            using namespace Eigen;

            Mat<n, n> J = Mat<n, n>::Zero(n, n);
            for (int i = 1; i < n; ++i)
            {
                double b = i / std::sqrt(4.0 * i * i - 1.0);
                J(i, i - 1) = b;
                J(i - 1, i) = b;
            }

            SelfAdjointEigenSolver<Mat<n, n>> eigensolver(J);
            Vec<n> points = eigensolver.eigenvalues();
            Mat<n, n> vectors = eigensolver.eigenvectors();

            Vec<n> weights = Vec<n>::Zero(n);
            for (ei::Index i = 0; i < n; ++i)
            {
                weights[i] = 2 * vectors(0, i) * vectors(0, i);
            }

            return {points, weights};
        }
    };

    template<typename Scalar>
    struct GaussLegendreHelper<Scalar, 1>
    {
        static constexpr unsigned n = 1;
        static constexpr std::pair<ei::Vector<Scalar, n>, ei::Vector<Scalar, n>> value()
        {
            return std::make_pair<ei::Vector<Scalar, n>, ei::Vector<Scalar, n>>(
                {0}, {2});
        }
    };

    template<typename Scalar>
    struct GaussLegendreHelper<Scalar, 2>
    {
        static constexpr unsigned n = 2;
        static constexpr std::pair<ei::Vector<Scalar, n>, ei::Vector<Scalar, n>> value()
        {
            return std::make_pair<ei::Vector<Scalar, n>, ei::Vector<Scalar, n>>(
                {1.0 / sqrt(3.0), -1.0 / sqrt(3.0)}, {8.0 / 9.0, 5.0 / 9.0});
        }
    };

    template<typename Scalar>
    struct Quadrature
    {
    private:
        template<ei::Index Rows, ei::Index Cols>
        using Mat = ei::Matrix<Scalar, Rows, Cols>;

        template<ei::Index Dim>
        using Vec = ei::Vector<Scalar, Dim>;

    public:
        template<unsigned n>
        static std::pair<Vec<n>, Vec<n>> gauss_legendre()
        {
            return GaussLegendreHelper<Scalar, n>::value();
        }

        static std::pair<Vec<ei::Dynamic>, Vec<ei::Dynamic>> gauss_legendre(unsigned n)
        {
            using namespace Eigen;

            constexpr ei::Index dyn = ei::Dynamic;

            Mat<dyn, dyn> J = Mat<dyn, dyn>::Zero(n, n);
            for (ei::Index i = 1; i < n; ++i)
            {
                double b = i / std::sqrt(4.0 * i * i - 1.0);
                J(i, i - 1) = b;
                J(i - 1, i) = b;
            }

            SelfAdjointEigenSolver<Mat<dyn, dyn>> eigensolver(J);
            Vec<dyn> points = eigensolver.eigenvalues();
            Mat<dyn, dyn> vectors = eigensolver.eigenvectors();

            Vec<dyn> weights = Vec<dyn>::Zero(n);
            for (ei::Index i = 0; i < n; ++i)
            {
                weights[i] = 2 * vectors(0, i) * vectors(0, i);
            }

            return {points, weights};
        }

        template<unsigned n, typename Fn>
        static std::invoke_result_t<Fn, const Scalar&> integ_over_unit_ball(const Fn& f)
        {
            auto [evalPoints, weights] = gauss_legendre<n>();

            std::invoke_result_t<Fn, const Scalar&> result = weights(0) * f(evalPoints(0));

            for (ei::Index i = 1; i < evalPoints.rows(); i++)
            {
                result += weights(i) * f(evalPoints(i));
            }

            return result;
        }

        template<typename Fn>
        static std::invoke_result_t<Fn, const Scalar&> integ_over_unit_ball(const Fn& f, unsigned n)
        {
            auto [evalPoints, weights] = gauss_legendre(n);

            std::invoke_result_t<Fn, const Scalar&> result = weights(0) * f(evalPoints(0));

            for (ei::Index i = 1; i < evalPoints.rows(); i++)
            {
                result += weights(i) * f(evalPoints(i));
            }

            return result;
        }
    };
}
