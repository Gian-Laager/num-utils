#include "nu/pch.h"

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include <cmath>
#include <type_traits>

namespace nu
{
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

        template<size_t OutDeg>
        operator typename std::enable_if<(OutDeg >= Deg), Poly<Scalar, OutDeg, Base>>::type()
        {
            Poly<Scalar, OutDeg, Base> result = Poly<Scalar, OutDeg, Base>::Zero();

            result.template block<Deg, 1>(0, 0) = self();
            return result;
        }

        template<size_t InDeg>
        explicit Poly(std::enable_if_t<(InDeg < Deg), const ei::Vector<Scalar, InDeg>&> v)
        {
            self().setZero();
            self().template block<InDeg, 1>(0, 0) = v;
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
}
