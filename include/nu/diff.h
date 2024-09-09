#include "nu/pch.h"
#include "Eigen/Dense"
#include <limits>

namespace nu
{
    template<typename Fn, typename in_t, typename Coeffs>
    std::invoke_result_t<Fn, in_t> disc_derivative(const Fn& f, in_t x, in_t h, const Coeffs& coeffs, const int START_OFFSET, const int END_OFFSET)
    {
        std::invoke_result_t<Fn, in_t> result = coeffs[0] * f(x + static_cast<in_t>(START_OFFSET) * h);
        for (int i = START_OFFSET + 1; i <= END_OFFSET; i++)
        {
            result += coeffs[i - START_OFFSET] * f(x + static_cast<in_t>(i) * h);
        }
        return result / h;
    }

    template<typename Fn, typename in_t, typename Coeffs>
    inline std::invoke_result_t<Fn, in_t> cd_derivative(const Fn& f, in_t x, in_t h, const Coeffs& coeffs)
    {
        const int START_OFFSET = -static_cast<int>(coeffs.size() / 2);
        const int END_OFFSET = coeffs.size() / 2;
        return disc_derivative<Fn, in_t, Coeffs>(f, x, h, coeffs, START_OFFSET, END_OFFSET);
    }

    template<size_t Order>
    struct CentralDifference;

#define IMPL_CENTRAL_DIFFERENCE(Order, ...)                                                                          \
    template<>                                                                                                       \
    struct CentralDifference<Order>                                                                                  \
    {                                                                                                                \
    private:                                                                                                         \
        template<typename Fn, typename in_t = double>                                                                \
        using Vec = std::invoke_result_t<Fn, in_t>;                                                                  \
                                                                                                                     \
        static constexpr std::array<double, Order + 1> coeffs{__VA_ARGS__};                                          \
                                                                                                                     \
    public:                                                                                                          \
        template<typename Fn>                                                                                        \
        static Vec<Fn> deriv(const Fn& f, double x)                                                                  \
        {                                                                                                            \
            return cd_derivative<Fn, double, decltype(coeffs)>(f, x, std::numeric_limits<float>::epsilon(), coeffs); \
        }                                                                                                            \
                                                                                                                     \
        template<typename Fn, typename in_t>                                                                         \
        static Vec<Fn, in_t> deriv(const Fn& f, in_t x, in_t h)                                                      \
        {                                                                                                            \
            return cd_derivative(f, x, h, coeffs);                                                                   \
        }                                                                                                            \
    }

    IMPL_CENTRAL_DIFFERENCE(2, -0.5, 0.0, 0.5);
    IMPL_CENTRAL_DIFFERENCE(4, 1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0);
    IMPL_CENTRAL_DIFFERENCE(6, -1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0);

    template<typename Fn, typename in_t, typename Coeffs>
    inline std::invoke_result_t<Fn, in_t> fd_derivative(const Fn& f, in_t x, in_t h, const Coeffs& coeffs)
    {
        const int START_OFFSET = 0;
        const int END_OFFSET = coeffs.size() - 1;
        return disc_derivative(f, x, h, coeffs, START_OFFSET, END_OFFSET);
    }

    template<size_t Order>
    struct ForwardDifference;

#define IMPL_FORWARD_DIFFERENCE(Order, ...)                                                                          \
    template<>                                                                                                       \
    struct ForwardDifference<Order>                                                                                  \
    {                                                                                                                \
    private:                                                                                                         \
        template<typename Fn, typename in_t = double>                                                                \
        using Vec = std::invoke_result_t<Fn, in_t>;                                                                  \
                                                                                                                     \
        static constexpr std::array<double, Order + 1> coeffs{__VA_ARGS__};                                          \
                                                                                                                     \
    public:                                                                                                          \
        template<typename Fn>                                                                                        \
        static Vec<Fn> deriv(const Fn& f, double x)                                                                  \
        {                                                                                                            \
            return fd_derivative<Fn, double, decltype(coeffs)>(f, x, std::numeric_limits<float>::epsilon(), coeffs); \
        }                                                                                                            \
                                                                                                                     \
        template<typename Fn, typename in_t>                                                                         \
        static Vec<Fn, in_t> deriv(const Fn& f, in_t x, in_t h)                                                      \
        {                                                                                                            \
            return fd_derivative(f, x, h, coeffs);                                                                   \
        }                                                                                                            \
    }

    IMPL_FORWARD_DIFFERENCE(1, -1.0, 1.0);
    IMPL_FORWARD_DIFFERENCE(2, -3.0 / 2.0, 2.0, -1.0 / 2.0);
    IMPL_FORWARD_DIFFERENCE(3, -11.0 / 6.0, 3.0, -3.0 / 2.0, 1.0 / 3.0);
    IMPL_FORWARD_DIFFERENCE(4, -25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0);
    IMPL_FORWARD_DIFFERENCE(5, -137.0 / 60.0, 5.0, -5.0, 10.0 / 3.0, -5.0 / 4.0, 1.0 / 5.0);
    IMPL_FORWARD_DIFFERENCE(6, -49.0 / 20.0, 6.0, -15.0 / 2.0, 20.0 / 3.0, -15.0 / 4.0, 6.0 / 5.0, -1.0 / 6.0);

    template<typename Fn, typename in_t, typename Coeffs>
    inline std::invoke_result_t<Fn, in_t> bd_derivative(const Fn& f, in_t x, in_t h, const Coeffs& coeffs)
    {
        const int START_OFFSET = -static_cast<int>(coeffs.size() - 1);
        const int END_OFFSET = 0;
        return disc_derivative(f, x, h, coeffs, START_OFFSET, END_OFFSET);
    }

    template<size_t Order>
    struct BackwardDifference;

#define IMPL_BACKWARD_DIFFERENCE(Order, ...)                                                                         \
    template<>                                                                                                       \
    struct BackwardDifference<Order>                                                                                 \
    {                                                                                                                \
    private:                                                                                                         \
        template<typename Fn, typename in_t = double>                                                                \
        using Vec = std::invoke_result_t<Fn, in_t>;                                                                  \
                                                                                                                     \
        static constexpr std::array<double, Order + 1> coeffs{__VA_ARGS__};                                          \
                                                                                                                     \
    public:                                                                                                          \
        template<typename Fn>                                                                                        \
        static Vec<Fn> deriv(const Fn& f, double x)                                                                  \
        {                                                                                                            \
            return bd_derivative<Fn, double, decltype(coeffs)>(f, x, std::numeric_limits<float>::epsilon(), coeffs); \
        }                                                                                                            \
                                                                                                                     \
        template<typename Fn, typename in_t>                                                                         \
        static Vec<Fn, in_t> deriv(const Fn& f, in_t x, in_t h)                                                      \
        {                                                                                                            \
            return bd_derivative(f, x, h, coeffs);                                                                   \
        }                                                                                                            \
    }

    IMPL_BACKWARD_DIFFERENCE(1, -1.0, 1.0);
    IMPL_BACKWARD_DIFFERENCE(2, 1.0 / 2.0, -2.0, 3.0 / 2.0);
    IMPL_BACKWARD_DIFFERENCE(3, -1.0 / 3.0, 3.0 / 2.0, -3.0, 11.0 / 6.0);

    template<typename Diff = CentralDifference<2>, typename in_t, int N, typename Fn>
    std::invoke_result_t<Fn, ei::Vector<in_t, N>> partial_deriv(const Fn& f, ei::Vector<in_t, N> x, size_t direction, in_t h)
    {
        auto restrictedF = [&](double y)
        {
            ei::Vector<double, N> xh = x;
            xh(direction) = y;
            return f(xh);
        };

        return Diff::deriv(restrictedF, x(direction), h);
    }

    template<typename Diff = CentralDifference<2>, int N, typename Fn>
    inline std::invoke_result_t<Fn, ei::Vector<double, N>> partial_deriv(const Fn& f, ei::Vector<double, N> x, size_t direction)
    {
        return partial_deriv<Diff, double, N, Fn>(f, x, direction, std::numeric_limits<float>::epsilon());
    }

    template<typename Diff = CentralDifference<2>, typename in_t, int N, typename Fn>
    ei::Matrix<double, std::invoke_result_t<Fn, ei::Vector<in_t, N>>::SizeAtCompileTime, N> differential(const Fn& f, const ei::Vector<in_t, N>& x, in_t h)
    {
        constexpr int D = std::invoke_result_t<Fn, ei::Vector<in_t, N>>::SizeAtCompileTime;
        ei::Matrix<double, D, N> result;
        ei::Vector<double, D> d0f = partial_deriv(f, x, 0, h);
        result.resize(d0f.rows(), x.rows());
        result.col(0) = d0f;

        for (int i = 1; i < x.rows(); i++)
        {
            result.col(i) = partial_deriv<Diff>(f, x, i, h);
        }

        return result;
    }

    template<typename Diff = CentralDifference<2>, int N, typename Fn>
    inline ei::Matrix<double, std::invoke_result_t<Fn, ei::Vector<double, N>>::SizeAtCompileTime, N> differential(const Fn& f, const ei::Vector<double, N>& x)
    {
        return differential<Diff, double, N, Fn>(f, x, std::numeric_limits<float>::epsilon());
    }

    struct NewtonsMethodParams
    {
        double h = std::numeric_limits<float>::epsilon();
        double zero = std::numeric_limits<float>::epsilon();
        size_t maxIters = 5e3;
    };

    template<typename Diff = CentralDifference<2>, typename Fn>
    std::optional<double> newtons_method(const Fn& f, double x0, NewtonsMethodParams params = NewtonsMethodParams{})
    {
        double xn = x0;

        for (size_t i = 0; i < params.maxIters; i++)
        {
            double fx = f(xn);

            if (std::abs(fx) < params.zero)
                return xn;

            double df = Diff::deriv(f, xn, params.h);

            if (std::abs(df) < std::numeric_limits<double>::epsilon())
                return std::nullopt;

            xn -= fx / df;
        }

        return std::nullopt;
    }

    template<typename Diff = CentralDifference<2>, int N, typename Fn>
    std::optional<ei::Vector<double, N>> newtons_method(const Fn& f, ei::Vector<double, N> x0, NewtonsMethodParams params = NewtonsMethodParams{})
    {
        constexpr int D = std::invoke_result_t<Fn, ei::Vector<double, N>>::SizeAtCompileTime;
        ei::Vector<double, N> xn = x0;

        for (size_t i = 0; i < params.maxIters; i++)
        {
            ei::Vector<double, D> fx = f(xn);

            if (fx.squaredNorm() < std::pow(params.zero, 2))
                return xn;

            ei::Matrix<double, D, N> df = differential<Diff>(f, xn, params.h);
            ei::Vector<double, N> step = df.colPivHouseholderQr().solve(-fx);

            xn += step;
        }

        return std::nullopt;
    }
}
