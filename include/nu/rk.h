#include "nu/pch.h"

#include <Eigen/Core>

namespace nu 
{
    template<ei::Index Dim>
    struct Rk45
    {
        static constexpr size_t BUTCHER_SIZE = 6;
        double h = 1e-3;

        double tolerance = std::numeric_limits<float>::epsilon();
        double hLowerBound = std::numeric_limits<float>::epsilon();
        double hUppoerBound = std::numeric_limits<double>::infinity();
        double hMultipier = 3.0 / 4.0;

        using StateVec = ei::Vector<double, Dim>;

        StateVec& state;
        double t = 0;

        Rk45(StateVec& state): state(state) {}

        const static ei::Vector<double, BUTCHER_SIZE> c;
        const static ei::Vector<double, BUTCHER_SIZE> b;
        const static ei::Vector<double, BUTCHER_SIZE> bs;
        const static ei::Matrix<double, BUTCHER_SIZE, BUTCHER_SIZE> a;

        template<typename Fn>
        ei::Matrix<double, Dim, BUTCHER_SIZE> calc_ks(const Fn& f) const
        {
            ei::Matrix<double, Dim, BUTCHER_SIZE> ks = ei::Matrix<double, Dim, BUTCHER_SIZE>::Zero(state.rows(), BUTCHER_SIZE);

            for (int i = 0; i < ks.cols(); i++)
            {
                ks.col(i) = f(t + c(i) * h, state + h * ks * a.row(i).transpose());
            }

            return ks;
        }

        template<typename Fn>
        std::pair<double, ei::VectorXd> compute_next_state(const Fn& f) const
        {
            ei::Matrix<double, Dim, BUTCHER_SIZE> ks = calc_ks(f);
            double new_t = t + h;
            ei::VectorXd new_state = state + h * ks * b;

            return std::make_pair(new_t, new_state);
        }

        template<typename Fn>
        void step(const Fn& f)
        {
            ei::Matrix<double, Dim, BUTCHER_SIZE> ks;
            double new_t;
            double current_h;

            while (true)
            {
                ks = calc_ks(f);
                new_t = t + h;
                current_h = h;
                StateVec error = h * ks * (b - bs);

                if (error.squaredNorm() < std::pow(tolerance * 0.1, 2) && h / hMultipier < hUppoerBound)
                {
                    h /= hMultipier;
                    break;
                }

                if (h * hMultipier < hLowerBound)
                {
                    break;
                }

                if (error.squaredNorm() > std::pow(tolerance, 2))
                {
                    h *= hMultipier;
                    continue;
                }
                break;
            }

            t = new_t;
            state += current_h * ks * b;
        }
    };

    using Rk45X = Rk45<ei::Dynamic>;

    template<ei::Index Dim>
    const ei::Vector<double, 6> Rk45<Dim>::c{0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0};
    template<ei::Index Dim>
    const ei::Vector<double, 6> Rk45<Dim>::b{16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55};
    template<ei::Index Dim>
    const ei::Vector<double, 6> Rk45<Dim>::bs{25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0};
    template<ei::Index Dim>
    const ei::Matrix<double, 6, 6> Rk45<Dim>::a{
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0},
        {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197, 0.0, 0.0, 0.0},
        {439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104, 0.0, 0.0},
        {-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0},
    };
}
