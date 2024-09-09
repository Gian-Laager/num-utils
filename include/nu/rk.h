#include "nu/pch.h"

#include <Eigen/Core>

namespace nu 
{
    struct Rk45
    {
        static constexpr size_t BUTCHER_SIZE = 6;
        double h = 1e-3;

        double tolerance = std::numeric_limits<float>::epsilon();
        double hLowerBound = std::numeric_limits<float>::epsilon();
        double hUppoerBound = std::numeric_limits<double>::infinity();
        double hMultipier = 3.0 / 4.0;

        ei::VectorXd& state;
        double t = 0;

        Rk45(ei::VectorXd& state): state(state) {}

        const static ei::Vector<double, BUTCHER_SIZE> c;
        const static ei::Vector<double, BUTCHER_SIZE> b;
        const static ei::Vector<double, BUTCHER_SIZE> bs;
        const static ei::Matrix<double, BUTCHER_SIZE, BUTCHER_SIZE> a;

        template<typename Fn>
        ei::Matrix<double, ei::Dynamic, BUTCHER_SIZE> calc_ks(const Fn& f) const
        {
            ei::Matrix<double, ei::Dynamic, BUTCHER_SIZE> ks = ei::Matrix<double, ei::Dynamic, BUTCHER_SIZE>::Zero(state.rows(), BUTCHER_SIZE);

            for (int i = 0; i < ks.cols(); i++)
            {
                ks.col(i) = f(t + c(i) * h, state + h * ks * a.row(i).transpose());
            }

            return ks;
        }

        template<typename Fn>
        std::pair<double, ei::VectorXd> compute_next_state(const Fn& f) const
        {
            ei::Matrix<double, ei::Dynamic, BUTCHER_SIZE> ks = calc_ks(f);
            double new_t = t + h;
            ei::VectorXd new_state = state + h * ks * b;

            return std::make_pair(new_t, new_state);
        }

        template<typename Fn>
        void step(const Fn& f)
        {
            ei::Matrix<double, ei::Dynamic, BUTCHER_SIZE> ks;
            double new_t;
            double current_h;

            while (true)
            {
                ks = calc_ks(f);
                new_t = t + h;
                current_h = h;
                ei::VectorXd error = h * ks * (b - bs);

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
}
