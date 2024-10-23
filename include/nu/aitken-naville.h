#pragma once

namespace nu
{
    template<ei::Index D>
    struct AitkenNaville
    {
    private:
        using vec_t = ei::Vector<double, D>;
        std::vector<vec_t> coeffs;
        std::vector<double> times;

    public:
        vec_t x;

        AitkenNaville(const vec_t& x): x(x) {}

        void add_point(double t, double y)
        {
            size_t l = coeffs.size();
            coeffs.push_back(vec_t::Constant(x.rows(), y)); // p_{l,l}(x)
            times.push_back(t);
            for (ssize_t k = l - 1; k >= 0; k--)
            {
                if (times[l] == times[k])
                {
                    throw std::runtime_error("Same node added twice with different value");
                }
                coeffs[k] = coeffs[k + 1].array() + (x.array() - times[l]) / (times[l] - times[k]) *
                                                       (coeffs[k + 1] - coeffs[k]).array();
            }
        }

        vec_t eval()
        {
            if (coeffs.empty())
                return vec_t::Zero(x.rows());
            return coeffs[0];
        }
    };

    using AitkenNavilleXd = AitkenNaville<ei::Dynamic>;

    extern  template struct AitkenNaville<ei::Dynamic>;
    extern  template struct AitkenNaville<1>;
    extern  template struct AitkenNaville<2>;
    extern  template struct AitkenNaville<3>;
    extern  template struct AitkenNaville<4>;
}
