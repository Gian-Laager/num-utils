#include "nu/aitken-naville.h"

namespace nu
{
    template struct AitkenNaville<ei::Dynamic>;
    template struct AitkenNaville<1>;
    template struct AitkenNaville<2>;
    template struct AitkenNaville<3>;
    template struct AitkenNaville<4>;
}
