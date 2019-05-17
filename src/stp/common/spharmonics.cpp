// Fix bug in spherical harmonics library (CHECK was not defined when NDEBUG is defined)
namespace sh {
namespace {
#ifdef NDEBUG
#define CHECK(condition, message) \
    do {                          \
    } while (false)
#endif
}
}

#include "sh/spherical_harmonics.cc"
