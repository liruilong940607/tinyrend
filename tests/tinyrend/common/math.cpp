#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/common/math.h"
#include "tinyrend/common/vec.h"

using namespace tinyrend;

int test_rsqrt() {
    int fails = 0;

    {
        auto const x = 4.0f;
        auto const y = rsqrt(x);
        auto const y_expected = 0.5f;
        fails += CHECK(is_close(y, y_expected), "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_rsqrt();

    if (fails > 0) {
        printf("[common/math.cpp] %d tests failed!\n", fails);
    } else {
        printf("[common/math.cpp] All tests passed!\n");
    }

    return fails;
}