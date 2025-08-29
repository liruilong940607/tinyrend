#include <cassert>
#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/common/scalar_grad.h"
#include "tinyrend/common/vec.h"

using namespace tinyrend;

using fsg = scalar_grad<float>;
using fsg_vec3 = vec<fsg, 3>;
using fsg_vec2 = vec<fsg, 2>;

template <typename T>
TREND_HOST_DEVICE bool is_close(
    const scalar_grad<T> &x, const scalar_grad<T> &other, T atol = 1e-5f, T rtol = 1e-5f
) {
    auto const is_scalar_close =
        fabs(x.scalar - other.scalar) <= atol + rtol * fabs(other.scalar);
    auto const is_grad_close =
        fabs(x.grad - other.grad) <= atol + rtol * fabs(other.grad);
    return is_scalar_close && is_grad_close;
}

template <typename T, size_t N>
TREND_HOST_DEVICE bool is_close(
    const vec<scalar_grad<T>, N> &x,
    const vec<scalar_grad<T>, N> &y,
    T atol = 1e-5f,
    T rtol = 1e-5f
) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        if (!is_close(x[i], y[i], atol, rtol)) {
            return false;
        }
    }
    return true;
}

int test() {
    int fails = 0;

    // Size of vec<scalar_grad<float>, N>
    { fails += CHECK(sizeof(fsg_vec3) == 3 * 2 * sizeof(float), ""); }

    // Cast float* to scalar_grad<float>* and cast scalar_grad<float>* to float*
    {
        float data[2] = {1.2f, 2.0f};
        fsg v = *reinterpret_cast<fsg *>(data);
        fails += CHECK(is_close(v, fsg(1.2f, 2.0f)), "");

        float *data2 = reinterpret_cast<float *>(&v);
        fails += CHECK(data2[0] == 1.2f, "");
        fails += CHECK(data2[1] == 2.0f, "");
    }

    // Initialize from values and pointer
    {
        fsg data[3] = {fsg(1.2f, 1.0f), fsg(2.0f, 2.0f), fsg(3.0f, 3.0f)};
        fsg_vec3 v1 = fsg_vec3::from_ptr(data);
        fsg_vec3 v2 = fsg_vec3(fsg(1.2f, 1.0f), fsg(2.0f, 2.0f), fsg(3.0f, 3.0f));
        fails += CHECK(v1 == v2, "");
    }

    // Zero initialization
    {
        fsg_vec3 v1 = fsg_vec3::zero();
        fails += CHECK(is_close(v1, fsg_vec3(fsg(0), fsg(0), fsg(0))), "");
    }

    // Ones initialization
    {
        fsg_vec3 v1 = fsg_vec3::ones();
        fails += CHECK(is_close(v1, fsg_vec3(fsg(1), fsg(1), fsg(1))), "");
    }

    // Sum
    {
        fsg_vec3 v1 = fsg_vec3(fsg(1.2f, 1.0f), fsg(2.0f, 2.0f), fsg(3.0f, 3.0f));
        fails += CHECK(is_close(v1.sum(), fsg(6.2f, 6.0f)), "");
    }

    // Vector-Vector operations
    {
        fsg_vec3 v1 = fsg_vec3(fsg(1.2f, 1.0f), fsg(2.0f, 2.0f), fsg(3.0f, 3.0f));
        fsg_vec3 v2 = fsg_vec3(fsg(4.0f, 4.0f), fsg(5.0f, 5.0f), fsg(6.0f, 6.0f));
        fails += CHECK(
            is_close(
                v1 + v2, fsg_vec3(fsg(5.2f, 5.0f), fsg(7.0f, 7.0f), fsg(9.0f, 9.0f))
            ),
            ""
        );
        fails += CHECK(
            is_close(
                v1 - v2,
                fsg_vec3(fsg(-2.8f, -3.0f), fsg(-3.0f, -3.0f), fsg(-3.0f, -3.0f))
            ),
            ""
        );
        fails += CHECK(
            is_close(
                v1 * v2, fsg_vec3(fsg(4.8f, 8.8f), fsg(10.0f, 20.0f), fsg(18.0f, 36.0f))
            ),
            ""
        );
        fails += CHECK(
            is_close(
                v1 / v2, fsg_vec3(fsg(0.3f, -0.05f), fsg(0.4f, 0.f), fsg(0.5f, 0.f))
            ),
            ""
        );
    }

    // Scalar-Vector operations
    {
        fsg_vec3 v1 = fsg_vec3(fsg(1.2f, 1.0f), fsg(2.0f, 2.0f), fsg(3.0f, 3.0f));
        fails += CHECK(
            is_close(
                1.0f + v1, fsg_vec3(fsg(2.2f, 1.0f), fsg(3.0f, 2.0f), fsg(4.0f, 3.0f))
            ),
            ""
        );
        fails += CHECK(
            is_close(
                1.0f - v1,
                fsg_vec3(fsg(-0.2f, -1.0f), fsg(-1.0f, -2.0f), fsg(-2.0f, -3.0f))
            ),
            ""
        );
        fails += CHECK(
            is_close(
                1.0f * v1, fsg_vec3(fsg(1.2f, 1.0f), fsg(2.0f, 2.0f), fsg(3.0f, 3.0f))
            ),
            ""
        );
        fails += CHECK(
            is_close(
                1.44f / v1,
                fsg_vec3(fsg(1.2f, -1.0f), fsg(0.72f, -0.72f), fsg(0.48f, -0.48f))
            ),
            ""
        );
    }

    // Compound assignment operators
    {
        fsg_vec3 v1 = fsg_vec3(fsg(1.2f), fsg(2.0f), fsg(3.0f));
        v1 += fsg_vec3(fsg(4.0f), fsg(5.0f), fsg(6.0f));
        fails += CHECK(is_close(v1, fsg_vec3(fsg(5.2f), fsg(7.0f), fsg(9.0f))), "");
        v1 -= fsg_vec3(fsg(4.0f), fsg(5.0f), fsg(6.0f));
        fails += CHECK(is_close(v1, fsg_vec3(fsg(1.2f), fsg(2.0f), fsg(3.0f))), "");
        v1 *= fsg_vec3(fsg(4.0f), fsg(5.0f), fsg(6.0f));
        fails += CHECK(is_close(v1, fsg_vec3(fsg(4.8f), fsg(10.0f), fsg(18.0f))), "");
        v1 /= fsg_vec3(fsg(4.0f), fsg(5.0f), fsg(6.0f));
        fails += CHECK(is_close(v1, fsg_vec3(fsg(1.2f), fsg(2.0f), fsg(3.0f))), "");
        v1 += fsg(1.0f);
        fails += CHECK(is_close(v1, fsg_vec3(fsg(2.2f), fsg(3.0f), fsg(4.0f))), "");
        v1 -= fsg(1.0f);
        fails += CHECK(is_close(v1, fsg_vec3(fsg(1.2f), fsg(2.0f), fsg(3.0f))), "");
        v1 *= fsg(2.0f);
        fails += CHECK(is_close(v1, fsg_vec3(fsg(2.4f), fsg(4.0f), fsg(6.0f))), "");
        v1 /= fsg(2.0f);
    }

    // Functions: length, safe_length
    {
        fsg_vec2 v1 = fsg_vec2(fsg(4.0f, 2.0f), fsg(3.0f, 1.0f));
        fails += CHECK(is_close(length(v1), safe_length(v1)), "");

        fsg_vec3 v2 = fsg_vec3(1.0f, 2.0f, 2.0f);
        fails += CHECK(is_close(length(v2), safe_length(v2)), "");
    }

    // Functions: normalize, safe_normalize
    {
        fsg_vec2 v1 = fsg_vec2(fsg(4.0f, 2.0f), fsg(3.0f, 1.0f));
        fails += CHECK(is_close(normalize(v1), safe_normalize(v1)), "");
    }

    // Functions: safe_normalize_vjp
    {
        fsg_vec2 v1 = fsg_vec2(fsg(4.0f, 2.0f), fsg(3.0f, 1.0f));
        fsg_vec2 dl_dout = fsg_vec2(fsg(1.0f, 1.0f), fsg(2.0f, 2.0f));
        fsg_vec2 dl_dv1_expected = fsg_vec2(fsg(-0.12f, -0.0736f), fsg(0.16f, 0.1248f));
        fails += CHECK(is_close(safe_normalize_vjp(v1, dl_dout), dl_dv1_expected), "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test();

    if (fails > 0) {
        printf("[common/vec_scalar_grad.cpp] %d tests failed!\n", fails);
    } else {
        printf("[common/vec_scalar_grad.cpp] All tests passed!\n");
    }

    return fails;
}