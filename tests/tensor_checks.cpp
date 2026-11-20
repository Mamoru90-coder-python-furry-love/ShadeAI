// Runtime shape/bounds checks must work even with NDEBUG defined.
#include "../src/core/tensor.h"
#include <cstdio>
#include <stdexcept>

template <typename Fn>
static bool throws(Fn fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

int main() {
    bool ok = true;
    ok &= throws([] { Tensor t(-1, 3); });
    ok &= throws([] { Tensor t(3, -1); });
    ok &= throws([] { Tensor t(std::numeric_limits<i32>::max(), 2); });
    ok &= throws([] { Tensor t(2, 2, std::vector<f32>(3)); });
    ok &= throws([] { Tensor t(2, 3); (void)t.at(2, 0); });
    ok &= throws([] { Tensor t(2, 3); (void)t.at(0, 3); });
    ok &= throws([] { Tensor t(2, 3); (void)t.at(-1, 0); });
    ok &= throws([] { Tensor t(2, 3); (void)t.slice(1, 3); });
    ok &= throws([] { Tensor a(2, 3), b(4, 5); (void)a.matmul(b); });
    ok &= throws([] { Tensor a(2, 3), b(3, 4); (void)a.add(b); });
    ok &= throws([] { Tensor a(2, 3), b(3, 2); (void)a.emul(b); });
    ok &= throws([] { Tensor a(2, 3), b(3, 2); (void)a.sub(b); });

    if (!ok) { std::fprintf(stderr, "tensor_checks: FAILED\n"); return 1; }
    std::puts("tensor_checks: OK");
    return 0;
}
