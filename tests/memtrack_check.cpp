// Runtime verification of the activation ledger: counts every live Tensor
// buffer and compares the measured forward+backward peak with the estimate.
// SHADE_MEM_TRACK is enabled for this TU only; production builds are untouched.
#define SHADE_MEM_TRACK
#define main gpt_cli_main
#include "../src/gpt.cpp"
#undef main

#include <cstdio>
#include <cstdlib>

int main() {
    bool ok = true;
    auto expect = [&](bool c, const char* what) {
        if (!c) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    struct Case { i32 vocab, dim, blocks, ctx; };
    const Case cases[] = {
        { 32, 16, 1, 8 },
        { 32, 16, 2, 8 },
        { 32, 16, 8, 8 },
        { 64, 32, 4, 16 },
        { 48, 16, 3, 32 }, // larger ctx: the T^2 term matters
    };

    int64_t first_single = -1, first_two = -1, first_eight = -1, first_P = -1;
    for (const auto& cs : cases) {
        GPTConfig cfg{ cs.vocab, cs.dim, cs.blocks, cs.ctx };
        int64_t params = 0, act_est = 0, persistent = 0, fwd = 0, bwd = 0;
        expect(gpt_config_budget(cfg, params), "budget accepts config");
        expect(gpt_activation_estimate(cfg, act_est), "activation estimate computes");
        expect(block_persistent_act(cfg.max_len, cfg.dim, persistent), "persistent peak computes");
        expect(block_forward_transient(cfg.max_len, cfg.dim, fwd), "forward peak computes");
        expect(block_backward_transient(cfg.max_len, cfg.dim, bwd), "backward peak computes");

        srand(1234);
        GPT gpt(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
        Tensor ids(1, cfg.max_len), targets(1, cfg.max_len);
        for (i32 i = 0; i < cfg.max_len; i++) {
            ids.data[i] = (f32)(i % cfg.vocab);
            targets.data[i] = (f32)((i + 1) % cfg.vocab);
        }

        // Model weights/grads/wT and the input tensors already exist; measure
        // only what forward()+compute_loss()+backward() add on top.
        const int64_t base = shade_mem::live();
        shade_mem::reset_peak();

        Tensor logits = gpt.forward(ids);
        Tensor dlogits;
        gpt.compute_loss(logits, targets, dlogits);
        gpt.backward(dlogits, ids);

        const int64_t measured = shade_mem::peak() - base;
        std::printf("  v=%d d=%d b=%d T=%d: measured=%lld estimate=%lld (P=%lld F=%lld W=%lld)\n",
                    cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len,
                    (long long)measured, (long long)act_est,
                    (long long)persistent, (long long)fwd, (long long)bwd);
        expect(measured > 0, "measurement is non-trivial");
        expect(measured <= act_est, "estimate covers the real activation peak");

        // Same config, only blocks differ: the real peak grows by exactly P
        // per block. This proves transients are NOT multiplied by block count.
        if (cs.vocab == 32 && cs.dim == 16 && cs.ctx == 8) {
            if (cs.blocks == 1) { first_single = measured; first_P = persistent; }
            if (cs.blocks == 2) first_two = measured;
            if (cs.blocks == 8) first_eight = measured;
        }
    }
    expect(first_two - first_single == first_P,
           "real peak grows by one persistent set from 1 to 2 blocks");
    expect(first_eight - first_two == 6 * first_P,
           "real peak grows by one persistent set per extra block");

    if (!ok) { std::fprintf(stderr, "memtrack_check: FAILED\n"); return 1; }
    std::puts("memtrack_check: OK");
    return 0;
}
