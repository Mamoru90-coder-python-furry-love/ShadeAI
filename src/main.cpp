#include "core/tensor.h"
#include "core/loss.h"
#include "layers/dense.h"
#include "layers/dropout.h"
#include "layers/batchnorm.h"
#include "layers/activations.h"
#include "layers/gru.h"
#include "layers/attention.h"
#include "optimizer/adamw.h"
#include "utils/serialize.h"
#include <iostream>
#include <cmath>

// ============== ORIGINAL WORKING XOR ==============
struct MLP {
    Dense fc1, fc2;
    Tensor last_f1, last_out;
    MLP() : fc1(2, 16, true), fc2(16, 2, false) {}
    Tensor forward(const Tensor& x) { last_f1 = fc1.forward(x); last_out = fc2.forward(last_f1); return last_out; }
    void backward(const Tensor& d) { Tensor d2 = fc2.backward(d); fc1.backward(d2); }
    std::vector<Tensor*> params() { return {&fc1.w, &fc1.b, &fc2.w, &fc2.b}; }
    std::vector<Tensor*> grads() { return {&fc1.gw, &fc1.gb, &fc2.gw, &fc2.gb}; }
};

void test_xor() {
    std::cout << "=== XOR (original working version) ===\n";
    std::srand(42);
    Tensor x(4,2), y(4,2);
    x.data = {0,0,0,1,1,0,1,1};
    y.data = {1,0,0,1,0,1,1,0};
    MLP m; AdamW opt(0.01f, 0.001f);
    for (auto* p : m.params()) opt.add(*p);
    for (int ep = 0; ep < 500; ep++) {
        auto out = m.forward(x); Tensor d;
        auto loss = CrossEntropyLoss::forward(out, y, d);
        m.backward(d);
        auto pp = m.params(); auto gp = m.grads(); opt.step(pp, gp); opt.zero_grad(gp);
        if (ep % 100 == 0) {
            int c=0; for(int i=0;i<4;i++){int pr=out.data[i*2]>out.data[i*2+1];int tr=y.data[i*2]>y.data[i*2+1];if(pr==tr)c++;}
            std::cout << "  ep " << ep << " loss=" << loss << " acc=" << c*25 << "%\n";
        }
    }
}

// ============== GRU ==============
void test_gru() {
    std::cout << "\n=== GRU (4-bit parity) ===\n";
    std::srand(42);
    int N=100, T=4;
    Tensor X(N,T), Y(N,2);
    for(int i=0;i<N;i++){int s=0;for(int t=0;t<T;t++){X.at(i,t)=rand()%2;s+=(int)X.at(i,t);}Y.at(i,s%2?1:0)=1;}
    GRU gru(1,16); Dense fc(16,2,false);
    AdamW opt(5e-3f,1e-3f);
    for(auto* p:gru.params())opt.add(*p);
    opt.add(fc.w);opt.add(fc.b);
    for(int ep=0;ep<500;ep++){
        auto pp=gru.params();pp.push_back(&fc.w);pp.push_back(&fc.b);
        auto gp=gru.grads();gp.push_back(&fc.gw);gp.push_back(&fc.gb);
        opt.zero_grad(gp);
        float loss=0;
        for(int s=0;s<N;s++){
            Tensor xs(T,1);for(int t=0;t<T;t++)xs.at(t,0)=X.at(s,t);
            Tensor h0(1,16,0);auto hs=gru.forward(xs,h0);
            Tensor lh(1,16);for(int j=0;j<16;j++)lh.data[j]=hs.at(T-1,j);
            auto out=fc.forward(lh);
            Tensor tgt(1,2);tgt.data[0]=Y.at(s,0);tgt.data[1]=Y.at(s,1);
            Tensor d;loss+=CrossEntropyLoss::forward(out,tgt,d);
            auto dh=fc.backward(d);
            Tensor dhf(T,16,0);for(int j=0;j<16;j++)dhf.at(T-1,j)=dh.data[j];
            gru.backward(dhf);
        }
        opt.step(pp,gp);
        if(ep%100==0){
            int c=0;
            for(int s=0;s<N;s++){
                Tensor xs(T,1);for(int t=0;t<T;t++)xs.at(t,0)=X.at(s,t);
                Tensor h0(1,16,0);auto hs=gru.forward(xs,h0);
                Tensor lh(1,16);for(int j=0;j<16;j++)lh.data[j]=hs.at(T-1,j);
                auto out=fc.forward(lh);
                if((out.data[0]>out.data[1])==(Y.at(s,0)>Y.at(s,1)))c++;
            }
            printf("  ep %d loss=%.4f acc=%.1f%%\n",ep,loss/N,100.0f*c/N);
        }
    }
    // test cases
    int tests[4][4]={{0,0,0,0},{0,0,1,1},{1,0,1,0},{1,1,1,1}};
    int correct=0;
    for(int t=0;t<4;t++){
        Tensor x(4,1);int sum=0;
        for(int i=0;i<4;i++){x.data[i]=(float)tests[t][i];sum+=tests[t][i];}
        Tensor h0(1,16,0);auto hs=gru.forward(x,h0);
        Tensor lh(1,16);for(int j=0;j<16;j++)lh.data[j]=hs.at(3,j);
        auto out=fc.forward(lh);
        int pred=out.data[0]>out.data[1]?0:1;
        int truth=sum%2==0?0:1;
        if(pred==truth)correct++;
        printf("    %d%d%d%d -> %s (%.3f,%.3f)\n",tests[t][0],tests[t][1],tests[t][2],tests[t][3],
               pred==0?"even":"odd",out.data[0],out.data[1]);
    }
    printf("  test acc: %d/4\n",correct);
}

// ============== Self-Attention ==============
void test_attn() {
    printf("\n=== Self-Attention (copy first token) ===\n");
    srand(42);
    int N=80,T=4,D=3;
    Tensor X(N,T*D),Y(N,D);
    for(int i=0;i<N;i++){for(int t=0;t<T;t++)for(int j=0;j<D;j++)X.at(i,t*D+j)=((float)rand()/RAND_MAX-0.5f)*2;for(int j=0;j<D;j++)Y.at(i,j)=X.at(i,j);}
    SelfAttention att(D,1);Dense fc(D,D,false);
    AdamW opt(1e-2f,1e-3f);
    for(auto*p:att.params())opt.add(*p);opt.add(fc.w);opt.add(fc.b);
    auto mse_loss=[](const Tensor&p,const Tensor&t)->float{float l=0;for(int i=0;i<p.size();i++){float d=p.data[i]-t.data[i];l+=d*d;}return l/p.size();};
    auto mse_grad=[](const Tensor&p,const Tensor&t,Tensor&d){d=Tensor(p.rows,p.cols);for(int i=0;i<d.size();i++)d.data[i]=2*(p.data[i]-t.data[i])/p.size();};
    for(int ep=0;ep<300;ep++){
        auto pp=att.params();pp.push_back(&fc.w);pp.push_back(&fc.b);
        auto gp=att.grads();gp.push_back(&fc.gw);gp.push_back(&fc.gb);
        opt.zero_grad(gp);float tl=0;
        for(int s=0;s<N;s++){
            Tensor x(T,D);for(int t=0;t<T;t++)for(int j=0;j<D;j++)x.at(t,j)=X.at(s,t*D+j);
            Tensor target(1,D);for(int j=0;j<D;j++)target.data[j]=Y.at(s,j);
            auto a=att.forward(x);
            Tensor pool(1,D);for(int j=0;j<D;j++){float su=0;for(int t=0;t<T;t++)su+=a.at(t,j);pool.data[j]=su/T;}
            auto pred=fc.forward(pool);Tensor d;tl+=mse_loss(pred,target);mse_grad(pred,target,d);
            auto dp=fc.backward(d);
            Tensor da(T,D);for(int j=0;j<D;j++)for(int t=0;t<T;t++)da.at(t,j)=dp.data[j]/T;
            att.backward(da);
        }
        opt.step(pp,gp);
        if(ep%50==0)printf("  ep %d loss=%.6f\n",ep,tl/N);
    }
    // test on an independent seed, not a replay of the training set
    srand(12345);
    float err=0;
    for(int t=0;t<10;t++){
        Tensor x(T,D);for(int i=0;i<T;i++)for(int j=0;j<D;j++)x.at(i,j)=((float)rand()/RAND_MAX-0.5f)*2;
        Tensor target(1,D);for(int j=0;j<D;j++)target.data[j]=x.at(0,j);
        auto a=att.forward(x);
        Tensor pool(1,D);for(int j=0;j<D;j++){float su=0;for(int i=0;i<T;i++)su+=a.at(i,j);pool.data[j]=su/T;}
        auto pred=fc.forward(pool);
        for(int j=0;j<D;j++)err+=fabs(pred.data[j]-target.data[j]);
    }
    printf("  test avg err: %.4f\n",err/30);
}

// ============== Save/Load ==============
void test_saveload() {
    printf("\n=== Save/Load ===\n");
    srand(42);
    Dense d1(2,4,false),d2(2,4,false);
    printf("  save: %s\n",Serialize::save("test_model.bin",{&d1.w,&d1.b})?"OK":"FAIL");
    for(int i=0;i<d2.w.size();i++)d2.w.data[i]=999;
    printf("  load: %s\n",Serialize::load("test_model.bin",{&d2.w,&d2.b})?"OK":"FAIL");
    float diff=0;for(int i=0;i<d1.w.size();i++)diff+=fabs(d1.w.data[i]-d2.w.data[i]);
    printf("  diff: %.8f\n",diff);
}

int main() {
    test_xor();
    test_gru();
    test_attn();
    test_saveload();
    printf("\n===== ALL DONE =====\n");
    return 0;
}
