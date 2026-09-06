#pragma once
#include "../core/tensor.h"
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct MNIST {
    Tensor train_x, train_y, test_x, test_y;

    bool load(const std::string& dir, int train_limit = 2000, int test_limit = 500) {
        std::vector<std::pair<int,std::string>> train_files, test_files;
        for (int d = 0; d < 10; d++) {
            std::string td = dir + "/mnist_png/training/" + std::to_string(d);
            std::string ted = dir + "/mnist_png/testing/" + std::to_string(d);
            if (fs::exists(td))
                for (auto& e : fs::directory_iterator(td))
                    train_files.push_back({d, e.path().string()});
            if (fs::exists(ted))
                for (auto& e : fs::directory_iterator(ted))
                    test_files.push_back({d, e.path().string()});
        }
        auto load = [&](std::vector<std::pair<int,std::string>>& files, Tensor& x, Tensor& y, int limit) {
            // Fisher-Yates shuffle
            for (int i = (int)files.size() - 1; i > 0; i--)
                std::swap(files[i], files[rand() % (i + 1)]);
            int n = std::min((int)files.size(), limit);
            x = Tensor(n, 784); y = Tensor(n, 10, 0.0f);
            for (int i = 0; i < n; i++) {
                int w, h, c;
                unsigned char* p = stbi_load(files[i].second.c_str(), &w, &h, &c, 1);
                if (p) { for (int j = 0; j < 784; j++) x.at(i, j) = p[j] / 255.0f; stbi_image_free(p); }
                y.at(i, files[i].first) = 1.0f;
            }
            printf("  loaded %d/%d\n", n, (int)files.size());
        };
        load(train_files, train_x, train_y, train_limit);
        load(test_files, test_x, test_y, test_limit);
        return train_x.rows > 0;
    }
};
