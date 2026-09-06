#pragma once
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

struct DataLoader {
    Tensor data, labels;
    i32 batch_size, n_samples;
    std::vector<i32> idx;
    i32 cursor;

    DataLoader(i32 bs = 32) : batch_size(bs), cursor(0) {}

    void load_csv(const std::string& path, bool has_header = true) {
        std::ifstream f(path);
        if (!f.is_open()) { printf("ERROR: can't open %s\n", path.c_str()); return; }
        std::string line;
        std::vector<std::vector<f32>> rows;
        if (has_header) std::getline(f, line); // skip header
        while (std::getline(f, line)) {
            std::vector<f32> row;
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ','))
                row.push_back(std::stof(val));
            if (!row.empty()) rows.push_back(row);
        }
        n_samples = (i32)rows.size();
        if (n_samples == 0) return;
        i32 cols = (i32)rows[0].size();
        data = Tensor(n_samples, cols - 1);
        labels = Tensor(n_samples, 1);
        for (i32 i = 0; i < n_samples; i++) {
            for (i32 j = 0; j < cols - 1; j++)
                data.at(i, j) = rows[i][j];
            labels.data[i] = rows[i][cols - 1];
        }
        idx.resize(n_samples);
        for (i32 i = 0; i < n_samples; i++) idx[i] = i;
    }

    void shuffle() {
        std::random_shuffle(idx.begin(), idx.end());
        cursor = 0;
    }

    void reset() { cursor = 0; }

    bool next_batch(Tensor& x_batch, Tensor& y_batch) {
        if (cursor >= n_samples) return false;
        i32 end = std::min(cursor + batch_size, n_samples);
        i32 bs = end - cursor;
        x_batch = Tensor(bs, data.cols);
        y_batch = Tensor(bs, labels.cols);
        for (i32 i = 0; i < bs; i++) {
            i32 si = idx[cursor + i];
            for (i32 j = 0; j < data.cols; j++)
                x_batch.at(i, j) = data.at(si, j);
            y_batch.data[i] = labels.data[si];
        }
        cursor = end;
        return true;
    }

    i32 num_batches() const {
        return (n_samples + batch_size - 1) / batch_size;
    }
};
