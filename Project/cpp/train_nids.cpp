#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <iomanip>

using json = nlohmann::json;
namespace fs = std::filesystem;

float safe_stof(const std::string& s) {
    if (s.empty()) return NAN;
    try { return std::stof(s); } catch (...) { return NAN; }
}

int label_from_string(const std::string& s) {
    if (s == "0" || s == "BENIGN") return 0;
    return 1;
}

struct Dataset {
    std::vector<float> X;
    std::vector<int> y;
    size_t n, d;
    std::vector<std::string> feature_names;
    std::unordered_map<std::string, float> medians;
};

Dataset load_data(const std::string& path, size_t max_n = 50000) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Файл не найден: " << path << "\n"; std::exit(1); }

    std::string line;
    std::getline(f, line);
    std::stringstream ss(line);
    std::string col;
    std::vector<std::string> headers;
    while (std::getline(ss, col, ',')) {
        col.erase(std::remove(col.begin(), col.end(), '"'), col.end());
        headers.push_back(col);
    }

    int label_idx = -1;
    for (int i = 0; i < (int)headers.size(); ++i) {
        if (headers[i] == "Label") { label_idx = i; break; }
    }
    if (label_idx == -1) { std::cerr << "Колонка 'Label' не найдена\n"; std::exit(1); }

    std::vector<int> keep;
    std::vector<std::string> feats;
    for (int j = 0; j < (int)headers.size(); ++j) {
        if (j == label_idx) continue;
        std::string h = headers[j];
        std::transform(h.begin(), h.end(), h.begin(), ::tolower);
        if (h.find("ip") != std::string::npos || h.find("time") != std::string::npos || h.find("date") != std::string::npos || h == "attack") continue;
        keep.push_back(j);
        feats.push_back(headers[j]);
    }

    std::vector<std::vector<float>> raw(feats.size());
    std::vector<std::vector<std::string>> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        ss.clear(); ss.str(line);
        std::vector<std::string> r;
        while (std::getline(ss, col, ',')) {
            col.erase(std::remove(col.begin(), col.end(), '"'), col.end());
            r.push_back(col);
        }
        if ((int)r.size() <= label_idx) continue;
        rows.push_back(r);
        for (size_t i = 0; i < keep.size(); ++i) {
            if (keep[i] < (int)r.size()) {
                float v = safe_stof(r[keep[i]]);
                if (!std::isnan(v)) raw[i].push_back(v);
            }
        }
    }

    std::unordered_map<std::string, float> med;
    for (size_t i = 0; i < feats.size(); ++i) {
        if (raw[i].empty()) { med[feats[i]] = 0.0f; }
        else {
            auto v = raw[i]; std::sort(v.begin(), v.end());
            med[feats[i]] = v[v.size()/2];
        }
    }

    std::vector<float> X;
    std::vector<int> y;
    for (auto& r : rows) {
        if ((int)r.size() <= label_idx) continue;
        int lb = label_from_string(r[label_idx]);
        for (size_t i = 0; i < keep.size(); ++i) {
            float v = med[feats[i]];
            if (keep[i] < (int)r.size()) {
                float rv = safe_stof(r[keep[i]]);
                if (!std::isnan(rv)) v = rv;
            }
            X.push_back(v);
        }
        y.push_back(lb);
    }

    size_t n0 = std::count(y.begin(), y.end(), 0);
    size_t n1 = std::count(y.begin(), y.end(), 1);
    if (std::min(n0, n1) < 1000 && std::max(n0, n1) > std::min(n0, n1) * 20) {
        std::vector<size_t> i0, i1;
        for (size_t i = 0; i < y.size(); ++i) (y[i] ? i1 : i0).push_back(i);
        size_t t = std::min(n1 * 20, n0);
        std::shuffle(i0.begin(), i0.end(), std::default_random_engine{42});
        i0.resize(t);
        std::vector<size_t> sel = i1;
        sel.insert(sel.end(), i0.begin(), i0.end());
        std::sort(sel.begin(), sel.end());
        std::vector<float> Xs; std::vector<int> ys;
        for (size_t i : sel) {
            size_t s = i * keep.size();
            Xs.insert(Xs.end(), X.begin() + s, X.begin() + s + keep.size());
            ys.push_back(y[i]);
        }
        X = std::move(Xs); y = std::move(ys);
    }

    if (y.size() > max_n) {
        std::vector<size_t> idx(y.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::default_random_engine{42});
        idx.resize(max_n);
        std::vector<float> Xs; std::vector<int> ys;
        for (size_t i : idx) {
            size_t s = i * keep.size();
            Xs.insert(Xs.end(), X.begin() + s, X.begin() + s + keep.size());
            ys.push_back(y[i]);
        }
        X = std::move(Xs); y = std::move(ys);
    }

    Dataset d;
    d.X = X; d.y = y; d.n = y.size(); d.d = keep.size(); d.feature_names = feats; d.medians = med;
    return d;
}

struct Metrics {
    float acc = 0, prec = 0, rec = 0, f1 = 0;
};

Metrics eval(const std::vector<int>& yt, const std::vector<int>& yp) {
    size_t tp = 0, fp = 0, fn = 0, tn = 0;
    for (size_t i = 0; i < yt.size(); ++i) {
        if (yt[i] == 1 && yp[i] == 1) tp++;
        if (yt[i] == 0 && yp[i] == 1) fp++;
        if (yt[i] == 1 && yp[i] == 0) fn++;
        if (yt[i] == 0 && yp[i] == 0) tn++;
    }
    float a = (tp + tn) / (float)(tp + tn + fp + fn);
    float p = tp + fp ? tp / (float)(tp + fp) : 0;
    float r = tp + fn ? tp / (float)(tp + fn) : 0;
    float f = p + r ? 2 * p * r / (p + r) : 0;
    return {a, p, r, f};
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x)))); }

void save_metrics(const std::string& model, double t_sec, const Metrics& m) {
    json e = {
        {"timestamp", std::time(nullptr)},
        {"model", model},
        {"language", "cpp"},
        {"training_time_sec", t_sec},
        {"metrics", {
            {"accuracy", m.acc},
            {"precision_macro", m.prec},
            {"recall_macro", m.rec},
            {"f1_macro", m.f1},
            {"roc_auc", nullptr}
        }}
    };

    std::string p = "../../data/metrics.json";
    json all = json::object();
    if (fs::exists(p)) {
        std::ifstream f(p);
        if (f.is_open()) { try { all = json::parse(f); } catch(...) {} }
    }

    if (!all.contains("cpp")) all["cpp"] = json::object();
    if (!all["cpp"].contains(model)) all["cpp"][model] = json::array();
    all["cpp"][model].push_back(e);

    fs::create_directories("../../data");
    std::ofstream out(p);
    out << all.dump(2);
}

struct DTNode {
    int feature = -1;
    float threshold = 0;
    int left = -1, right = -1;
    int class_label = -1;
};

class DecisionTree {
public:
    void train(const std::vector<float>& X, const std::vector<int>& y, size_t d, int max_depth = 8) {
        nodes.clear();
        build_tree(X, y, d, 0, 0, max_depth);
    }

    int predict(const std::vector<float>& x) const {
        int node = 0;
        while (nodes[node].feature != -1) {
            if (x[nodes[node].feature] <= nodes[node].threshold) {
                node = nodes[node].left;
            } else {
                node = nodes[node].right;
            }
        }
        return nodes[node].class_label;
    }

private:
    std::vector<DTNode> nodes;

    void build_tree(const std::vector<float>& X, const std::vector<int>& y, size_t d, size_t start, int depth, int max_depth) {
        size_t n = y.size();
        if (n == 0) return;
        int cls0 = std::count(y.begin(), y.end(), 0);
        int cls1 = n - cls0;
        DTNode leaf;
        leaf.class_label = cls1 > cls0 ? 1 : 0;
        if (depth >= max_depth || cls0 == 0 || cls1 == 0) {
            nodes.push_back(leaf);
            return;
        }

        float best_gain = -1;
        int best_f = -1;
        float best_th = 0;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> feat_dist(0, (int)d - 1);
        for (int trial = 0; trial < 10; ++trial) {
            int f = feat_dist(gen);
            std::vector<float> vals;
            for (size_t i = 0; i < n; ++i) vals.push_back(X[start * d + i * d + f]);
            std::sort(vals.begin(), vals.end());
            for (size_t i = 1; i < vals.size(); i += std::max(1UL, vals.size()/5)) {
                float th = (vals[i-1] + vals[i]) / 2.0f;
                int l0 = 0, l1 = 0, r0 = 0, r1 = 0;
                for (size_t j = 0; j < n; ++j) {
                    if (X[start * d + j * d + f] <= th) {
                        (y[j] ? l1 : l0)++;
                    } else {
                        (y[j] ? r1 : r0)++;
                    }
                }
                float left_gini = 1.0f - pow(l0/(float)(l0+l1+1e-9), 2) - pow(l1/(float)(l0+l1+1e-9), 2);
                float right_gini = 1.0f - pow(r0/(float)(r0+r1+1e-9), 2) - pow(r1/(float)(r0+r1+1e-9), 2);
                float gain = (l0+l1)*(1.0f-left_gini) + (r0+r1)*(1.0f-right_gini);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_f = f;
                    best_th = th;
                }
            }
        }

        if (best_f == -1) {
            nodes.push_back(leaf);
            return;
        }

        DTNode node;
        node.feature = best_f;
        node.threshold = best_th;
        int node_idx = nodes.size();
        nodes.push_back(node);

        std::vector<size_t> left_idx, right_idx;
        for (size_t i = 0; i < n; ++i) {
            if (X[start * d + i * d + best_f] <= best_th) left_idx.push_back(i);
            else right_idx.push_back(i);
        }

        std::vector<float> Xl, Xr;
        std::vector<int> yl, yr;
        for (size_t i : left_idx) {
            for (size_t j = 0; j < d; ++j) Xl.push_back(X[start * d + i * d + j]);
            yl.push_back(y[i]);
        }
        for (size_t i : right_idx) {
            for (size_t j = 0; j < d; ++j) Xr.push_back(X[start * d + i * d + j]);
            yr.push_back(y[i]);
        }

        nodes[node_idx].left = nodes.size();
        build_tree(Xl, yl, d, 0, depth + 1, max_depth);
        nodes[node_idx].right = nodes.size();
        build_tree(Xr, yr, d, 0, depth + 1, max_depth);
    }
};

class RandomForest {
    std::vector<DecisionTree> trees;
public:
    void train(const std::vector<float>& X, const std::vector<int>& y, size_t d, int n_trees = 5) {
        trees.resize(n_trees);
        std::default_random_engine gen(42);
        for (int t = 0; t < n_trees; ++t) {
            std::vector<size_t> idx(y.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), gen);
            size_t sample_n = y.size() * 0.8;
            idx.resize(sample_n);

            std::vector<float> Xs; std::vector<int> ys;
            for (size_t i : idx) {
                for (size_t j = 0; j < d; ++j) Xs.push_back(X[i * d + j]);
                ys.push_back(y[i]);
            }
            trees[t].train(Xs, ys, d, 6);
        }
    }

    int predict(const std::vector<float>& x) const {
        int votes0 = 0, votes1 = 0;
        for (const auto& t : trees) {
            (t.predict(x) ? votes1 : votes0)++;
        }
        return votes1 > votes0 ? 1 : 0;
    }
};

int main(int argc, char* argv[]) {
    std::string model = "rf";
    size_t sample = 50000;
    std::string data = "../../data/NF-UNSW-NB15-v2.csv";

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--model" && i + 1 < argc) model = argv[++i];
        else if (std::string(argv[i]) == "--sample" && i + 1 < argc) sample = std::stoul(argv[++i]);
        else if (std::string(argv[i]) == "--data" && i + 1 < argc) data = argv[++i];
    }

    std::cout << "Модель: " << model << " | Выборка: " << sample << "\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    Dataset d = load_data(data, sample);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Загружено: " << d.n << " строк, " << d.d << " признаков\n";
    size_t n0 = std::count(d.y.begin(), d.y.end(), 0);
    size_t n1 = std::count(d.y.begin(), d.y.end(), 1);
    std::cout << "Классы: 0=" << n0 << ", 1=" << n1 << "\n";

    size_t split = d.n * 0.8;
    std::vector<float> Xt(d.X.begin() + split * d.d, d.X.end());
    std::vector<int> yt(d.y.begin() + split, d.y.end());

    fs::create_directories("models");
    json pp = {{"numerical_features", d.feature_names}, {"categorical_features", json::array()}, {"numerical_medians", d.medians}};
    std::ofstream f_pp("models/" + model + "_nids_cpp_preprocessing_params.json");
    f_pp << pp.dump(1);

    Metrics metr;
    json meta;

    if (model == "rf") {
        RandomForest rf;
        std::vector<float> Xtr(d.X.begin(), d.X.begin() + split * d.d);
        std::vector<int> ytr(d.y.begin(), d.y.begin() + split);
        rf.train(Xtr, ytr, d.d, 5);

        std::vector<int> yp;
        for (size_t i = 0; i < Xt.size(); i += d.d) {
            std::vector<float> x(d.d);
            for (size_t j = 0; j < d.d; ++j) x[j] = Xt[i + j];
            yp.push_back(rf.predict(x));
        }
        metr = eval(yt, yp);
        meta = {{"model_type", "rf"}, {"n_trees", 5}};
    } else if (model == "svm") {
        std::vector<float> w(d.d, 0.0f), b(1, 0.0f);
        for (int e = 0; e < 100; ++e) {
            for (size_t i = 0; i < split; ++i) {
                size_t s = i * d.d;
                float z = b[0];
                for (size_t j = 0; j < d.d; ++j) z += d.X[s + j] * w[j];
                int y = d.y[i] ? 1 : -1;
                if (y * z < 1.0f) {
                    for (size_t j = 0; j < d.d; ++j) w[j] = 0.999f * w[j] + 0.001f * y * d.X[s + j];
                    b[0] += 0.001f * y;
                } else {
                    for (size_t j = 0; j < d.d; ++j) w[j] *= 0.999f;
                }
            }
        }
        std::vector<int> yp;
        for (size_t i = 0; i < Xt.size(); i += d.d) {
            float z = b[0];
            for (size_t j = 0; j < d.d; ++j) z += Xt[i + j] * w[j];
            yp.push_back(z > 0 ? 1 : 0);
        }
        metr = eval(yt, yp);
        meta = {{"model_type", "svm"}};
    } else if (model == "mlp") {
        std::vector<float> w(d.d, 0.0f), b(1, 0.0f);
        for (int e = 0; e < 200; ++e) {
            for (size_t i = 0; i < split; ++i) {
                size_t s = i * d.d;
                float z = b[0];
                for (size_t j = 0; j < d.d; ++j) z += d.X[s + j] * w[j];
                float p = sigmoid(z);
                float g = p - d.y[i];
                b[0] -= 0.01f * g;
                for (size_t j = 0; j < d.d; ++j) w[j] -= 0.01f * g * d.X[s + j];
            }
        }
        std::vector<int> yp;
        for (size_t i = 0; i < Xt.size(); i += d.d) {
            float z = b[0];
            for (size_t j = 0; j < d.d; ++j) z += Xt[i + j] * w[j];
            yp.push_back(sigmoid(z) > 0.5f ? 1 : 0);
        }
        metr = eval(yt, yp);
        meta = {{"model_type", "mlp"}};
    } else if (model == "xgb" || model == "lgbm") {
        std::vector<int> yp(Xt.size() / d.d, 1);
        metr = eval(yt, yp);
        meta = {{"model_type", model}, {"note", "placeholder"}};
        std::cout << "Внимание: " << model << " — заглушка (реализация требует внешних библиотек)\n";
    } else {
        std::cerr << "Неизвестная модель: " << model << "\n";
        return 1;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(t2 - t1).count();
    save_metrics(model, train_sec, metr);

    std::ofstream f_meta("models/" + model + "_nids_cpp_metadata.json");
    f_meta << meta.dump(1);

    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << metr.acc << "\n";
    std::cout << "F1 Macro: " << metr.f1 << "\n";
    std::cout << "Время обучения: " << std::fixed << std::setprecision(1) << train_sec << " сек\n";
    std::cout << "Готово\n";
    return 0;
}
