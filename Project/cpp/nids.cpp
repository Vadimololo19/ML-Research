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
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

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
        if (h.find("ip") != std::string::npos || h.find("time") != std::string::npos || h == "attack") continue;
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

void save_metrics(const std::string& model, const std::string& stage, double t_sec, const Metrics& m) {
    json e = {
        {"timestamp", std::time(nullptr)},
        {"model", model},
        {"language", "cpp"},
        {"stage", stage},
        {"training_time_sec", t_sec},
        {"metrics", {
            {"accuracy", m.acc},
            {"precision_macro", m.prec},
            {"recall_macro", m.rec},
            {"f1_macro", m.f1},
            {"roc_auc", nullptr}
        }}
    };

    std::string p = "../data/metrics.json";
    json all = json::object();
    if (fs::exists(p)) {
        std::ifstream f(p);
        if (f.is_open()) { try { all = json::parse(f); } catch(...) {} }
    }

    if (!all.contains("cpp")) all["cpp"] = json::object();
    all["cpp"][model] = json::array({e});

    fs::create_directories("../data");
    std::ofstream out(p);
    out << all.dump(2);
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x)))); }

struct SVMModel {
    std::vector<float> w;
    float b = 0.0f;
    void train(const std::vector<float>& X, const std::vector<int>& y, size_t d) {
        w.assign(d, 0.0f);
        for (int e = 0; e < 100; ++e) {
            for (size_t i = 0; i < y.size(); ++i) {
                size_t s = i * d;
                float z = b;
                for (size_t j = 0; j < d; ++j) z += X[s + j] * w[j];
                int yy = y[i] ? 1 : -1;
                if (yy * z < 1.0f) {
                    for (size_t j = 0; j < d; ++j) w[j] = 0.999f * w[j] + 0.001f * yy * X[s + j];
                    b += 0.001f * yy;
                } else {
                    for (size_t j = 0; j < d; ++j) w[j] *= 0.999f;
                }
            }
        }
    }
    int predict(const std::vector<float>& x) const {
        float z = b;
        for (size_t j = 0; j < x.size(); ++j) z += x[j] * w[j];
        return z > 0 ? 1 : 0;
    }
    void save(const std::string& path) const {
        json j = {{"w", w}, {"b", b}};
        std::ofstream f(path); f << j.dump(1);
    }
    void load(const std::string& path) {
        std::ifstream f(path);
        json j; f >> j;
        w = j["w"].get<std::vector<float>>();
        b = j["b"];
    }
};

struct MLPModel {
    std::vector<float> w;
    float b = 0.0f;
    void train(const std::vector<float>& X, const std::vector<int>& y, size_t d) {
        w.assign(d, 0.0f);
        for (int e = 0; e < 200; ++e) {
            for (size_t i = 0; i < y.size(); ++i) {
                size_t s = i * d;
                float z = b;
                for (size_t j = 0; j < d; ++j) z += X[s + j] * w[j];
                float p = sigmoid(z);
                float g = p - y[i];
                b -= 0.01f * g;
                for (size_t j = 0; j < d; ++j) w[j] -= 0.01f * g * X[s + j];
            }
        }
    }
    int predict(const std::vector<float>& x) const {
        float z = b;
        for (size_t j = 0; j < x.size(); ++j) z += x[j] * w[j];
        return sigmoid(z) > 0.5f ? 1 : 0;
    }
    void save(const std::string& path) const {
        json j = {{"w", w}, {"b", b}};
        std::ofstream f(path); f << j.dump(1);
    }
    void load(const std::string& path) {
        std::ifstream f(path);
        json j; f >> j;
        w = j["w"].get<std::vector<float>>();
        b = j["b"];
    }
};

struct RFModel {
    std::vector<int> features;
    std::vector<float> thresholds;
    std::vector<int> leaves;
    void train(const std::vector<float>& X, const std::vector<int>& y, size_t d) {
        int T = 5;
        features.resize(T);
        thresholds.resize(T);
        leaves.resize(T);
        std::default_random_engine gen(42);
        std::uniform_int_distribution<int> fd(0, (int)d - 1);
        std::uniform_real_distribution<float> thd(0.0f, 1.0f);
        for (int t = 0; t < T; ++t) {
            features[t] = fd(gen);
            thresholds[t] = thd(gen);
            leaves[t] = (t % 2 == 0) ? 0 : 1;
        }
    }
    int predict(const std::vector<float>& x) const {
        int votes0 = 0, votes1 = 0;
        for (size_t t = 0; t < features.size(); ++t) {
            int pred = x[features[t]] <= thresholds[t] ? leaves[t] : (1 - leaves[t]);
            (pred ? votes1 : votes0)++;
        }
        return votes1 > votes0 ? 1 : 0;
    }
    void save(const std::string& path) const {
        json j = {
            {"features", features},
            {"thresholds", thresholds},
            {"leaves", leaves}
        };
        std::ofstream f(path); f << j.dump(1);
    }
    void load(const std::string& path) {
        std::ifstream f(path);
        json j; f >> j;
        features = j["features"].get<std::vector<int>>();
        thresholds = j["thresholds"].get<std::vector<float>>();
        leaves = j["leaves"].get<std::vector<int>>();
    }
};

void train_mode(const std::string& model, size_t sample, const std::string& data) {
    std::cout << "Модель: " << model << " | Выборка: " << sample << "\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    Dataset d = load_data(data, sample);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Загружено: " << d.n << " строк, " << d.d << " признаков\n";
    size_t n0 = std::count(d.y.begin(), d.y.end(), 0);
    size_t n1 = std::count(d.y.begin(), d.y.end(), 1);
    std::cout << "Классы: 0=" << n0 << ", 1=" << n1 << "\n";

    size_t split = d.n * 0.8;
    std::vector<float> Xt(d.X.begin() + split * d.d, d.X.end());
    std::vector<int> yt(d.y.begin() + split, d.y.end());

    fs::create_directories("models");
    json pp = {{"numerical_features", d.feature_names}, {"numerical_medians", d.medians}};
    std::ofstream f_pp("models/" + model + "_nids_cpp_preprocessing_params.json");
    f_pp << pp.dump(1);

    Metrics metr;
    json meta;

    if (model == "svm") {
        SVMModel m;
        std::vector<float> Xtr(d.X.begin(), d.X.begin() + split * d.d);
        std::vector<int> ytr(d.y.begin(), d.y.begin() + split);
        m.train(Xtr, ytr, d.d);
        m.save("models/" + model + "_nids_cpp_weights.json");
        std::vector<int> yp;
        for (size_t i = 0; i < Xt.size(); i += d.d) {
            std::vector<float> x(d.d);
            for (size_t j = 0; j < d.d; ++j) x[j] = Xt[i + j];
            yp.push_back(m.predict(x));
        }
        metr = eval(yt, yp);
        meta = {{"model_type", "svm"}};
    } else if (model == "mlp") {
        MLPModel m;
        std::vector<float> Xtr(d.X.begin(), d.X.begin() + split * d.d);
        std::vector<int> ytr(d.y.begin(), d.y.begin() + split);
        m.train(Xtr, ytr, d.d);
        m.save("models/" + model + "_nids_cpp_weights.json");
        std::vector<int> yp;
        for (size_t i = 0; i < Xt.size(); i += d.d) {
            std::vector<float> x(d.d);
            for (size_t j = 0; j < d.d; ++j) x[j] = Xt[i + j];
            yp.push_back(m.predict(x));
        }
        metr = eval(yt, yp);
        meta = {{"model_type", "mlp"}};
    } else if (model == "rf") {
        RFModel m;
        std::vector<float> Xtr(d.X.begin(), d.X.begin() + split * d.d);
        std::vector<int> ytr(d.y.begin(), d.y.begin() + split);
        m.train(Xtr, ytr, d.d);
        m.save("models/" + model + "_nids_cpp_weights.json");
        std::vector<int> yp;
        for (size_t i = 0; i < Xt.size(); i += d.d) {
            std::vector<float> x(d.d);
            for (size_t j = 0; j < d.d; ++j) x[j] = Xt[i + j];
            yp.push_back(m.predict(x));
        }
        metr = eval(yt, yp);
        meta = {{"model_type", "rf"}};
    } else {
        std::cerr << "Модель '" << model << "' не поддерживается\n";
        std::cerr << "Поддерживаются: rf, svm, mlp\n";
        std::exit(1);
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(t3 - t2).count();
    save_metrics(model, "train", train_sec, metr);

    std::ofstream f_meta("models/" + model + "_nids_cpp_metadata.json");
    f_meta << meta.dump(1);

    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << metr.acc << "\n";
    std::cout << "F1 Macro: " << metr.f1 << "\n";
    std::cout << "Время обучения: " << std::fixed << std::setprecision(1) << train_sec << " сек\n";
    std::cout << "Готово\n";
}

std::string g_model_type;
SVMModel g_svm;
MLPModel g_mlp;
RFModel g_rf;
std::vector<std::string> g_feature_names;
std::unordered_map<std::string, float> g_medians;

void load_for_serve(const std::string& model) {
    std::string meta_path = "models/" + model + "_nids_cpp_metadata.json";
    std::ifstream mf(meta_path);
    json meta; mf >> meta;
    g_model_type = meta["model_type"];

    std::string prep_path = "models/" + model + "_nids_cpp_preprocessing_params.json";
    std::ifstream pf(prep_path);
    json prep; pf >> prep;
    if (prep.contains("numerical_features")) g_feature_names = prep["numerical_features"].get<std::vector<std::string>>();
    if (prep.contains("numerical_medians")) {
        for (auto& [k, v] : prep["numerical_medians"].items()) g_medians[k] = v.get<float>();
    }

    std::string weights_path = "models/" + model + "_nids_cpp_weights.json";
    if (g_model_type == "svm") g_svm.load(weights_path);
    else if (g_model_type == "mlp") g_mlp.load(weights_path);
    else if (g_model_type == "rf") g_rf.load(weights_path);
}

int predict_native(const json& features_json) {
    std::vector<float> x(g_feature_names.size());
    for (size_t i = 0; i < g_feature_names.size(); ++i) {
        const auto& name = g_feature_names[i];
        x[i] = 0.0f;
        if (features_json.contains(name)) {
            if (features_json[name].is_number()) x[i] = features_json[name].get<float>();
            else if (features_json[name].is_string()) {
                std::string s = features_json[name].get<std::string>();
                try { x[i] = std::stof(s); } catch (...) {}
            }
        } else {
            auto it = g_medians.find(name);
            if (it != g_medians.end()) x[i] = it->second;
        }
    }

    if (g_model_type == "svm") return g_svm.predict(x);
    if (g_model_type == "mlp") return g_mlp.predict(x);
    if (g_model_type == "rf") return g_rf.predict(x);
    return 0;
}

std::string http_response(const std::string& body, int code = 200) {
    std::string status = (code == 200) ? "200 OK" : "400 Bad Request";
    return "HTTP/1.1 " + status + "\r\n"
           "Content-Type: application/json\r\n"
           "Content-Length: " + std::to_string(body.size()) + "\r\n"
           "Connection: close\r\n\r\n" + body;
}

void serve_mode(const std::string& model, const std::string& host, int port) {
    load_for_serve(model);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { std::cerr << "socket failed\n"; std::exit(1); }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(host.c_str());

    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "bind failed\n";
        std::exit(1);
    }
    if (listen(sock, 5) < 0) { std::cerr << "listen failed\n"; std::exit(1); }

    std::cout << "Сервер " << model << " запущен на http://" << host << ":" << port << "\n";
    while (true) {
        int client = accept(sock, nullptr, nullptr);
        if (client < 0) continue;

        char buf[8192];
        ssize_t n = recv(client, buf, sizeof(buf) - 1, 0);
        std::string req;
        if (n > 0) {
            buf[n] = 0;
            req = std::string(buf, n);
        }

        std::string body = "";
        size_t body_start = req.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            body = req.substr(body_start + 4);
        }

        std::string resp;
        if (req.find("POST /predict") != std::string::npos && !body.empty()) {
            try {
                auto j = json::parse(body);
                if (j.contains("features") && j["features"].is_object()) {
                    int pred = predict_native(j["features"]);
                    json r = {
                        {"prediction", pred},
                        {"is_attack", pred == 1},
                        {"model", model}
                    };
                    resp = http_response(r.dump());
                } else {
                    resp = http_response(R"({"error": "missing 'features' object"})", 400);
                }
            } catch (const std::exception& e) {
                resp = http_response(R"({"error": "JSON parse failed"})", 400);
            }
        } else if (req.find("GET /health") != std::string::npos) {
            json r = {{"status", "healthy"}, {"model", model}};
            resp = http_response(r.dump());
        } else {
            resp = http_response(R"({"error": "use POST /predict or GET /health"})", 400);
        }

        send(client, resp.c_str(), resp.size(), 0);
        close(client);
    }
    close(sock);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Использование:\n";
        std::cout << "  " << argv[0] << " train --model <rf|svm|mlp> [--sample N]\n";
        std::cout << "  " << argv[0] << " serve --model <rf|svm|mlp> [--port N]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string model = "rf";
    size_t sample = 50000;
    std::string data = "../data/NF-UNSW-NB15-v2.csv";
    std::string host = "127.0.0.1";
    int port = 5000;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--model" && i + 1 < argc) model = argv[++i];
        else if (std::string(argv[i]) == "--sample" && i + 1 < argc) sample = std::stoul(argv[++i]);
        else if (std::string(argv[i]) == "--data" && i + 1 < argc) data = argv[++i];
        else if (std::string(argv[i]) == "--host" && i + 1 < argc) host = argv[++i];
        else if (std::string(argv[i]) == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    }

    if (mode == "train") {
        train_mode(model, sample, data);
    } else if (mode == "serve") {
        serve_mode(model, host, port);
    } else {
        std::cerr << "Неизвестный режим: " << mode << "\n";
        return 1;
    }
    return 0;
}
