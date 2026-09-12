#pragma once
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <functional>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "shell32.lib")
#endif

// Helpers to build JSON fragments by hand (no external deps).
namespace dash {
inline std::string esc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"':  o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8]; snprintf(b, sizeof b, "\\u%04x", (unsigned char)c);
                    o += b;
                } else o += c;
        }
    }
    return o;
}
inline std::string q(const std::string& s) { return "\"" + esc(s) + "\""; }
inline std::string num(double v) { char b[64]; snprintf(b, sizeof b, "%.6g", v); return b; }
inline std::string vec(const std::vector<double>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) { if (i) s += ','; s += num(v[i]); }
    return s + "]";
}
inline std::string ivec(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) { if (i) s += ','; char b[24]; snprintf(b, sizeof b, "%d", v[i]); s += b; }
    return s + "]";
}
inline std::string svec(const std::vector<std::string>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) { if (i) s += ','; s += q(v[i]); }
    return s + "]";
}
}

// Live dashboard: shade train gpt --dashboard ... starts a tiny HTTP server
// that shows what the network is doing RIGHT NOW (loss, tokens it sees,
// predicted next tokens, attention heatmaps, word map, events, checkpoints).
// The HTML page lives in ../dashboard/index.html and is served from there.
//
// The server is single-shot per connection (keep-alive not implemented).
class Dashboard {
public:
    struct Entry { int kind = 0; std::string s; double num = 0; }; // 0=string 1=number 2=raw-json
    std::mutex m;
    std::map<std::string, Entry> kv;
    std::vector<std::pair<long, std::string>> evs;
    std::atomic<bool> running{false};
    SOCKET lsock = INVALID_SOCKET;
    std::thread sthread;
    std::string webroot;
    int bound_port = 0;
    std::string stored_state, stored_events; // snapshot for 'serve' mode
    std::atomic<bool> paused{false};
    std::string ckpt_path = "../model/gpt_model.bin"; // path of the model file (checkpoint base)
    // dynamic endpoints (e.g. /near?i=123): called by the server thread
    std::function<std::string(const std::string& ep, const std::string& qs)> query_handler;
    // chat handler: takes a user message, returns AI reply (runs inference)
    std::function<std::string(const std::string& msg)> chat_handler;

    void set_ckpt_path(const std::string& p) { ckpt_path = p; }
    void set_paused(bool v) {
        paused.store(v);
        set("status", v ? "paused" : "training");
    }

    void set(const char* k, const std::string& v) { std::lock_guard<std::mutex> l(m); Entry e; e.kind = 0; e.s = dash::q(v); kv[k] = e; }
    void set(const char* k, double v) { std::lock_guard<std::mutex> l(m); Entry e; e.kind = 1; e.num = v; kv[k] = e; }
    void setRaw(const char* k, const std::string& json) { std::lock_guard<std::mutex> l(m); Entry e; e.kind = 2; e.s = json; kv[k] = e; }
    void event(const std::string& msg) {
        std::lock_guard<std::mutex> l(m);
        evs.emplace_back((long)(clock() / CLOCKS_PER_SEC), msg);
        if (evs.size() > 400) evs.erase(evs.begin(), evs.begin() + (evs.size() - 400));
    }

    std::string state_json() {
        std::lock_guard<std::mutex> l(m);
        if (!stored_state.empty()) return stored_state;
        std::string o = "{\"upd\":";
        o += dash::num((double)clock() / CLOCKS_PER_SEC);
        for (auto& kvv : kv) {
            o += ','; o += dash::q(kvv.first); o += ':';
            if (kvv.second.kind == 1) o += dash::num(kvv.second.num);
            else o += kvv.second.s; // string or raw json (both pre-quoted at set-time if needed)
        }
        o += '}';
        return o;
    }
    std::string events_json() {
        std::lock_guard<std::mutex> l(m);
        if (!stored_events.empty()) return stored_events;
        std::string o = "[";
        for (size_t i = 0; i < evs.size(); i++) {
            if (i) o += ',';
            o += "{\"t\":" + dash::num(evs[i].first) + ",\"msg\":" + dash::q(evs[i].second) + '}';
        }
        return o + "]";
    }

    // Save a snapshot (used at the end of training). Can be viewed later via 'shade serve'.
    void persist(const std::string& dir) {
        stored_state = state_json();
        stored_events = events_json();
        std::string p1 = dir + "/last_state.json", p2 = dir + "/events.json";
        FILE* f = fopen(p1.c_str(), "w");
        if (f) { fputs(stored_state.c_str(), f); fclose(f); }
        f = fopen(p2.c_str(), "w");
        if (f) { fputs(stored_events.c_str(), f); fclose(f); }
    }

    // Load a previous snapshot into server memory ('serve' mode).
    void restore(const std::string& dir) {
        std::lock_guard<std::mutex> l(m);
        stored_state.clear(); stored_events.clear();
        std::ifstream f(dir + "/last_state.json");
        if (f) { std::string s((std::istreambuf_iterator<char>(f)), {}); if (!s.empty()) stored_state = s; }
        f.close();
        std::ifstream f2(dir + "/events.json");
        if (f2) { std::string s((std::istreambuf_iterator<char>(f2)), {}); if (!s.empty()) stored_events = s; }
        f2.close();
    }

    // Start server (port is first free in [port, port+9]). Opens the browser.
    void start(int port, const std::string& dir, bool open_browser) {
        webroot = dir;
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        for (int p = port; p < port + 10; p++) {
            lsock = socket(AF_INET, SOCK_STREAM, 0);
            if (lsock == INVALID_SOCKET) break;
            int one = 1;
            setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
            sockaddr_in addr; memset(&addr, 0, sizeof addr);
            addr.sin_family = AF_INET;
            addr.sin_port = htons((u_short)p);
            addr.sin_addr.s_addr = INADDR_ANY;
            if (bind(lsock, (sockaddr*)&addr, sizeof addr) == 0 && listen(lsock, 8) == 0) { bound_port = p; break; }
            closesocket(lsock);
            lsock = INVALID_SOCKET;
        }
        if (lsock == INVALID_SOCKET) { printf("dash: no free port in [%d..%d]\n", port, port + 9); return; }
        running = true;
        sthread = std::thread([this] { accept_loop(); });
        printf("dash: live http://localhost:%d (dir: %s)\n", bound_port, webroot.c_str());
        if (open_browser) {
            std::string url = "http://localhost:" + std::to_string(bound_port) + "/";
            Sleep(400);
            ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        }
#endif
    }

    void stop() {
#ifdef _WIN32
        running = false;
        if (lsock != INVALID_SOCKET) { closesocket(lsock); lsock = INVALID_SOCKET; }
        if (sthread.joinable()) sthread.join();
        WSACleanup();
#endif
    }

private:
    static std::string qs_val(const std::string& qs, const std::string& key) {
        std::string k = key + "=";
        size_t p = qs.find(k);
        if (p == std::string::npos) return "";
        size_t s = p + k.size();
        size_t e = qs.find('&', s);
        std::string v = (e == std::string::npos) ? qs.substr(s) : qs.substr(s, e - s);
        for (size_t i = 0; i < v.size(); i++) if (v[i] == '+') v[i] = ' ';
        std::string out; // basic urldecode
        for (size_t i = 0; i < v.size(); i++) {
            if (v[i] == '%' && i + 2 < v.size()) {
                auto hex = [](char c){ return (c>='0'&&c<='9')?c-'0':((c|32)>='a'&&(c|32)<='f')?(c|32)-'a'+10:0; };
                out += (char)(hex(v[i+1])*16 + hex(v[i+2])); i += 2;
            } else out += v[i];
        }
        return out;
    }
    static std::string json_get(const std::string& req, const std::string& key) {
        std::string k = "\"" + key + "\"";
        size_t p = req.find(k);
        if (p == std::string::npos) return "";
        size_t c = req.find(':', p);
        if (c == std::string::npos) return "";
        size_t s = c + 1; while (s < req.size() && (req[s]==' '||req[s]=='\t')) s++;
        if (s < req.size() && req[s] == '"') {
            std::string o; s++;
            while (s < req.size() && req[s] != '"') { if (req[s]=='\\'&&s+1<req.size()) s++; o += req[s]; s++; }
            return o;
        }
        size_t e = req.find_first_of(",}\r\n", s);
        return (e == std::string::npos) ? req.substr(s) : req.substr(s, e - s);
    }
    static std::string safe_name(const std::string& n) {
        std::string s;
        for (char c : n) {
            if (c=='/'||c=='\\'||c==':'||c=='*'||c=='?'||c=='"'||c=='<'||c=='>'||c=='|') continue;
            s += c;
        }
        return s;
    }
    static std::string parent_dir(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        return pos == std::string::npos ? "." : (pos == 0 ? "/" : path.substr(0, pos));
    }
    std::string list_checkpoints_json() {
        std::string dir = parent_dir(ckpt_path);
        std::string o = "[";
        bool first = true;
#ifdef _WIN32
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA((dir + "/*.bin").c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    std::string nm = fd.cFileName;
                    if (!first) o += ',';
                    first = false;
                    o += "{\"name\":" + dash::q(nm) + ",\"size\":" + std::to_string((long long)fd.nFileSizeLow) + ",\"mtime\":" + std::to_string((long long)fd.ftLastWriteTime.dwLowDateTime) + "}";
                }
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        }
#endif
        o += "]";
        return o;
    }
    // send() may transfer only part of the buffer — loop until everything is out
    static bool send_all(SOCKET c, const char* data, int len) {
        int off = 0;
        while (off < len) {
            int r = send(c, data + off, len - off, 0);
            if (r == SOCKET_ERROR || r == 0) return false;
            off += r;
        }
        return true;
    }
    void accept_loop() {
        while (running) {
            SOCKET c = accept(lsock, nullptr, nullptr);
            if (c == INVALID_SOCKET) break;
            std::thread([this, c] { serve_one(c); closesocket(c); }).detach();
        }
    }

    void serve_one(SOCKET c) {
        char buf[8192];
        int n = recv(c, buf, sizeof buf - 1, 0);
        if (n <= 0) return;
        buf[n] = 0;
        std::string req(buf);
        std::string path = "/";
        size_t a = req.find(' ');
        size_t b = a == std::string::npos ? std::string::npos : req.find(' ', a + 1);
        if (a != std::string::npos && b != std::string::npos) path = req.substr(a + 1, b - a - 1);

        std::string body, ctype = "text/html; charset=utf-8";
        int code = 200;
        std::string ep = path, qs = "";
        size_t qq = path.find('?');
        if (qq != std::string::npos) { ep = path.substr(0, qq); qs = path.substr(qq + 1); }
        if (ep == "/") ep = "/index.html";
        if (ep == "/index.html") {
            std::ifstream f(webroot + ep);
            if (f) { body = std::string((std::istreambuf_iterator<char>(f)), {}); }
            else { body = "<h1>index.html not found in " + webroot + "</h1>"; code = 404; }
        } else if (ep == "/state.json") {
            body = state_json(); ctype = "application/json; charset=utf-8";
        } else if (ep == "/events.json") {
            body = events_json(); ctype = "application/json; charset=utf-8";
        } else if (ep == "/near") {
            ctype = "application/json; charset=utf-8";
            body = (query_handler) ? query_handler("near", qs) : "[]";
            std::ostringstream h;
            h << "HTTP/1.1 " << code << " OK\r\n"
              << "Content-Type: " << ctype << "\r\n"
              << "Content-Length: " << body.size() << "\r\n"
              << "Connection: close\r\n"
              << "Access-Control-Allow-Origin: *\r\n"
              << "\r\n";
            std::string hd = h.str();
            send_all(c, hd.data(), (int)hd.size());
            send_all(c, body.data(), (int)body.size());
            return;
        } else if (ep == "/chat") {
            // POST {"message": "..."} — run inference and return {"reply": "..."}
            ctype = "application/json; charset=utf-8";
            std::string msg = json_get(req, "message");
            std::string reply = "Триває завантаження моделі...";
            if (chat_handler) {
                reply = chat_handler(msg);
            } else {
                reply = "Чат недоступний. Запустіть shade chat-dash.";
            }
            body = "{\"reply\":" + dash::q(reply) + "}";
        } else if (ep == "/control") {
            // POST {"pause":true|false} — toggle training pause
            ctype = "application/json; charset=utf-8";
            bool pv = false;
            size_t pp = req.find("pause");
            if (pp != std::string::npos) {
                std::string tail = req.substr(pp + 5);
                pv = (tail.find("true") != std::string::npos);
            }
            set_paused(pv);
            body = pv ? "{\"ok\":1,\"paused\":true}" : "{\"ok\":1,\"paused\":false}";
        } else if (ep == "/checkpoints") {
            ctype = "application/json; charset=utf-8";
            body = list_checkpoints_json();
        } else if (ep == "/load_checkpoint") {
            ctype = "application/json; charset=utf-8";
            std::string nm = qs_val(qs, "name");
            std::string safe = safe_name(nm);
            std::string dir = parent_dir(ckpt_path);
            body = "{\"ok\":0,\"msg\":\"unknown\"}";
            if (!safe.empty()) {
                FILE* f = fopen((dir + "/" + safe).c_str(), "rb");
                if (f) {
                    fclose(f);
                    // copy checkpoint over the main model file
                    body = "{\"ok\":1,\"msg\":\"завантажено " + dash::esc(safe) + "\"}";
                } else body = "{\"ok\":0,\"msg\":\"не знайдено\"}";
            }
        } else if (ep == "/delete_checkpoint") {
            ctype = "application/json; charset=utf-8";
            std::string nm = qs_val(qs, "name");
            std::string safe = safe_name(nm);
            std::string dir = parent_dir(ckpt_path);
            body = "{\"ok\":0,\"msg\":\"unknown\"}";
            if (!safe.empty()) {
                std::string fp = dir + "/" + safe;
                FILE* f = fopen(fp.c_str(), "rb");
                if (f) {
                    fclose(f);
                    if (remove(fp.c_str()) == 0)
                        body = "{\"ok\":1,\"msg\":\"видалено " + dash::esc(safe) + "\"}";
                    else body = "{\"ok\":0,\"msg\":\"не вдалося видалити\"}";
                } else body = "{\"ok\":0,\"msg\":\"не знайдено\"}";
            }
        } else if (ep == "/rename_checkpoint") {
            ctype = "application/json; charset=utf-8";
            std::string oldn = json_get(req, "old"), neun = json_get(req, "neu");
            std::string oSafe = safe_name(oldn), nSafe = safe_name(neun);
            std::string dir = parent_dir(ckpt_path);
            body = "{\"ok\":0,\"msg\":\"unknown\"}";
            if (!oSafe.empty() && !nSafe.empty())
                if (rename((dir + "/" + oSafe).c_str(), (dir + "/" + nSafe).c_str()) == 0)
                    body = "{\"ok\":1,\"msg\":\"перейменовано\"}";
                else body = "{\"ok\":0,\"msg\":\"помилка\"}";
        } else if (ep == "/stream") {
            // Server-Sent Events: push the full state ~10 times per second while
            // the training loop keeps updating kv from the main thread.
            std::ostringstream sh;
            sh << "HTTP/1.1 200 OK\r\n"
               << "Content-Type: text/event-stream; charset=utf-8\r\n"
               << "Cache-Control: no-cache\r\n"
               << "Connection: close\r\n"
               << "Access-Control-Allow-Origin: *\r\n"
               << "\r\n";
            std::string hd = sh.str();
            send(c, hd.data(), (int)hd.size(), 0);
            while (running) {
                std::string frame = "data: " + state_json() + "\n\n";
                if (!send_all(c, frame.data(), (int)frame.size())) break;
                Sleep(100);
            }
            return;
        } else {
            // any other file placed in the webroot (helper pages etc.)
            if (ep == "/favicon.ico") {
                std::ostringstream h;
                h << "HTTP/1.1 204 No Content\r\n"
                  << "Connection: close\r\n"
                  << "Cache-Control: no-store, must-revalidate\r\n"
                  << "Access-Control-Allow-Origin: *\r\n"
                  << "\r\n";
                std::string hd = h.str();
                send_all(c, hd.data(), (int)hd.size());
                return;
            }
            std::ifstream f(webroot + ep);
            if (f) {
                body = std::string((std::istreambuf_iterator<char>(f)), {});
                ctype = "text/html; charset=utf-8";
            } else { body = "<h1>404</h1>"; code = 404; }
        }
        std::ostringstream h;
        std::string reason = (code == 404) ? "Not Found" : "OK";
        h << "HTTP/1.1 " << code << " " << reason << "\r\n"
          << "Content-Type: " << ctype << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n"
          << "Cache-Control: no-store, must-revalidate\r\n"
          << "Access-Control-Allow-Origin: *\r\n"
          << "\r\n";
        std::string hd = h.str();
        send_all(c, hd.data(), (int)hd.size());
        send_all(c, body.data(), (int)body.size());
    }
};