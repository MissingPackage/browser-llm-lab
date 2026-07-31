// oracle-moe/trace.cpp — ROUTE_TRACE + LOOKA online sull'oracolo llama.cpp
// (spec engine-fase-c1, §Strumentazione). Zero patch upstream: osserva i tensori
// nominati via cb_eval e legge i pesi del router direttamente dal GGUF (gguf.h).
//
// - ROUTE_TRACE: ffn_moe_topk-<il> -> traccia (posizione, layer, top-4).
// - LOOKA (decisioni (a)/(b) del ruling): al tap ffn_norm-<L> replica la selezione
//   sigmoid(W_router·h)+exp_probs_b -> top-K per: (autotest) lo stesso layer L,
//   gate hard recall>=0.999; (lookahead) il layer L+1, recall@{4,6,8} vs topk vero.
// - baseline_prev: overlap top-4 tra posizioni decode consecutive (solo traccia).
//
// Build: tools/oracle-moe/build.sh   Run: tools/oracle-moe/run-trace.sh
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <array>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>

struct PosRec {
    int32_t tok;
    char phase; // 'p' prefill, 'd' decode
    std::vector<std::array<int32_t,4>> topk; // indicizzato per slot moe (il - dense_lead)
    // predizioni top-8 per layer target, SOLO sulle posizioni di decode: sono
    // l'input del replay di prefetch in fase 5 (spec: replay delle predizioni
    // vere, mai un recall sintetico). Vuoto sul prefill (traccia 2x piu' magra).
    std::vector<std::array<uint8_t,8>> pred;
};

// pesi router + contatori LOOKA (indici per il assoluto)
struct Looka {
    int n_embd = 0, n_layer = 0, dense_lead = 0;
    std::vector<std::vector<float>> W;    // [il] -> n_expert*n_embd (riga = expert)
    std::vector<std::vector<float>> bias; // [il] -> n_expert
    // contatori: self (autotest, tutte le fasi), next per fase, baseline decode
    struct PerLayer {
        int64_t selfHit = 0, selfTot = 0;
        int64_t nextTot[2] = {0,0};                 // [0]=prefill, [1]=decode (unita': 4 slot veri)
        int64_t nextHit[2][3] = {{0,0,0},{0,0,0}};  // K in {4,6,8}
        int64_t prevOverlap = 0, prevTot = 0;       // baseline decode
        int64_t extraTot = 0, extraHit8 = 0;        // target il=1 predetto dal layer denso (extra, non in spec)
    };
    std::vector<PerLayer> pl; // [il]
};

struct CbData {
    int dense_lead = 0;
    int n_moe = 0;
    int n_layer = 0;
    int64_t base_pos = 0;
    int expect_cols = 0;
    char phase = 'p';
    std::vector<PosRec> * recs = nullptr;
    std::map<int,int> cols_seen;
    std::vector<uint8_t> tmp;
    bool fail = false;
    std::string fail_msg;
    Looka * lk = nullptr;
    // predizioni del batch corrente: [target il][col] -> top-8 ordinato
    std::vector<std::vector<std::array<uint8_t,8>>> pred_next;
    std::vector<std::vector<std::array<uint8_t,4>>> pred_self;
    // baseline: ultimo top-4 decode visto, per layer (reset a ogni prompt)
    std::vector<std::array<int32_t,4>> prev_top4;
    std::vector<bool> prev_valid;
};

static void topk_from_scores(const float * s, int n, int k, uint8_t * out) {
    // selezione top-k per punteggio decrescente (pareggi: indice minore, come argsort stabile)
    std::array<uint8_t,64> idx;
    for (int i = 0; i < n; ++i) idx[i] = (uint8_t) i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.begin() + n,
                      [&](uint8_t a, uint8_t b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    for (int i = 0; i < k; ++i) out[i] = idx[i];
}

// predice la selezione del layer `target` dallo stato h (n_embd f32): replica
// esatta di build_moe_ffn per questo modello: sigmoid(W·h) + exp_probs_b, top-k.
static void predict(const Looka & lk, int target, const float * h, uint8_t * out8) {
    const auto & W = lk.W[target];
    const auto & B = lk.bias[target];
    const int E = (int) B.size();
    float sel[64];
    for (int e = 0; e < E; ++e) {
        const float * w = W.data() + (size_t) e * lk.n_embd;
        float acc = 0.f;
        for (int i = 0; i < lk.n_embd; ++i) acc += w[i] * h[i];
        sel[e] = 1.f / (1.f + expf(-acc)) + B[e];
    }
    topk_from_scores(sel, E, 8, out8);
}

static int overlap4(const int32_t * truth, const uint8_t * pred, int k) {
    int n = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < k; ++j)
            if (truth[i] == (int32_t) pred[j]) { ++n; break; }
    return n;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (CbData *) user_data;
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_norm = strncmp(t->name, "ffn_norm-", 9) == 0;
    if (ask) return is_topk || is_norm;
    if (cb->fail) return true;

    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        cb->tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, cb->tmp.data(), 0, ggml_nbytes(t));
        data = cb->tmp.data();
    }

    if (is_norm && cb->lk) {
        const int il = atoi(t->name + 9);
        if (t->type != GGML_TYPE_F32 || (int) t->ne[0] != cb->lk->n_embd) {
            cb->fail = true; cb->fail_msg = std::string("ffn_norm inatteso: ") + t->name; return true;
        }
        const int64_t n_cols = t->ne[1];
        const bool has_self = il >= cb->dense_lead;                       // il e' MoE: autotest
        const bool has_next = il + 1 <= cb->n_layer - 1;                  // esiste il layer il+1
        for (int64_t j = 0; j < n_cols; ++j) {
            const float * h = (const float *)(data + j * t->nb[1]);
            if (has_self) {
                uint8_t p8[8];
                predict(*cb->lk, il, h, p8);
                auto & dst = cb->pred_self[il][j];
                for (int i = 0; i < 4; ++i) dst[i] = p8[i];
            }
            if (has_next) {
                predict(*cb->lk, il + 1, h, cb->pred_next[il + 1][j].data());
            }
        }
        return true;
    }

    if (!is_topk) return true;
    const int il = atoi(t->name + 13);
    if (t->type != GGML_TYPE_I32 || t->ne[0] < 4) {
        cb->fail = true; cb->fail_msg = std::string("tensore inatteso: ") + t->name; return true;
    }
    const int64_t n_cols = t->ne[1];
    const int slot = il - cb->dense_lead;
    const int ph = cb->phase == 'd' ? 1 : 0;
    for (int64_t j = 0; j < n_cols; ++j) {
        const int64_t pos = cb->base_pos + j;
        if (slot < 0 || slot >= cb->n_moe || pos >= (int64_t) cb->recs->size()) {
            cb->fail = true; cb->fail_msg = "slot/posizione fuori range su " + std::string(t->name); return true;
        }
        auto & dst = (*cb->recs)[pos].topk[slot];
        int32_t truth[4];
        for (int e = 0; e < 4; ++e) {
            const int32_t id = *(const int32_t *)(data + j * t->nb[1] + e * t->nb[0]);
            if (id < 0 || id >= 64) { cb->fail = true; cb->fail_msg = "expert id fuori [0,64) su " + std::string(t->name); return true; }
            dst[e] = id; truth[e] = id;
        }
        if (cb->lk) {
            auto & P = cb->lk->pl[il];
            if (cb->phase == 'd') { // dump predizioni per il replay di fase 5
                auto & pr = (*cb->recs)[pos].pred;
                if (pr.empty()) pr.assign((size_t) cb->n_moe, std::array<uint8_t,8>{});
                pr[slot] = cb->pred_next[il][j];
            }
            // autotest: predizione dello stesso layer (da ffn_norm-il, gia' osservato)
            P.selfHit += overlap4(truth, cb->pred_self[il][j].data(), 4);
            P.selfTot += 4;
            // lookahead: predizione fatta al layer il-1 (spec: predittore = layer MoE; il-1==0 e' extra)
            if (il - 1 >= cb->dense_lead) {
                const uint8_t * p = cb->pred_next[il][j].data();
                P.nextTot[ph] += 4;
                P.nextHit[ph][0] += overlap4(truth, p, 4);
                P.nextHit[ph][1] += overlap4(truth, p, 6);
                P.nextHit[ph][2] += overlap4(truth, p, 8);
            } else if (il - 1 >= 0) {
                P.extraTot += 4;
                P.extraHit8 += overlap4(truth, cb->pred_next[il][j].data(), 8);
            }
            // baseline decode: stessi expert del token precedente
            if (cb->phase == 'd') {
                if (cb->prev_valid[il]) {
                    int n = 0;
                    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) if (truth[a] == cb->prev_top4[il][b]) { ++n; break; }
                    P.prevOverlap += n; P.prevTot += 4;
                }
                for (int e = 0; e < 4; ++e) cb->prev_top4[il][e] = truth[e];
                cb->prev_valid[il] = true;
            }
        }
    }
    cb->cols_seen[il] += (int) n_cols;
    return true;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[trace] file mancante: %s\n", path.c_str()); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static std::string meta_str(const llama_model * model, const char * key) {
    char buf[256] = {0};
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return n >= 0 ? std::string(buf) : std::string("");
}

// legge i pesi router (f32) direttamente dal GGUF: blk.L.ffn_gate_inp.weight [n_embd, 64]
// e blk.L.exp_probs_b.bias [64], per L in [dense_lead, n_layer-1].
static bool load_router_weights(const std::string & gguf_path, Looka & lk) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &meta };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), ip);
    if (!g) { fprintf(stderr, "[looka] gguf_init fallita\n"); return false; }
    const size_t base = gguf_get_data_offset(g);
    std::ifstream f(gguf_path, std::ios::binary);
    lk.W.resize(lk.n_layer);
    lk.bias.resize(lk.n_layer);
    lk.pl.assign(lk.n_layer, {});
    char name[128];
    for (int il = lk.dense_lead; il < lk.n_layer; ++il) {
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", il);
        const int64_t tw = gguf_find_tensor(g, name);
        ggml_tensor * mw = ggml_get_tensor(meta, name);
        snprintf(name, sizeof(name), "blk.%d.exp_probs_b.bias", il);
        const int64_t tb = gguf_find_tensor(g, name);
        ggml_tensor * mb = ggml_get_tensor(meta, name);
        if (tw < 0 || tb < 0 || !mw || !mb) { fprintf(stderr, "[looka] tensori router mancanti a il=%d\n", il); return false; }
        if (mw->type != GGML_TYPE_F32 || mb->type != GGML_TYPE_F32) {
            fprintf(stderr, "[looka] tipo router inatteso a il=%d: %s/%s (atteso f32)\n", il, ggml_type_name(mw->type), ggml_type_name(mb->type));
            return false;
        }
        if ((int) mw->ne[0] != lk.n_embd || mw->ne[1] != mb->ne[0]) { fprintf(stderr, "[looka] shape router inattesa a il=%d\n", il); return false; }
        const int n_exp = (int) mb->ne[0];
        lk.W[il].resize((size_t) n_exp * lk.n_embd);
        lk.bias[il].resize(n_exp);
        f.seekg((std::streamoff)(base + gguf_get_tensor_offset(g, tw)));
        f.read((char *) lk.W[il].data(), (std::streamsize)(lk.W[il].size() * sizeof(float)));
        f.seekg((std::streamoff)(base + gguf_get_tensor_offset(g, tb)));
        f.read((char *) lk.bias[il].data(), (std::streamsize)(lk.bias[il].size() * sizeof(float)));
        if (!f) { fprintf(stderr, "[looka] read pesi fallita a il=%d\n", il); return false; }
    }
    gguf_free(g);
    ggml_free(meta);
    fprintf(stderr, "[looka] pesi router caricati: layer %d..%d, %d expert, n_embd %d\n",
            lk.dense_lead, lk.n_layer - 1, (int) lk.bias[lk.dense_lead].size(), lk.n_embd);
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, out_prefix = "trace", corpus_hash = "n/a", gguf_sha = "n/a", commit = "n/a";
    std::vector<std::string> prompts;
    int n_threads = 16, n_predict = 640, n_chunk = 512;
    bool do_looka = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * f) { if (i + 1 >= argc) { fprintf(stderr, "manca valore per %s\n", f); exit(1); } return std::string(argv[++i]); };
        if      (a == "--model")      model_path = need("--model");
        else if (a == "--prompt")     prompts.push_back(need("--prompt"));
        else if (a == "--out-prefix") out_prefix = need("--out-prefix");
        else if (a == "--threads")    n_threads = atoi(need("--threads").c_str());
        else if (a == "--n-predict")  n_predict = atoi(need("--n-predict").c_str());
        else if (a == "--corpus-hash") corpus_hash = need("--corpus-hash");
        else if (a == "--gguf-sha256") gguf_sha = need("--gguf-sha256");
        else if (a == "--commit")     commit = need("--commit");
        else if (a == "--no-looka")   do_looka = false;
        else { fprintf(stderr, "arg sconosciuto: %s\n", a.c_str()); exit(1); }
    }
    if (model_path.empty() || prompts.empty()) {
        fprintf(stderr, "uso: trace --model M --prompt file [--prompt file ...] [--threads N] [--n-predict N] [--no-looka]\n");
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "[trace] load modello fallita\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_layer = llama_model_n_layer(model);
    const int n_nextn = llama_model_n_layer_nextn(model);
    const std::string arch = meta_str(model, "general.architecture");
    const int dense_lead = atoi(meta_str(model, (arch + ".leading_dense_block_count").c_str()).c_str());
    const int n_embd = atoi(meta_str(model, (arch + ".embedding_length").c_str()).c_str());
    const int n_moe = n_layer - dense_lead;
    fprintf(stderr, "[trace] arch=%s n_layer=%d nextn=%d dense_lead=%d n_embd=%d -> n_moe=%d looka=%d\n",
            arch.c_str(), n_layer, n_nextn, dense_lead, n_embd, n_moe, do_looka ? 1 : 0);
    if (n_moe <= 0 || n_embd <= 0) { fprintf(stderr, "[trace] hparams incoerenti\n"); return 1; }

    Looka lk;
    lk.n_embd = n_embd; lk.n_layer = n_layer; lk.dense_lead = dense_lead;
    if (do_looka && !load_router_weights(model_path, lk)) return 1;

    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) { fprintf(stderr, "[trace] chat template assente nel GGUF\n"); return 1; }

    std::ofstream jl(out_prefix + ".jsonl");
    jl << "{\"kind\":\"moe-route-trace\",\"schemaVersion\":1,\"arch\":\"" << arch
       << "\",\"nLayer\":" << n_layer << ",\"denseLead\":" << dense_lead
       << ",\"nMoe\":" << n_moe << ",\"nExpert\":64,\"nExpertUsed\":4"
       << ",\"ggufSha256\":\"" << gguf_sha << "\",\"llamaCppCommit\":\"" << commit
       << "\",\"corpusHash\":\"" << corpus_hash << "\",\"threads\":" << n_threads
       << ",\"nPredict\":" << n_predict << ",\"greedy\":true}\n";

    int64_t tot_prefill = 0, tot_decode = 0;
    bool any_eos = false;
    std::ostringstream per_prompt;

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const std::string user = read_file(prompts[pi]);
        llama_chat_message msg = { "user", user.c_str() };
        std::vector<char> fmt(user.size() * 2 + 4096);
        int32_t flen = llama_chat_apply_template(tmpl, &msg, 1, true, fmt.data(), (int32_t) fmt.size());
        if (flen > (int32_t) fmt.size()) { fmt.resize(flen + 1); flen = llama_chat_apply_template(tmpl, &msg, 1, true, fmt.data(), (int32_t) fmt.size()); }
        if (flen < 0) { fprintf(stderr, "[trace] template fallito su %s\n", prompts[pi].c_str()); return 1; }
        std::string text(fmt.data(), flen);

        std::vector<llama_token> toks(text.size() + 16);
        int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, true);
        if (n_tok < 0) { toks.resize(-n_tok); n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, true); }
        toks.resize(n_tok);

        CbData cb;
        cb.dense_lead = dense_lead;
        cb.n_moe = n_moe;
        cb.n_layer = n_layer;
        cb.lk = do_looka ? &lk : nullptr;
        if (do_looka) {
            cb.pred_next.assign(n_layer, {});
            cb.pred_self.assign(n_layer, {});
            for (int il = 0; il < n_layer; ++il) { cb.pred_next[il].resize(n_chunk); cb.pred_self[il].resize(n_chunk); }
            cb.prev_top4.assign(n_layer, {{-1,-1,-1,-1}});
            cb.prev_valid.assign(n_layer, false);
        }
        std::vector<PosRec> recs;
        recs.reserve(n_tok + n_predict);
        cb.recs = &recs;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 8192;
        cparams.n_batch = (uint32_t) n_chunk;
        cparams.n_ubatch = (uint32_t) n_chunk;
        cparams.n_threads = cparams.n_threads_batch = n_threads;
        cparams.cb_eval = cb_eval;
        cparams.cb_eval_user_data = &cb;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fprintf(stderr, "[trace] init ctx fallita\n"); return 1; }

        // batch custom con logits=true su OGNI posizione: senza, llama.cpp pota
        // l'ultimo layer alle sole righe di output (inp_out_ids) e il routing
        // del layer finale sparisce dalla traccia per il resto del chunk.
        llama_batch batch = llama_batch_init(n_chunk, 0, 1);

        auto run_batch = [&](llama_token * p, int n, char phase) -> bool {
            cb.base_pos = (int64_t) recs.size();
            cb.expect_cols = n;
            cb.phase = phase;
            cb.cols_seen.clear();
            batch.n_tokens = n;
            for (int k = 0; k < n; ++k) {
                batch.token[k] = p[k];
                batch.pos[k] = (llama_pos) (recs.size() + (size_t) k);
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k] = true;
            }
            for (int k = 0; k < n; ++k) recs.push_back({ p[k], phase, std::vector<std::array<int32_t,4>>((size_t) n_moe, std::array<int32_t,4>{-1,-1,-1,-1}) });
            if (llama_decode(ctx, batch)) { cb.fail = true; cb.fail_msg = "llama_decode != 0"; return false; }
            if ((int) cb.cols_seen.size() != n_moe) { cb.fail = true; cb.fail_msg = "layer MoE osservati " + std::to_string(cb.cols_seen.size()) + " != " + std::to_string(n_moe); return false; }
            for (auto & kv : cb.cols_seen) if (kv.second != n) { cb.fail = true; cb.fail_msg = "colonne " + std::to_string(kv.second) + " != " + std::to_string(n) + " su il=" + std::to_string(kv.first); return false; }
            return !cb.fail;
        };

        for (int off = 0; off < n_tok; off += n_chunk) {
            const int n = std::min(n_chunk, n_tok - off);
            if (!run_batch(toks.data() + off, n, 'p')) { fprintf(stderr, "[trace] FAIL prefill p%zu: %s\n", pi, cb.fail_msg.c_str()); return 2; }
        }
        const int64_t n_prefill = (int64_t) recs.size();

        int n_dec = 0; bool eos = false;
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (; n_dec < n_predict; ++n_dec) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            llama_token best = 0; float bv = logits[0];
            for (int v = 1; v < n_vocab; ++v) if (logits[v] > bv) { bv = logits[v]; best = v; }
            if (llama_vocab_is_eog(vocab, best)) { eos = true; any_eos = true; break; }
            llama_token t = best;
            if (!run_batch(&t, 1, 'd')) { fprintf(stderr, "[trace] FAIL decode p%zu: %s\n", pi, cb.fail_msg.c_str()); return 2; }
        }

        for (size_t i = 0; i < recs.size(); ++i) {
            jl << "{\"p\":" << pi << ",\"i\":" << i << ",\"tok\":" << recs[i].tok
               << ",\"ph\":\"" << recs[i].phase << "\",\"e\":[";
            for (int s = 0; s < n_moe; ++s) {
                const auto & q = recs[i].topk[s];
                if (q[0] < 0) { fprintf(stderr, "[trace] FAIL: posizione %zu slot %d mai scritto (p%zu)\n", i, s, pi); return 2; }
                jl << q[0] << "," << q[1] << "," << q[2] << "," << q[3];
                jl << (s + 1 < n_moe ? "," : "");
            }
            jl << "]";
            if (!recs[i].pred.empty()) {
                jl << ",\"pr\":[";
                for (int s = 0; s < n_moe; ++s) {
                    const auto & q = recs[i].pred[s];
                    for (int e = 0; e < 8; ++e) jl << (int) q[e] << ((e < 7 || s + 1 < n_moe) ? "," : "");
                }
                jl << "]";
            }
            jl << "}\n";
        }
        tot_prefill += n_prefill;
        tot_decode  += n_dec;
        per_prompt << (pi ? "," : "") << "{\"file\":\"" << prompts[pi] << "\",\"prefill\":" << n_prefill
                   << ",\"decode\":" << n_dec << ",\"eosEarly\":" << (eos ? "true" : "false") << "}";
        fprintf(stderr, "[trace] p%zu %s: prefill=%lld decode=%d eos=%d\n", pi, prompts[pi].c_str(), (long long) n_prefill, n_dec, eos ? 1 : 0);
        llama_batch_free(batch);
        llama_free(ctx);
    }
    jl.close();

    const int64_t tot = tot_prefill + tot_decode;
    bool ok_positions = tot >= 16384 && tot_decode >= 4096;
    bool ok_autotest = true;

    if (do_looka) {
        // aggregati
        int64_t sh = 0, st = 0;
        int64_t nt[2] = {0,0}, nh[2][3] = {{0,0,0},{0,0,0}};
        int64_t bo = 0, bt = 0;
        double self_min = 2.0; int self_min_il = -1; // init >1: altrimenti nessun layer vince il confronto quando sono tutti a 1.0
        for (int il = dense_lead; il < n_layer; ++il) {
            const auto & P = lk.pl[il];
            sh += P.selfHit; st += P.selfTot;
            if (P.selfTot) { const double r = (double) P.selfHit / P.selfTot; if (r < self_min) { self_min = r; self_min_il = il; } }
            for (int ph = 0; ph < 2; ++ph) { nt[ph] += P.nextTot[ph]; for (int k = 0; k < 3; ++k) nh[ph][k] += P.nextHit[ph][k]; }
            bo += P.prevOverlap; bt += P.prevTot;
        }
        const double self_r = st ? (double) sh / st : 0.0;
        ok_autotest = self_r >= 0.999;
        // sanity la_tot: target = layer MoE con predecessore MoE = n_moe-1; unita' = 4 per posizione
        const int64_t la_expected = (int64_t)(n_moe - 1) * tot * 4;
        const bool ok_la = (nt[0] + nt[1]) == la_expected;
        const int64_t self_expected = (int64_t) n_moe * tot * 4;
        const bool ok_self_tot = st == self_expected;
        ok_positions = ok_positions && ok_la && ok_self_tot;

        std::ofstream rj(out_prefix + "-recall.json");
        rj << "{\"kind\":\"moe-looka-recall\",\"schemaVersion\":1,"
           << "\"ggufSha256\":\"" << gguf_sha << "\",\"llamaCppCommit\":\"" << commit << "\","
           << "\"corpusHash\":\"" << corpus_hash << "\",\"threads\":" << n_threads << ","
           << "\"predictor\":\"router(L+1) su ffn_norm(L), sigmoid+exp_probs_b, replica build_moe_ffn\","
           << "\"autotest\":{\"recall\":" << self_r << ",\"gate\":0.999,\"pass\":" << (ok_autotest ? "true" : "false")
           << ",\"minPerLayer\":" << self_min << ",\"minLayer\":" << self_min_il << ",\"tot\":" << st << ",\"expected\":" << self_expected << "},"
           << "\"lookahead\":{";
        const char * phn[2] = {"prefill", "decode"};
        for (int ph = 0; ph < 2; ++ph) {
            rj << "\"" << phn[ph] << "\":{\"tot\":" << nt[ph];
            const char * kn[3] = {"recallAt4", "recallAt6", "recallAt8"};
            for (int k = 0; k < 3; ++k) rj << ",\"" << kn[k] << "\":" << (nt[ph] ? (double) nh[ph][k] / nt[ph] : 0.0);
            rj << "}" << (ph == 0 ? "," : "");
        }
        rj << ",\"laTotExpected\":" << la_expected << ",\"laTotPass\":" << (ok_la ? "true" : "false") << "},"
           << "\"baselinePrev\":{\"decodeTransitions\":" << bt / 4 << ",\"overlap\":" << (bt ? (double) bo / bt : 0.0) << "},"
           << "\"perLayer\":[";
        for (int il = dense_lead; il < n_layer; ++il) {
            const auto & P = lk.pl[il];
            const int64_t dt = P.nextTot[1];
            rj << (il > dense_lead ? "," : "") << "{\"il\":" << il
               << ",\"self\":" << (P.selfTot ? (double) P.selfHit / P.selfTot : -1)
               << ",\"decodeR4\":" << (dt ? (double) P.nextHit[1][0] / dt : -1)
               << ",\"decodeR6\":" << (dt ? (double) P.nextHit[1][1] / dt : -1)
               << ",\"decodeR8\":" << (dt ? (double) P.nextHit[1][2] / dt : -1)
               << ",\"prevOverlap\":" << (P.prevTot ? (double) P.prevOverlap / P.prevTot : -1) << "}";
        }
        // extra fuori-spec: primo layer MoE predetto dallo stato del layer DENSO
        const auto & P1 = lk.pl[dense_lead];
        rj << "],\"extraDenseToFirstMoe\":{\"tot\":" << P1.extraTot << ",\"recallAt8\":"
           << (P1.extraTot ? (double) P1.extraHit8 / P1.extraTot : -1) << "}}\n";
        rj.close();
        fprintf(stderr, "[looka] autotest self-recall %.6f (gate 0.999, min layer %d: %.6f) %s\n",
                self_r, self_min_il, self_min, ok_autotest ? "PASS" : "FAIL");
        fprintf(stderr, "[looka] lookahead decode: R@4 %.4f R@6 %.4f R@8 %.4f (tot %lld) — baseline prev %.4f\n",
                nt[1] ? (double) nh[1][0] / nt[1] : 0, nt[1] ? (double) nh[1][1] / nt[1] : 0,
                nt[1] ? (double) nh[1][2] / nt[1] : 0, (long long) nt[1], bt ? (double) bo / bt : 0);
    }

    std::ofstream sum(out_prefix + "-summary.json");
    sum << "{\"kind\":\"moe-route-trace-summary\",\"schemaVersion\":1,"
        << "\"ggufSha256\":\"" << gguf_sha << "\",\"llamaCppCommit\":\"" << commit << "\","
        << "\"corpusHash\":\"" << corpus_hash << "\",\"threads\":" << n_threads << ","
        << "\"nMoe\":" << n_moe << ",\"prompts\":[" << per_prompt.str() << "],"
        << "\"totPrefill\":" << tot_prefill << ",\"totDecode\":" << tot_decode << ",\"tot\":" << tot << ","
        << "\"rowsExpected\":" << tot * n_moe << ",\"anyEosEarly\":" << (any_eos ? "true" : "false") << ","
        << "\"sanity\":{\"colsPerLayerPerBatch\":\"PASS\",\"allSlotsWritten\":\"PASS\","
        << "\"gatePositions\":" << (ok_positions ? "\"PASS\"" : "\"FAIL\"")
        << ",\"autotest\":" << (do_looka ? (ok_autotest ? "\"PASS\"" : "\"FAIL\"") : "\"SKIPPED\"") << "}}\n";
    sum.close();
    fprintf(stderr, "[trace] TOTALE prefill=%lld decode=%lld tot=%lld (gate: %s, autotest: %s)\n",
            (long long) tot_prefill, (long long) tot_decode, (long long) tot,
            ok_positions ? "PASS" : "FAIL", do_looka ? (ok_autotest ? "PASS" : "FAIL") : "skipped");
    llama_model_free(model);
    llama_backend_free();
    if (!ok_autotest) return 4;
    return ok_positions ? 0 : 3;
}
