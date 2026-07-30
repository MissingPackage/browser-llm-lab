// oracle-moe/trace.cpp — ROUTE_TRACE sull'oracolo llama.cpp (spec engine-fase-c1,
// §Strumentazione). Osserva ffn_moe_topk-<il> via cb_eval (zero patch upstream) e
// scrive la traccia di routing (posizione, layer, top-4) + summary con sanity-gate.
// Fase 4 (LOOKA online) estenderà questo stesso tool con ffn_norm-<il>.
//
// Build: tools/oracle-moe/build.sh (linka la build CPU-only di ~/Projects/llama.cpp-oracle)
// Run:   tools/oracle-moe/run-trace.sh
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>

struct PosRec {
    int32_t tok;
    char phase; // 'p' prefill, 'd' decode
    std::vector<std::array<int32_t,4>> topk; // indicizzato per slot moe (il - dense_lead)
};

struct CbData {
    int dense_lead = 0;
    int n_moe = 0;
    int64_t base_pos = 0;      // posizione assoluta della colonna 0 dell'ubatch corrente
    int expect_cols = 0;       // n_tokens del decode corrente
    char phase = 'p';
    std::vector<PosRec> * recs = nullptr; // recs del prompt corrente (indice = posizione)
    std::map<int,int> cols_seen;          // il -> colonne osservate nel decode corrente
    std::vector<uint8_t> tmp;
    bool fail = false;
    std::string fail_msg;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (CbData *) user_data;
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) return is_topk;
    if (!is_topk || cb->fail) return true;

    const int il = atoi(t->name + 13);
    if (t->type != GGML_TYPE_I32 || t->ne[0] < 4) {
        cb->fail = true;
        cb->fail_msg = std::string("tensore inatteso: ") + t->name;
        return true;
    }
    const int64_t n_cols = t->ne[1];
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        cb->tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, cb->tmp.data(), 0, ggml_nbytes(t));
        data = cb->tmp.data();
    }
    const int slot = il - cb->dense_lead;
    for (int64_t j = 0; j < n_cols; ++j) {
        const int64_t pos = cb->base_pos + j;
        if (slot < 0 || slot >= cb->n_moe || pos >= (int64_t) cb->recs->size()) {
            cb->fail = true;
            cb->fail_msg = "slot/posizione fuori range su " + std::string(t->name);
            return true;
        }
        auto & dst = (*cb->recs)[pos].topk[slot];
        for (int e = 0; e < 4; ++e) {
            const int32_t id = *(const int32_t *)(data + j * t->nb[1] + e * t->nb[0]);
            if (id < 0 || id >= 64) {
                cb->fail = true;
                cb->fail_msg = "expert id fuori [0,64) su " + std::string(t->name);
                return true;
            }
            dst[e] = id;
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

int main(int argc, char ** argv) {
    std::string model_path, out_prefix = "trace", corpus_hash = "n/a", gguf_sha = "n/a", commit = "n/a";
    std::vector<std::string> prompts;
    int n_threads = 16, n_predict = 640, n_chunk = 512;
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
        else { fprintf(stderr, "arg sconosciuto: %s\n", a.c_str()); exit(1); }
    }
    if (model_path.empty() || prompts.empty()) {
        fprintf(stderr, "uso: trace --model M --prompt file [--prompt file ...] [--threads N] [--n-predict N]\n");
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
    const int n_moe = n_layer - dense_lead; // layer routed attesi nel forward (NextN escluso dal forward)
    fprintf(stderr, "[trace] arch=%s n_layer=%d nextn=%d dense_lead=%d -> n_moe=%d\n",
            arch.c_str(), n_layer, n_nextn, dense_lead, n_moe);
    if (n_moe <= 0) { fprintf(stderr, "[trace] hparams incoerenti\n"); return 1; }

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
            jl << "]}\n";
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
    const bool ok_positions = tot >= 16384 && tot_decode >= 4096;
    std::ofstream sum(out_prefix + "-summary.json");
    sum << "{\"kind\":\"moe-route-trace-summary\",\"schemaVersion\":1,"
        << "\"ggufSha256\":\"" << gguf_sha << "\",\"llamaCppCommit\":\"" << commit << "\","
        << "\"corpusHash\":\"" << corpus_hash << "\",\"threads\":" << n_threads << ","
        << "\"nMoe\":" << n_moe << ",\"prompts\":[" << per_prompt.str() << "],"
        << "\"totPrefill\":" << tot_prefill << ",\"totDecode\":" << tot_decode << ",\"tot\":" << tot << ","
        << "\"rowsExpected\":" << tot * n_moe << ",\"anyEosEarly\":" << (any_eos ? "true" : "false") << ","
        << "\"sanity\":{\"colsPerLayerPerBatch\":\"PASS\",\"allSlotsWritten\":\"PASS\","
        << "\"gatePositions\":" << (ok_positions ? "\"PASS\"" : "\"FAIL\"") << "}}\n";
    sum.close();
    fprintf(stderr, "[trace] TOTALE prefill=%lld decode=%lld tot=%lld (gate >=16384 e decode >=4096: %s)\n",
            (long long) tot_prefill, (long long) tot_decode, (long long) tot, ok_positions ? "PASS" : "FAIL");
    llama_model_free(model);
    llama_backend_free();
    return ok_positions ? 0 : 3;
}
