// oracle-moe/golden.cpp — golden logits GLM dall'oracolo llama.cpp (goal
// engine-fase-c2, fase 1). Pattern di scripts/gen-golden.py (fase A) portato
// sull'oracolo C1 a commit pinnato: per ogni prompt del corpus, prefill +
// decode greedy, e per ogni posizione generata dump di argmax + top-K
// (id, logit f32). Il motore consuma token id: i promptTokens stanno nel
// payload. Zero patch upstream, CPU-only, deterministico (greedy, thread
// registrati, SHA-256 del GGUF nel payload).
//
// Build: tools/oracle-moe/build-golden.sh   Run: tools/oracle-moe/run-golden.sh
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[golden] file mancante: %s\n", path.c_str()); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char ** argv) {
    std::string model_path, out_path = "golden.json", corpus_hash = "n/a", gguf_sha = "n/a", commit = "n/a";
    std::vector<std::string> prompts;
    int n_threads = 16, n_predict = 128, top_k = 32, n_chunk = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * f) { if (i + 1 >= argc) { fprintf(stderr, "manca valore per %s\n", f); exit(1); } return std::string(argv[++i]); };
        if      (a == "--model")       model_path = need("--model");
        else if (a == "--prompt")      prompts.push_back(need("--prompt"));
        else if (a == "--out")         out_path = need("--out");
        else if (a == "--threads")     n_threads = atoi(need("--threads").c_str());
        else if (a == "--n-predict")   n_predict = atoi(need("--n-predict").c_str());
        else if (a == "--top-k")       top_k = atoi(need("--top-k").c_str());
        else if (a == "--corpus-hash") corpus_hash = need("--corpus-hash");
        else if (a == "--gguf-sha256") gguf_sha = need("--gguf-sha256");
        else if (a == "--commit")      commit = need("--commit");
        else { fprintf(stderr, "arg sconosciuto: %s\n", a.c_str()); exit(1); }
    }
    if (model_path.empty() || prompts.empty()) {
        fprintf(stderr, "uso: golden --model M --prompt file [--prompt file ...] [--threads N] [--n-predict N] [--top-k K] --out out.json\n");
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "[golden] load modello fallita\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) { fprintf(stderr, "[golden] chat template assente nel GGUF\n"); return 1; }

    std::ostringstream pj;
    int64_t tot_prefill = 0, tot_decode = 0;
    std::vector<int> top_idx(n_vocab);

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const std::string user = read_file(prompts[pi]);
        llama_chat_message msg = { "user", user.c_str() };
        std::vector<char> fmt(user.size() * 2 + 4096);
        int32_t flen = llama_chat_apply_template(tmpl, &msg, 1, true, fmt.data(), (int32_t) fmt.size());
        if (flen > (int32_t) fmt.size()) { fmt.resize(flen + 1); flen = llama_chat_apply_template(tmpl, &msg, 1, true, fmt.data(), (int32_t) fmt.size()); }
        if (flen < 0) { fprintf(stderr, "[golden] template fallito su %s\n", prompts[pi].c_str()); return 1; }
        std::string text(fmt.data(), flen);

        std::vector<llama_token> toks(text.size() + 16);
        int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, true);
        if (n_tok < 0) { toks.resize(-n_tok); n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, true); }
        toks.resize(n_tok);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 8192;
        cparams.n_batch = (uint32_t) n_chunk;
        cparams.n_ubatch = (uint32_t) n_chunk;
        cparams.n_threads = cparams.n_threads_batch = n_threads;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fprintf(stderr, "[golden] init ctx fallita\n"); return 1; }
        llama_batch batch = llama_batch_init(n_chunk, 0, 1);

        int64_t pos = 0;
        auto run_batch = [&](llama_token * p, int n, bool want_logits_last) -> bool {
            batch.n_tokens = n;
            for (int k = 0; k < n; ++k) {
                batch.token[k] = p[k];
                batch.pos[k] = (llama_pos)(pos + k);
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k] = want_logits_last && k == n - 1;
            }
            pos += n;
            return llama_decode(ctx, batch) == 0;
        };

        for (int off = 0; off < n_tok; off += n_chunk) {
            const int n = std::min(n_chunk, n_tok - off);
            const bool last_chunk = off + n >= n_tok;
            if (!run_batch(toks.data() + off, n, last_chunk)) { fprintf(stderr, "[golden] FAIL prefill p%zu\n", pi); return 2; }
        }

        std::ostringstream posj;
        std::vector<llama_token> generated;
        int eos_at = -1;
        for (int d = 0; d < n_predict; ++d) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) { fprintf(stderr, "[golden] logits nulli p%zu d=%d\n", pi, d); return 2; }
            std::iota(top_idx.begin(), top_idx.end(), 0);
            std::partial_sort(top_idx.begin(), top_idx.begin() + top_k, top_idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
            const llama_token best = top_idx[0];
            posj << (d ? "," : "") << "{\"argmax\":" << best << ",\"top\":[";
            for (int k = 0; k < top_k; ++k) {
                char lb[32];
                snprintf(lb, sizeof(lb), "%.4f", logits[top_idx[k]]);
                posj << "[" << top_idx[k] << "," << lb << "]" << (k + 1 < top_k ? "," : "");
            }
            posj << "]}";
            if (llama_vocab_is_eog(vocab, best)) { eos_at = d; break; } // la posizione EOG resta nel golden, non si esegue
            generated.push_back(best);
            llama_token t = best;
            if (!run_batch(&t, 1, true)) { fprintf(stderr, "[golden] FAIL decode p%zu d=%d\n", pi, d); return 2; }
        }

        pj << (pi ? "," : "") << "{\"id\":\"p" << pi << "\",\"file\":\"" << prompts[pi] << "\",\"promptTokens\":[";
        for (int k = 0; k < n_tok; ++k) pj << toks[k] << (k + 1 < n_tok ? "," : "");
        pj << "],\"generated\":[";
        for (size_t k = 0; k < generated.size(); ++k) pj << generated[k] << (k + 1 < generated.size() ? "," : "");
        pj << "],\"eosAt\":" << eos_at << ",\"positions\":[" << posj.str() << "]}";
        tot_prefill += n_tok;
        tot_decode += (int64_t) generated.size() + (eos_at >= 0 ? 1 : 0);
        fprintf(stderr, "[golden] p%zu %s: prefill=%d posizioni=%zu eosAt=%d\n",
                pi, prompts[pi].c_str(), n_tok, generated.size() + (eos_at >= 0 ? 1 : 0), eos_at);
        llama_batch_free(batch);
        llama_free(ctx);
    }

    std::ofstream out(out_path);
    out << "{\"schemaVersion\":1,\"kind\":\"engine-golden\",\"model\":\"" << model_path.substr(model_path.find_last_of('/') + 1)
        << "\",\"modelSha256\":\"" << gguf_sha << "\",\"arch\":\"deepseek2\","
        << "\"oracle\":{\"impl\":\"llama.cpp-oracle\",\"commit\":\"" << commit << "\",\"nThreads\":" << n_threads
        << ",\"nCtx\":8192,\"nBatch\":" << n_chunk << ",\"backend\":\"CPU\",\"sampling\":\"greedy\"},"
        << "\"corpusHash\":\"" << corpus_hash << "\",\"genTokens\":" << n_predict << ",\"topK\":" << top_k << ","
        << "\"totPrefill\":" << tot_prefill << ",\"totDecodePositions\":" << tot_decode << ","
        << "\"prompts\":[" << pj.str() << "]}\n";
    out.close();
    fprintf(stderr, "[golden] TOTALE prefill=%lld posizioni-golden=%lld -> %s\n",
            (long long) tot_prefill, (long long) tot_decode, out_path.c_str());
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
