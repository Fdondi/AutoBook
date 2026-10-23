// llama.cpp for the reader (see llm/LlamaNative.kt): load GGUF models, chat with
// streaming and optional JSON-schema constrained output, and pooled embeddings.
// Text crosses JNI as UTF-8 byte arrays: JNI's "modified UTF-8" mangles characters
// outside the BMP, and a token can end mid-character.

#include <jni.h>
#include <android/log.h>
#include <unistd.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama.h"

#define TAG "autoreader-llama"

namespace {

struct Chat {
    llama_model *model;
    llama_context *ctx;
    const llama_vocab *vocab;
    std::string tmpl;
    std::vector<llama_token> cached;   // tokens in the KV cache: a repeated prompt prefix is not decoded again
};

struct Embedder {
    llama_model *model;
    llama_context *ctx;
    const llama_vocab *vocab;
};

void log_cb(ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", text);
    else if (level == GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, "%s", text);
}

std::string bytes(JNIEnv *env, jbyteArray a) {
    if (a == nullptr) return {};
    jsize n = env->GetArrayLength(a);
    std::string s(n, '\0');
    env->GetByteArrayRegion(a, 0, n, reinterpret_cast<jbyte *>(s.data()));
    return s;
}

jbyteArray to_bytes(JNIEnv *env, const std::string &s) {
    jbyteArray a = env->NewByteArray((jsize) s.size());
    env->SetByteArrayRegion(a, 0, (jsize) s.size(), reinterpret_cast<const jbyte *>(s.data()));
    return a;
}

void throw_java(JNIEnv *env, const std::string &msg) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), msg.c_str());
}

std::string apply_template(const Chat &c, const std::vector<llama_chat_message> &msgs) {
    const char *tmpl = c.tmpl.empty() ? "chatml" : c.tmpl.c_str();
    std::vector<char> buf(4096);
    for (int attempt = 0; attempt < 2; attempt++) {
        int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), (int32_t) buf.size());
        if (n < 0) throw std::runtime_error("the model's chat template is not supported");
        if (n <= (int) buf.size()) return {buf.data(), (size_t) n};
        buf.resize(n);
    }
    throw std::runtime_error("chat template output keeps growing");
}

llama_sampler *make_sampler(const llama_vocab *vocab, const std::string &grammar, float temperature) {
    llama_sampler *s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!grammar.empty()) llama_sampler_chain_add(s, llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(s, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(s, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(s, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(s, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(s, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    return s;
}

// Decode `tokens` after the longest prefix already in the KV cache.
void prefill(Chat &c, const std::vector<llama_token> &tokens) {
    size_t keep = 0;
    while (keep < c.cached.size() && keep < tokens.size() && c.cached[keep] == tokens[keep]) keep++;
    if (keep == tokens.size()) keep--;                       // the last prompt token must be decoded for logits
    llama_memory_t mem = llama_get_memory(c.ctx);
    if (!llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1)) {
        llama_memory_clear(mem, true);
        keep = 0;
    }
    c.cached.assign(tokens.begin(), tokens.begin() + (long) keep);
    const size_t n_batch = llama_n_batch(c.ctx);
    for (size_t i = keep; i < tokens.size(); i += n_batch) {
        size_t n = std::min(n_batch, tokens.size() - i);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + i), (int32_t) n);
        if (llama_decode(c.ctx, batch) != 0) {
            llama_memory_clear(mem, true);
            c.cached.clear();
            throw std::runtime_error("prompt decode failed");
        }
        c.cached.insert(c.cached.end(), tokens.begin() + (long) i, tokens.begin() + (long) (i + n));
    }
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_com_example_autoreader_llm_LlamaNative_init(JNIEnv *, jclass) {
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
}

JNIEXPORT jlong JNICALL
Java_com_example_autoreader_llm_LlamaNative_loadModel(JNIEnv *env, jclass, jstring jpath) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(path, params);
    env->ReleaseStringUTFChars(jpath, path);
    if (model == nullptr) throw_java(env, "couldn't load the model file");
    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL
Java_com_example_autoreader_llm_LlamaNative_freeModel(JNIEnv *, jclass, jlong model) {
    llama_model_free(reinterpret_cast<llama_model *>(model));
}

JNIEXPORT jstring JNICALL
Java_com_example_autoreader_llm_LlamaNative_metadata(JNIEnv *env, jclass, jlong model, jstring jkey) {
    const char *key = env->GetStringUTFChars(jkey, nullptr);
    char buf[256];
    int n = llama_model_meta_val_str(reinterpret_cast<llama_model *>(model), key, buf, sizeof buf);
    env->ReleaseStringUTFChars(jkey, key);
    return n < 0 ? nullptr : env->NewStringUTF(buf);
}

JNIEXPORT jlong JNICALL
Java_com_example_autoreader_llm_LlamaNative_newChat(JNIEnv *env, jclass, jlong jmodel, jint n_ctx, jint n_threads) {
    auto *model = reinterpret_cast<llama_model *>(jmodel);
    llama_context_params params = llama_context_default_params();
    params.n_ctx = (uint32_t) n_ctx;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, params);
    if (ctx == nullptr) {
        throw_java(env, "couldn't create a context for the model");
        return 0;
    }
    const char *tmpl = llama_model_chat_template(model, nullptr);
    auto *c = new Chat{model, ctx, llama_model_get_vocab(model), tmpl ? tmpl : "", {}};
    return reinterpret_cast<jlong>(c);
}

JNIEXPORT void JNICALL
Java_com_example_autoreader_llm_LlamaNative_freeChat(JNIEnv *, jclass, jlong jchat) {
    auto *c = reinterpret_cast<Chat *>(jchat);
    llama_free(c->ctx);
    delete c;
}

// Generate a reply to `roles`/`contents`. With `schema` (JSON Schema, UTF-8) the output is
// constrained to match it. `callback.onToken(byte[])` receives each piece and returns false
// to stop early. Returns the whole reply.
JNIEXPORT jbyteArray JNICALL
Java_com_example_autoreader_llm_LlamaNative_chat(JNIEnv *env, jclass, jlong jchat, jobjectArray jroles,
                                                 jobjectArray jcontents, jbyteArray jschema, jfloat temperature,
                                                 jint max_tokens, jobject callback) {
    Chat &c = *reinterpret_cast<Chat *>(jchat);
    llama_sampler *smpl = nullptr;
    try {
        const jsize n = env->GetArrayLength(jroles);
        std::vector<std::string> roles(n), contents(n);
        std::vector<llama_chat_message> msgs(n);
        for (jsize i = 0; i < n; i++) {
            auto role = (jstring) env->GetObjectArrayElement(jroles, i);
            const char *r = env->GetStringUTFChars(role, nullptr);
            roles[i] = r;
            env->ReleaseStringUTFChars(role, r);
            contents[i] = bytes(env, (jbyteArray) env->GetObjectArrayElement(jcontents, i));
        }
        for (jsize i = 0; i < n; i++) msgs[i] = {roles[i].c_str(), contents[i].c_str()};

        const std::string prompt = apply_template(c, msgs);
        std::vector<llama_token> tokens = common_tokenize(c.vocab, prompt, true, true);
        if (tokens.size() + max_tokens > llama_n_ctx(c.ctx)) {
            throw std::runtime_error("prompt too long: " + std::to_string(tokens.size()) + " tokens");
        }
        const std::string schema = bytes(env, jschema);
        const std::string grammar = schema.empty() ? "" : json_schema_to_grammar(common_json::parse(schema));
        smpl = make_sampler(c.vocab, grammar, temperature);
        prefill(c, tokens);

        jmethodID on_token = nullptr;
        if (callback != nullptr) on_token = env->GetMethodID(env->GetObjectClass(callback), "onToken", "([B)Z");
        std::string out;
        for (int i = 0; i < max_tokens; i++) {
            llama_token tok = llama_sampler_sample(smpl, c.ctx, -1);
            if (llama_vocab_is_eog(c.vocab, tok)) break;
            const std::string piece = common_token_to_piece(c.vocab, tok, false);
            out += piece;
            if (on_token != nullptr) {
                jbyteArray p = to_bytes(env, piece);
                jboolean go_on = env->CallBooleanMethod(callback, on_token, p);
                env->DeleteLocalRef(p);
                if (env->ExceptionCheck() || !go_on) break;
            }
            llama_batch batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(c.ctx, batch) != 0) throw std::runtime_error("decode failed");
            c.cached.push_back(tok);
        }
        llama_sampler_free(smpl);
        if (env->ExceptionCheck()) return nullptr;
        return to_bytes(env, out);
    } catch (const std::exception &e) {
        if (smpl != nullptr) llama_sampler_free(smpl);
        throw_java(env, e.what());
        return nullptr;
    }
}

JNIEXPORT jlong JNICALL
Java_com_example_autoreader_llm_LlamaNative_newEmbedder(JNIEnv *env, jclass, jlong jmodel, jint n_threads) {
    auto *model = reinterpret_cast<llama_model *>(jmodel);
    llama_context_params params = llama_context_default_params();
    // Pooled embeddings need the whole input in one ubatch.
    const uint32_t n_ctx = (uint32_t) std::min(2048, llama_model_n_ctx_train(model));
    params.n_ctx = n_ctx;
    params.n_batch = n_ctx;
    params.n_ubatch = n_ctx;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.embeddings = true;
    llama_context *ctx = llama_init_from_model(model, params);
    if (ctx == nullptr) {
        throw_java(env, "couldn't create an embedding context");
        return 0;
    }
    return reinterpret_cast<jlong>(new Embedder{model, ctx, llama_model_get_vocab(model)});
}

JNIEXPORT void JNICALL
Java_com_example_autoreader_llm_LlamaNative_freeEmbedder(JNIEnv *, jclass, jlong jemb) {
    auto *e = reinterpret_cast<Embedder *>(jemb);
    llama_free(e->ctx);
    delete e;
}

// Unit-length pooled embedding of `text` (UTF-8).
JNIEXPORT jfloatArray JNICALL
Java_com_example_autoreader_llm_LlamaNative_embed(JNIEnv *env, jclass, jlong jemb, jbyteArray jtext) {
    Embedder &e = *reinterpret_cast<Embedder *>(jemb);
    std::vector<llama_token> tokens = common_tokenize(e.vocab, bytes(env, jtext), true, false);
    tokens.resize(std::min(tokens.size(), (size_t) llama_n_batch(e.ctx)));
    if (tokens.empty()) {
        throw_java(env, "nothing to embed");
        return nullptr;
    }
    llama_memory_clear(llama_get_memory(e.ctx), true);
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = (int32_t) tokens.size();
    const bool encoder_only = llama_model_has_encoder(e.model) && !llama_model_has_decoder(e.model);
    const int rc = encoder_only ? llama_encode(e.ctx, batch) : llama_decode(e.ctx, batch);
    const float *embd = rc == 0 ? llama_get_embeddings_seq(e.ctx, 0) : nullptr;
    if (embd == nullptr) {
        llama_batch_free(batch);
        throw_java(env, "embedding failed");
        return nullptr;
    }
    const int dim = llama_model_n_embd_out(e.model);
    std::vector<float> out(dim);
    common_embd_normalize(embd, out.data(), dim, 2);
    llama_batch_free(batch);
    jfloatArray result = env->NewFloatArray(dim);
    env->SetFloatArrayRegion(result, 0, dim, out.data());
    return result;
}

}  // extern "C"
