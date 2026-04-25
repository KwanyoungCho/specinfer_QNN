#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

struct common_sampler_profile_snapshot {
    int64_t sample_calls = 0;
    int64_t sample_total_us = 0;
    int64_t sync_us = 0;
    int64_t set_logits_us = 0;
    int64_t get_logits_us = 0;
    int64_t build_candidates_us = 0;
    int64_t grammar_apply_us = 0;
    int64_t chain_apply_us = 0;
    int64_t grammar_check_us = 0;
    int64_t resample_count = 0;
    int64_t resample_set_logits_us = 0;
    int64_t resample_get_logits_us = 0;
    int64_t resample_build_candidates_us = 0;
    int64_t resample_grammar_apply_us = 0;
    int64_t resample_chain_apply_us = 0;
};

inline common_sampler_profile_snapshot common_sampler_profile_diff(
        const common_sampler_profile_snapshot & end,
        const common_sampler_profile_snapshot & begin) {
    common_sampler_profile_snapshot diff;
    diff.sample_calls                 = end.sample_calls                 - begin.sample_calls;
    diff.sample_total_us              = end.sample_total_us              - begin.sample_total_us;
    diff.sync_us                      = end.sync_us                      - begin.sync_us;
    diff.set_logits_us                = end.set_logits_us                - begin.set_logits_us;
    diff.get_logits_us                = end.get_logits_us                - begin.get_logits_us;
    diff.build_candidates_us          = end.build_candidates_us          - begin.build_candidates_us;
    diff.grammar_apply_us             = end.grammar_apply_us             - begin.grammar_apply_us;
    diff.chain_apply_us               = end.chain_apply_us               - begin.chain_apply_us;
    diff.grammar_check_us             = end.grammar_check_us             - begin.grammar_check_us;
    diff.resample_count               = end.resample_count               - begin.resample_count;
    diff.resample_set_logits_us       = end.resample_set_logits_us       - begin.resample_set_logits_us;
    diff.resample_get_logits_us       = end.resample_get_logits_us       - begin.resample_get_logits_us;
    diff.resample_build_candidates_us = end.resample_build_candidates_us - begin.resample_build_candidates_us;
    diff.resample_grammar_apply_us    = end.resample_grammar_apply_us    - begin.resample_grammar_apply_us;
    diff.resample_chain_apply_us      = end.resample_chain_apply_us      - begin.resample_chain_apply_us;
    return diff;
}

inline void common_sampler_profile_accumulate(
        common_sampler_profile_snapshot & dst,
        const common_sampler_profile_snapshot & src) {
    dst.sample_calls                 += src.sample_calls;
    dst.sample_total_us              += src.sample_total_us;
    dst.sync_us                      += src.sync_us;
    dst.set_logits_us                += src.set_logits_us;
    dst.get_logits_us                += src.get_logits_us;
    dst.build_candidates_us          += src.build_candidates_us;
    dst.grammar_apply_us             += src.grammar_apply_us;
    dst.chain_apply_us               += src.chain_apply_us;
    dst.grammar_check_us             += src.grammar_check_us;
    dst.resample_count               += src.resample_count;
    dst.resample_set_logits_us       += src.resample_set_logits_us;
    dst.resample_get_logits_us       += src.resample_get_logits_us;
    dst.resample_build_candidates_us += src.resample_build_candidates_us;
    dst.resample_grammar_apply_us    += src.resample_grammar_apply_us;
    dst.resample_chain_apply_us      += src.resample_chain_apply_us;
}

void common_sampler_profile_reset();
common_sampler_profile_snapshot common_sampler_profile_get();

// llama_sampler API overloads

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if accept_grammar is true, the token is accepted both by the sampling chain and the grammar
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// apply the configured sampler chain to an externally supplied candidate set
// candidates should contain token ids in the model token-id space and raw logits
// when grammar_first is true, grammar constraints are applied before the sampler chain
// if do_sort == true, the returned candidates are sorted by probability in descending order
llama_token_data_array * common_sampler_apply_candidates(
        struct common_sampler * gsmpl,
        const llama_token_data * candidates,
        size_t size,
        bool grammar_first = false,
        bool do_sort = false);

// apply the configured sampler chain to token/logit pairs without forcing the
// caller to materialize an intermediate llama_token_data array first
llama_token_data_array * common_sampler_apply_logits(
        struct common_sampler * gsmpl,
        const llama_token * token_ids,
        const float * logits,
        size_t size,
        bool grammar_first = false,
        bool do_sort = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);
