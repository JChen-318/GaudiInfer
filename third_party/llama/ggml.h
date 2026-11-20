// Minimal stand-in for ggml.h: the vendored llama.cpp sources (unicode, json, jinja) only need GGML_ASSERT.
#pragma once

#include <cstdio>
#include <cstdlib>

#define GGML_ASSERT(x)                                                                                  \
    do {                                                                                                \
        if (!(x)) {                                                                                     \
            std::fprintf(stderr, "%s:%d: GGML_ASSERT(%s) failed\n", __FILE__, __LINE__, #x);            \
            std::abort();                                                                               \
        }                                                                                               \
    } while (0)
