// Direct Gaudi2 MME + TPC + SRAM runtime via Synapse's SCAL instance.
// Adapted from verified patterns in gaudi2-mme-programming/direct/attention (MIT).
#include "g2e/direct_mme.hpp"

#include <hlthunk.h>
#include <synapse_api.h>
#include <synapse_types.h>
#include "gaudi2/gaudi2.h"
#include "gaudi2/gaudi2_packets.h"

#include "gaudi2_direct/hcl/gaudi2_arc_sched_packets.h"
#include "gaudi2_direct/hcl/scal.h"
#include "gaudi2_direct/synapseai-core/gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi2_direct/synapseai-core/gaudi2/gaudi2_tpc_descriptor.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>

namespace g2e {

namespace {

constexpr uint32_t kRateLimiter = 0x040404u;
constexpr uint32_t kOwnRing = 65536u;
constexpr uint32_t kOwnBudget = kOwnRing - 512u;
constexpr uint64_t kMme0FenceAddr = 0x1000007ffc0ca2dcull;
constexpr uint64_t kMme2FenceAddr = 0x1000007ffc4ca2dcull;

// CP_FENCE<id>_RDATA of the lower CP (index 4) of an engine's QMAN: fence 3 at 0x2dc, 0x14 per fence id below it
uint64_t fence_reg(uint64_t fence3_addr, uint32_t fence_id) { return fence3_addr - uint64_t(3 - fence_id) * 0x14ull; }

uint64_t tpc_fence_addr(uint32_t index, uint32_t fence_id = 3) {
    return fence_reg(0x1000007ffc00a2dcull + uint64_t(index / 6) * 0x200000ull + uint64_t(index % 6) * 0x10000ull, fence_id);
}

uint32_t tpc_hw_dtype(DType dt) {
    switch (dt) {
        case DType::I8:   return 0;
        case DType::I16:  return 1;
        case DType::I32:  return 2;
        case DType::U8:   return 3;
        case DType::BF16: return 6;
        case DType::F32:  return 7;
        case DType::F16:  return 8;
        case DType::F8:   return 10;  // verified from compiled Gaudi2 recipe (0x11f0a)
        default: throw std::invalid_argument("direct_mme: unsupported TPC dtype");
    }
}

void push_reg(std::vector<char>& raw, uint16_t addr, uint32_t val, bool eng_barrier = false) {
    const size_t off = raw.size();
    raw.resize(off + sizeof(packet_wreg32), 0);
    auto* p = reinterpret_cast<packet_wreg32*>(raw.data() + off);
    p->opcode = PACKET_WREG_32;
    p->msg_barrier = 1;
    p->reg_offset = addr;
    p->value = val;
    p->eng_barrier = eng_barrier ? 1 : 0;
}

void push_regs_bulk(std::vector<char>& raw, uint16_t addr, const uint32_t* words, uint32_t count, bool eng_barrier) {
    uint32_t i = 0;
    if ((addr & 7) == 4 && count >= 3) {
        push_reg(raw, addr, words[0], eng_barrier);
        i = 1;
    }
    if (((addr + i * 4) & 7) == 0 && count - i >= 2) {
        const uint32_t pairs = (count - i) / 2;
        const size_t off = raw.size();
        raw.resize(off + sizeof(packet_wreg_bulk) + size_t(pairs) * 8, 0);
        auto* p = reinterpret_cast<packet_wreg_bulk*>(raw.data() + off);
        p->size64 = pairs;
        p->opcode = PACKET_WREG_BULK;
        p->msg_barrier = 1;
        p->eng_barrier = eng_barrier ? 1 : 0;
        p->reg_offset = uint16_t(addr + i * 4);
        std::memcpy(raw.data() + off + sizeof(packet_wreg_bulk), words + i, size_t(pairs) * 8);
        i += pairs * 2;
    }
    for (; i < count; ++i) push_reg(raw, uint16_t(addr + i * 4), words[i], i == 0 && eng_barrier);
}

// Byte offset (inside `raw`, scanning packets from `from`) of the payload dword written to register `reg`; -1 if none.
int64_t find_reg_payload(const std::vector<char>& raw, size_t from, uint32_t reg) {
    for (size_t off = from; off + 8 <= raw.size();) {
        const auto* p = reinterpret_cast<const packet_wreg32*>(raw.data() + off);
        if (p->opcode == PACKET_WREG_BULK) {
            const auto* b = reinterpret_cast<const packet_wreg_bulk*>(p);
            const uint32_t first = b->reg_offset, n = b->size64 * 2u;
            if (reg >= first && reg < first + n * 4u) return int64_t(off + 8 + (reg - first));
            off += 8 + size_t(b->size64) * 8;
            continue;
        }
        if (p->opcode == PACKET_WREG_32 && p->reg_offset == reg)
            return int64_t(off + offsetof(packet_wreg32, value));
        off += 8;
    }
    return -1;
}

void push_fence(std::vector<char>& raw, uint32_t id = 3) {
    const size_t off = raw.size();
    raw.resize(off + sizeof(packet_fence), 0);
    auto* f = reinterpret_cast<packet_fence*>(raw.data() + off);
    f->opcode = PACKET_FENCE;
    f->msg_barrier = 1;
    f->id = id;
    f->target_val = 1;
    f->dec_val = 1;
}

void push_finish(std::vector<char>& raw, bool eng_barrier = true) {
    const size_t off = raw.size();
    raw.resize(off + sizeof(packet_nop), 0);
    auto* p = reinterpret_cast<packet_nop*>(raw.data() + off);
    p->opcode = PACKET_NOP;
    p->eng_barrier = eng_barrier ? 1 : 0;
    p->msg_barrier = 1;
    p->swtc = 1;
}

bool validate_static_program(const std::vector<char>& raw, uint32_t group) {
    if (raw.empty() || (raw.size() % 8) != 0 || raw.size() > 8191) return false;
    for (size_t off = 0; off < raw.size(); off += 8) {
        const auto* p = reinterpret_cast<const packet_wreg32*>(raw.data() + off);
        if (p->opcode == PACKET_WREG_BULK) {
            const auto* b = reinterpret_cast<const packet_wreg_bulk*>(p);
            const uint32_t first = b->reg_offset;
            const uint32_t last = first + b->size64 * 8u - 4u;
            const bool allowed = (group == SCAL_MME_COMPUTE_GROUP)
                                     ? (first >= 0xb008 && last <= 0xb2dc)
                                     : (first >= 0xb5dc && last < 0xb5dc + sizeof(gaudi2::Gaudi2TpcDesc));
            if (!allowed || !b->size64 || !b->msg_barrier || b->swtc || off + 8 + size_t(b->size64) * 8 > raw.size())
                return false;
            off += size_t(b->size64) * 8;
            continue;
        }
        if (p->opcode == PACKET_WREG_32) {
            const uint32_t reg = p->reg_offset;
            const bool allowed = (group == SCAL_MME_COMPUTE_GROUP)
                                     ? ((reg >= 0xb008 && reg <= 0xb2dc) || reg == 0xb004)
                                     : ((reg >= 0xb5dc && reg < 0xb5dc + sizeof(gaudi2::Gaudi2TpcDesc)) ||
                                        reg == 0xbd94 || reg == 0xbd98);
            if (!allowed || !p->msg_barrier || p->swtc) return false;
        } else if (p->opcode == PACKET_FENCE) {
            const auto* f = reinterpret_cast<const packet_fence*>(p);
            if ((f->id != 3 && f->id != 2) || f->target_val != 1 || f->dec_val != 1)
                return false;
        } else if (p->opcode == PACKET_NOP) {
            const auto* n = reinterpret_cast<const packet_nop*>(p);
            if (off + 8 != raw.size() || !n->swtc) return false;
        } else {
            return false;
        }
    }
    return reinterpret_cast<const packet_nop*>(raw.data() + raw.size() - 8)->opcode == PACKET_NOP;
}

std::vector<char> extract_elf_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("direct_mme: cannot open TPC object " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
    if (buf.size() < EI_NIDENT || std::memcmp(buf.data(), ELFMAG, SELFMAG) != 0)
        throw std::runtime_error("direct_mme: not an ELF file " + path);
    if (buf[EI_CLASS] == ELFCLASS32) {
        if (buf.size() < sizeof(Elf32_Ehdr)) throw std::runtime_error("direct_mme: truncated ELF32 " + path);
        const auto* eh = reinterpret_cast<const Elf32_Ehdr*>(buf.data());
        const auto* shs = reinterpret_cast<const Elf32_Shdr*>(buf.data() + eh->e_shoff);
        const char* shstr = buf.data() + shs[eh->e_shstrndx].sh_offset;
        for (uint16_t i = 0; i < eh->e_shnum; ++i) {
            if (std::strcmp(shstr + shs[i].sh_name, ".text") == 0) {
                if (shs[i].sh_offset + shs[i].sh_size > buf.size())
                    throw std::runtime_error("direct_mme: invalid .text bounds in " + path);
                return std::vector<char>(buf.begin() + shs[i].sh_offset, buf.begin() + shs[i].sh_offset + shs[i].sh_size);
            }
        }
    } else if (buf[EI_CLASS] == ELFCLASS64) {
        if (buf.size() < sizeof(Elf64_Ehdr)) throw std::runtime_error("direct_mme: truncated ELF64 " + path);
        const auto* eh = reinterpret_cast<const Elf64_Ehdr*>(buf.data());
        const auto* shs = reinterpret_cast<const Elf64_Shdr*>(buf.data() + eh->e_shoff);
        const char* shstr = buf.data() + shs[eh->e_shstrndx].sh_offset;
        for (uint16_t i = 0; i < eh->e_shnum; ++i) {
            if (std::strcmp(shstr + shs[i].sh_name, ".text") == 0) {
                if (shs[i].sh_offset + shs[i].sh_size > buf.size())
                    throw std::runtime_error("direct_mme: invalid .text bounds in " + path);
                return std::vector<char>(buf.begin() + shs[i].sh_offset, buf.begin() + shs[i].sh_offset + shs[i].sh_size);
            }
        }
    } else {
        throw std::runtime_error("direct_mme: unsupported ELF class in " + path);
    }
    throw std::runtime_error("direct_mme: .text section not found in " + path);
}

std::array<uint32_t, 182> parse_mme_desc_text(const std::string& text) {
    std::array<uint32_t, 182> words{};
    auto set = [&](uint32_t off, uint32_t val) { words[(off - 8) / 4] = val; };

    auto slice = [&](const char* start, const char* end) -> std::string {
        const auto p0 = text.find(start);
        if (p0 == std::string::npos) throw std::runtime_error(std::string("mme_desc: missing ") + start);
        const auto p1 = text.find(end, p0 + std::strlen(start));
        if (p1 == std::string::npos) throw std::runtime_error(std::string("mme_desc: missing ") + end);
        return text.substr(p0 + std::strlen(start), p1 - (p0 + std::strlen(start)));
    };

    const struct { const char* s; const char* e; uint32_t off; uint32_t cnt; } sections[] = {
        {"Brains:", "Header:", 0x28, 2},
        {"Header:", "Ctrl:", 0x30, 2},
        {"Ctrl:", "Tensor: tensorA", 0x38, 2},
        {"Tensor: tensorA", "Tensor: tensorB", 0x40, 22},
        {"Tensor: tensorB", "Tensor: tensorCout", 0x98, 22},
        {"Tensor: tensorCout", "SyncObject:", 0xF0, 22},
        {"SyncObject:", "AguIn:", 0x148, 5},
    };
    const std::regex dw_re(R"(dw\[(\d+)\]\s*:\s*0x([0-9a-fA-F]+))");
    for (const auto& sec : sections) {
        const std::string blk = slice(sec.s, sec.e);
        uint32_t got = 0;
        for (std::sregex_iterator it(blk.begin(), blk.end(), dw_re), end; it != end; ++it) {
            const uint32_t idx = std::stoul((*it)[1]);
            const uint32_t val = uint32_t(std::stoull((*it)[2], nullptr, 16));
            if (idx >= sec.cnt) throw std::runtime_error("mme_desc: dw index out of range");
            set(sec.off + idx * 4, val);
            ++got;
        }
        if (got != sec.cnt) throw std::runtime_error(std::string("mme_desc: incomplete section ") + sec.s);
    }

    auto parse_agu = [&](const char* tag, uint32_t base, uint32_t cnt) {
        const std::regex agu_re(std::string(tag) + R"(\[(\d+)\]\[(MASTER|SLAVE)\]:\s*\[([^\]]+)\])");
        uint32_t got = 0;
        for (std::sregex_iterator it(text.begin(), text.end(), agu_re), end; it != end; ++it) {
            const uint32_t i = std::stoul((*it)[1]);
            const bool slave = (*it)[2] == "SLAVE";
            const std::string arr = (*it)[3];
            uint32_t j = 0;
            size_t pos = 0;
            while (pos < arr.size() && j < 5) {
                while (pos < arr.size() && (arr[pos] == ' ' || arr[pos] == ',')) ++pos;
                size_t next = arr.find(',', pos);
                if (next == std::string::npos) next = arr.size();
                const int64_t v = std::stoll(arr.substr(pos, next - pos));
                set(base + i * 40 + (slave ? 20u : 0u) + j * 4, uint32_t(v));
                pos = next + 1;
                ++j;
            }
            if (j != 5) throw std::runtime_error("mme_desc: AGU entry != 5 values");
            ++got;
        }
        if (got != cnt * 2) throw std::runtime_error(std::string("mme_desc: incomplete ") + tag);
    };
    parse_agu("aguIn", 0x15C, 5);
    parse_agu("aguOut", 0x22C, 2);

    auto find_hex = [&](const std::string& s, const std::string& pat) -> uint32_t {
        std::smatch m;
        if (!std::regex_search(s, m, std::regex(pat))) throw std::runtime_error("mme_desc: missing " + pat);
        return uint32_t(std::stoull(m[1], nullptr, 16));
    };
    set(0x224, find_hex(text, R"(spatialSizeMinus1A\s*:\s*0x([0-9a-fA-F]+))"));
    set(0x228, find_hex(text, R"(spatialSizeMinus1B\s*:\s*0x([0-9a-fA-F]+))"));
    set(0x27C, find_hex(text, R"(spatialSizeMinus1Cout\s*:\s*0x([0-9a-fA-F]+))"));

    const std::string conv = slice("Conv:", "OuterLoop:");
    set(0x280, find_hex(conv, R"(dw\[0\]:\s*0x([0-9a-fA-F]+))"));
    uint32_t h[4] = {};
    const std::regex w_re(R"(w\[([1-4])\]:\s*0x([0-9a-fA-F]+))");
    for (std::sregex_iterator it(conv.begin(), conv.end(), w_re), end; it != end; ++it)
        h[std::stoul((*it)[1]) - 1] = uint32_t(std::stoull((*it)[2], nullptr, 16));
    set(0x284, h[0] | (h[1] << 16));
    set(0x288, h[2] | (h[3] << 16));

    const std::string outer = slice("OuterLoop:", "numIterationsMinus1");
    set(0x28C, find_hex(outer, R"(dw:\s*0x([0-9a-fA-F]+))"));
    set(0x290, find_hex(text, R"(numIterationsMinus1\s*:\s*0x([0-9a-fA-F]+))"));

    const std::string rep = slice("SBRepeat:", "Fp8Bias:");
    set(0x294, find_hex(rep, R"(dw:\s*0x([0-9a-fA-F]+))"));

    const std::string bias = slice("Fp8Bias:", "AxiUserData:");
    const uint32_t ba = find_hex(bias, R"(a:\s*0x([0-9a-fA-F]+))");
    const uint32_t bb = find_hex(bias, R"(b:\s*0x([0-9a-fA-F]+))");
    const uint32_t bo = find_hex(bias, R"(out:\s*0x([0-9a-fA-F]+))");
    set(0x298, (ba & 0xfu) | ((bb & 0xfu) << 4) | ((bo & 0xfu) << 8));
    set(0x29C, kRateLimiter);

    // Zero base addresses and so0Addr (patched at plan build time).
    for (uint32_t off : {0x08u, 0x0cu, 0x10u, 0x14u, 0x18u, 0x1cu, 0x20u, 0x24u, 0x14cu}) set(off, 0);
    return words;
}

double now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

}  // namespace

MmeTemplate MmeTemplate::load_dir(const std::string& dir) {
    MmeTemplate t;
    const std::regex fn_re(R"(act_(\d+)_mme_descriptor_dump_(\d+)\.txt$)");
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_search(name, m, fn_re)) continue;
        std::ifstream f(entry.path());
        std::string text((std::istreambuf_iterator<char>(f)), {});
        MmeDesc d;
        d.act = uint32_t(std::stoul(m[1]));
        d.engine = uint32_t(std::stoul(m[2]));
        d.words = parse_mme_desc_text(text);
        if (d.signals()) ++t.signal_count;
        t.engine_mask |= (1u << d.engine);
        t.descs.push_back(d);
    }
    std::sort(t.descs.begin(), t.descs.end(), [](const MmeDesc& a, const MmeDesc& b) {
        return a.act != b.act ? a.act < b.act : a.engine < b.engine;
    });
    if (t.descs.empty()) throw std::runtime_error("direct_mme: no descriptors in " + dir);
    return t;
}

struct PreparedEcb {
    uint64_t meta_core_addr = 0;
    uint64_t code_dev_addr = 0;
    uint32_t group = 0;
    std::vector<uint32_t> engine_offsets;
    std::vector<char> shadow;
    bool dirty = false;
};

// 0 / 1 from the environment, -1 when the variable is unset
static int env_switch(const char* name) {
    const char* e = std::getenv(name);
    return (e && *e) ? int(*e != '0') : -1;
}

struct DirectRuntime::Impl {
    DirectRuntime::Options opt;
    synDeviceId dev_id = 0;
    int fd = -1;
    uint64_t sram_base = 0;
    uint64_t sram_size = 0;
    uint64_t sram_usable = 0;

    scal_handle_t scal = nullptr;
    scal_stream_handle_t stream = nullptr;
    scal_host_fence_counter_handle_t fence = nullptr;
    scal_pool_handle_t host_pool = nullptr;
    scal_pool_handle_t hbm_pool = nullptr;
    // the prepared plans' share of the pools (256 MiB / 128 MiB); shared with the plans, which may outlive the runtime
    struct PoolUse { uint64_t code = 0, meta = 0; uint32_t plans = 0; };
    std::shared_ptr<PoolUse> pool_use = std::make_shared<PoolUse>();
    scal_pool_handle_t bin_pool = nullptr;
    scal_buffer_handle_t cmd_buf = nullptr;
    void* commands = nullptr;
    volatile const uint64_t* host_counter = nullptr;
    const uint64_t* host_req_counter = nullptr;

    uint32_t pi = 0;
    uint32_t run_start = 0;
    uint32_t sub_align = 1;
    uint32_t cmd_align = 64;
    uint32_t host_sob = 0;
    uint32_t sob_base = 0, sob_count = 0;
    uint32_t mon_base = 0, mon_count = 0;
    uint64_t sched_dccm_mq = 0;
    uint64_t host_sob_addr64 = 0;
    uint32_t chain_sob = 0;
    uint32_t chain_mon_fence = 0;
    uint32_t chain_mon_host = 0;
    uint32_t chain_mon_both = 0;
    static constexpr uint32_t kChainFenceId = 63;
    bool chain_enabled = false;

    std::array<uint64_t, 2048> mon_cfg{};
    std::array<uint8_t, 2048> mon_wr{};
    bool mon_rearm_only = true;

    std::map<std::string, uint64_t> isa_cache;
    synStreamHandle patch_stream = nullptr;
    void* patch_stage = nullptr;
    size_t patch_stage_cap = 0;
    scal_buffer_handle_t pdma_buf = nullptr;
    void* pdma_host = nullptr;
    uint64_t pdma_dev = 0;
    size_t pdma_cap = 0;

    ~Impl() {
        if (patch_stream) synStreamDestroy(patch_stream);
        if (patch_stage) synHostFree(dev_id, patch_stage, 0);
        if (pdma_buf) scal_free_buffer(pdma_buf);
    }

    void ensure_pdma_buf(size_t bytes) {
        if (pdma_buf && bytes <= pdma_cap) return;
        if (pdma_buf) { scal_free_buffer(pdma_buf); pdma_buf = nullptr; }
        pdma_cap = std::max<size_t>((bytes + 4095) & ~size_t(4095), 256u << 10);
        if (scal_allocate_buffer(host_pool, pdma_cap, &pdma_buf))
            throw std::runtime_error("direct_mme: cannot allocate host_shared pdma buffer");
        scal_buffer_info_t bi{};
        if (scal_buffer_get_info(pdma_buf, &bi) || !bi.host_address || !bi.device_address)
            throw std::runtime_error("direct_mme: host_shared pdma buffer info missing host/device addr");
        pdma_host = bi.host_address;
        pdma_dev = bi.device_address;
    }

    void emit_pdma_batch(uint32_t group, uint32_t dir, const sched_arc_pdma_commands_params_t* ps, uint32_t count, uint32_t sig_sob_abs) {
        // count must be 1 or 2 so command size (16 + 20*count <= 56) fits inside cmd_align (64).
        struct __attribute__((aligned(4), packed)) Cmd2 {
            sched_arc_cmd_pdma_batch_transfer_t hdr;
            sched_arc_pdma_commands_params_t p[2];
        } cmd{};
        cmd.hdr.opcode = SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER;
        cmd.hdr.engine_group_type = group;
        cmd.hdr.watch_dog_sig_value = 0;
        cmd.hdr.signal_to_cg = 0;
        cmd.hdr.workload_type = ENG_PDMA_ARC_CMD_BATCH_NO_FRAGMENTATION;
        cmd.hdr.stream_ctxt_id = dir;
        cmd.hdr.api_id = 0;
        cmd.hdr.has_payload = 1;
        cmd.hdr.batch_count = count;
        cmd.hdr.pay_data = 0x80000001u;
        cmd.hdr.pay_addr = uint32_t(mmDCORE2_SYNC_MNGR_OBJS_BASE + sig_sob_abs * 4);
        for (uint32_t i = 0; i < count; ++i) cmd.p[i] = ps[i];
        append(&cmd, 16u + 20u * count);
    }

    void emit_pdma_1(uint32_t group, uint32_t dir, uint64_t src, uint64_t dst, uint32_t bytes, uint32_t sig_sob_abs) {
        sched_arc_pdma_commands_params_t p{bytes, src, dst};
        emit_pdma_batch(group, dir, &p, 1, sig_sob_abs);
    }

    void ensure_patch_buf(size_t bytes) {
        if (!patch_stream) check(synStreamCreateGeneric(&patch_stream, dev_id, 0), "direct_mme patch stream");
        if (bytes > patch_stage_cap) {
            if (patch_stage) synHostFree(dev_id, patch_stage, 0);
            patch_stage_cap = std::max<size_t>(bytes, 256u << 10);
            check(synHostMalloc(dev_id, patch_stage_cap, 0, &patch_stage), "direct_mme patch stage");
        }
    }

    void begin() { run_start = pi; }
    char* slot() { return static_cast<char*>(commands) + (pi & (kOwnRing - 1)); }

    void reserve(uint32_t bytes) {
        const uint32_t rem = cmd_align - (pi % cmd_align);
        const uint32_t pad = rem < bytes ? rem : 0;
        if (pi - run_start + pad + bytes > kOwnBudget)
            throw std::runtime_error("direct_mme: SCAL command buffer overflow (>64 KiB)");
        if (pad) {
            auto* nop = reinterpret_cast<sched_arc_cmd_nop_t*>(slot());
            nop->opcode = SCHED_COMPUTE_ARC_CMD_NOP;
            nop->padding_count = pad / 4 - 1;
            pi += pad;
        }
    }

    void append(const void* data, uint32_t bytes) {
        if (bytes > cmd_align || (bytes % 4) != 0) throw std::runtime_error("direct_mme: bad SCAL cmd size");
        reserve(bytes);
        std::memcpy(slot(), data, bytes);
        pi += bytes;
    }

    void lbw(uint64_t addr, uint32_t val) {
        sched_arc_cmd_lbw_write_t w{};
        w.opcode = SCHED_COMPUTE_ARC_CMD_LBW_WRITE;
        w.dst_addr = uint32_t(addr);
        w.src_data = val;
        append(&w, sizeof(w));
    }

    void fence_wait(uint32_t fence_id, uint32_t target = 1) {
        sched_arc_cmd_fence_wait_t w{};
        w.opcode = SCHED_COMPUTE_ARC_CMD_FENCE_WAIT;
        w.fence_id = fence_id;
        w.target = target;
        append(&w, sizeof(w));
    }

    void cfg_chain_monitor_1(uint32_t monitor, uint32_t sob, uint64_t dst, uint32_t payload) {
        const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
        packet_msg_short f{};
        f.mon_config_register.msb_sid = (sob / 8) >> 8;
        lbw(base + 0x10000 + monitor * 4, f.value);
        lbw(base + 0x8000 + monitor * 4, uint32_t(dst));
        lbw(base + 0xa000 + monitor * 4, uint32_t(dst >> 32));
        lbw(base + 0xc000 + monitor * 4, payload);
    }

    void cfg_chain_monitor_2(uint32_t monitor, uint32_t sob, uint64_t dst0, uint32_t pay0, uint64_t dst1, uint32_t pay1) {
        const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
        const uint64_t dsts[2] = {dst0, dst1};
        const uint32_t pays[2] = {pay0, pay1};
        for (uint32_t i = 0; i < 2; ++i) {
            packet_msg_short f{};
            f.mon_config_register.msb_sid = (sob / 8) >> 8;
            f.mon_config_register.wr_num = i ? 0 : 1;
            const uint32_t m = monitor + i;
            lbw(base + 0x10000 + m * 4, f.value);
            lbw(base + 0x8000 + m * 4, uint32_t(dsts[i]));
            lbw(base + 0xa000 + m * 4, uint32_t(dsts[i] >> 32));
            lbw(base + 0xc000 + m * 4, pays[i]);
        }
    }

    void arm_only(uint32_t monitor, uint32_t sob, uint32_t target) {
        const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
        packet_msg_short f{};
        f.mon_arm_register.sync_group_id = sob / 8;
        f.mon_arm_register.mask = ~(1u << (sob & 7));
        f.mon_arm_register.mode = 0;  // >= target
        f.mon_arm_register.sync_value = target;
        lbw(base + 0xe000 + monitor * 4, f.value);
    }

    void submit() {
        __sync_synchronize();
        const int rc = scal_stream_submit(stream, pi & (kOwnRing - 1), sub_align);
        if (rc) throw std::runtime_error("direct_mme: scal_stream_submit failed rc=" + std::to_string(rc));
    }

    void wait_fence(uint64_t credits, uint64_t timeout_us = 2000000) {
        if (opt.spin_wait && host_counter && host_req_counter) {
            const uint64_t target = *host_req_counter + credits;
            if (*host_counter < target) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
                for (uint32_t it = 0; *host_counter < target; ++it) {
                    if ((it & 255u) == 255u && std::chrono::steady_clock::now() >= deadline) break;
#if defined(__x86_64__) || defined(_M_X64)
                    __builtin_ia32_pause();
#endif
                }
            }
        }
        const int rc = scal_host_fence_counter_wait(fence, credits, timeout_us);
        if (rc) {
            const uint64_t cur = host_counter ? *host_counter : 0;
            throw std::runtime_error("direct_mme: scal_host_fence_counter_wait failed rc=" + std::to_string(rc) +
                                     " (want=" + std::to_string(credits) + ", host_ctr=" + std::to_string(cur) + ")");
        }
    }

    void arm_monitor(uint32_t monitor, uint32_t sob, uint32_t target, uint64_t dst) {
        const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
        packet_msg_short f{};
        const uint32_t sl = monitor % 2048;
        const uint64_t key = dst ^ (uint64_t((sob / 8) >> 8) << 1) ^ 1ull;
        if (!mon_rearm_only || mon_cfg[sl] != key || mon_wr[sl] != 0) {
            f.mon_config_register.msb_sid = (sob / 8) >> 8;
            lbw(base + 0x10000 + monitor * 4, f.value);
            lbw(base + 0x8000 + monitor * 4, uint32_t(dst));
            lbw(base + 0xa000 + monitor * 4, uint32_t(dst >> 32));
            lbw(base + 0xc000 + monitor * 4, 1);
            mon_cfg[sl] = key;
            mon_wr[sl] = 0;
        }
        f.value = 0;
        f.mon_arm_register.sync_group_id = sob / 8;
        f.mon_arm_register.mask = ~(1u << (sob & 7));
        f.mon_arm_register.mode = 0;  // >= target
        f.mon_arm_register.sync_value = target;
        lbw(base + 0xe000 + monitor * 4, f.value);
    }

    void arm_monitor_multi(uint32_t monitor, uint32_t count, uint32_t sob, uint32_t target, const uint64_t* dsts) {
        const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
        packet_msg_short f{};
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t m = monitor + i, sl = m % 2048;
            const uint8_t wr = i ? 0 : uint8_t(count - 1);
            const uint64_t key = dsts[i] ^ (uint64_t((sob / 8) >> 8) << 1) ^ 1ull;
            if (mon_rearm_only && mon_cfg[sl] == key && mon_wr[sl] == wr) continue;
            f.value = 0;
            f.mon_config_register.msb_sid = (sob / 8) >> 8;
            f.mon_config_register.wr_num = wr;
            lbw(base + 0x10000 + m * 4, f.value);
            lbw(base + 0x8000 + m * 4, uint32_t(dsts[i]));
            lbw(base + 0xa000 + m * 4, uint32_t(dsts[i] >> 32));
            lbw(base + 0xc000 + m * 4, 1);
            mon_cfg[sl] = key;
            mon_wr[sl] = wr;
        }
        f.value = 0;
        f.mon_arm_register.sync_group_id = sob / 8;
        f.mon_arm_register.mask = ~(1u << (sob & 7));
        f.mon_arm_register.mode = 0;
        f.mon_arm_register.sync_value = target;
        lbw(base + 0xe000 + monitor * 4, f.value);
    }

    void queue_ecb(const PreparedEcb& job) {
        struct {
            sched_arc_cmd_update_recipe_base_v2_t command;
            uint64_t address;
        } update{};
        update.command.opcode = SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2;
        update.command.num_recipe_addrs = 1;
        update.command.num_engine_group_type = 1;
        update.command.recipe_base_index0 = 6;
        update.command.engine_group_type[0] = job.group;
        update.address = job.code_dev_addr;
        append(&update, sizeof(update));
        sched_arc_cmd_dispatch_compute_ecb_list_t dispatch{};
        dispatch.opcode = SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3;
        dispatch.engine_group_type = job.group;
        dispatch.cmpt_stream_index = 3;
        dispatch.single_static_chunk = 1;
        dispatch.single_dynamic_chunk = 1;
        dispatch.static_ecb_list_offset = 1;
        dispatch.dynamic_ecb_list_addr_256 = job.meta_core_addr >> 8;
        append(&dispatch, sizeof(dispatch));
    }

    static void build_ecb_bytes(uint32_t group, const std::vector<std::vector<char>>& programs,
                                std::vector<char>& combined, std::vector<uint32_t>& offsets) {
        const uint32_t engines = uint32_t(programs.size());
        if (!engines || engines > 30) throw std::runtime_error("direct_mme: bad engine count");
        offsets.clear();
        combined.clear();
        for (const auto& raw : programs) {
            if (!validate_static_program(raw, group))
                throw std::runtime_error("direct_mme: static QMAN program validation failed");
            combined.resize((combined.size() + 63) & ~size_t(63), 0);
            offsets.push_back(uint32_t(combined.size()));
            combined.insert(combined.end(), raw.begin(), raw.end());
        }
        combined.resize((combined.size() + 255) & ~size_t(255), 0);
    }

    static void fill_ecb_meta(void* meta_dst, const std::vector<std::vector<char>>& programs,
                              const std::vector<uint32_t>& offsets) {
        const uint32_t engines = uint32_t(programs.size());
        std::memset(meta_dst, 0, 512);
        for (uint32_t part = 0; part < 2; ++part) {
            auto* sz = reinterpret_cast<eng_arc_cmd_list_size_t*>(static_cast<char*>(meta_dst) + 256 * part);
            sz->cmd_type = ECB_CMD_LIST_SIZE;
            sz->topology_start = 1;
            sz->list_size = 256;
        }
        auto* dyn = reinterpret_cast<eng_arc_cmd_nop_t*>(static_cast<char*>(meta_dst) + 4);
        dyn->cmd_type = ECB_CMD_NOP;
        dyn->switch_cq = 1;
        dyn->yield = 1;
        dyn->padding = 62;
        auto* entries = reinterpret_cast<eng_arc_cmd_static_desc_v2_t*>(static_cast<char*>(meta_dst) + 260);
        for (uint32_t i = 0; i < engines; ++i) {
            entries[i].cmd_type = ECB_CMD_STATIC_DESC_V2;
            entries[i].cpu_index = i;
            entries[i].addr_index = 6;
            entries[i].size = uint32_t(programs[i].size());
            entries[i].addr_offset = offsets[i];
            entries[i].yield = (i == engines - 1) ? 1 : 0;
        }
        const uint32_t used = 4 + engines * uint32_t(sizeof(*entries));
        auto* pad = reinterpret_cast<eng_arc_cmd_nop_t*>(static_cast<char*>(meta_dst) + 256 + used);
        pad->cmd_type = ECB_CMD_NOP;
        pad->padding = (256 - used) / 4 - 1;
    }

    void upload_and_verify(const std::vector<char>& all_code, uint64_t dst_dev_addr) {
        ensure_patch_buf(all_code.size());
        std::memcpy(patch_stage, all_code.data(), all_code.size());
        check(synMemCopyAsync(patch_stream, reinterpret_cast<uintptr_t>(patch_stage), all_code.size(),
                              dst_dev_addr, HOST_TO_DRAM), "direct_mme upload plan ECBs");
        check(synStreamSynchronize(patch_stream), "direct_mme sync upload plan ECBs");
        static const bool verify_rb = std::getenv("G2E_DIRECT_VERIFY_ECB") != nullptr;
        if (verify_rb) {
            std::memset(patch_stage, 0, all_code.size());
            check(synMemCopyAsync(patch_stream, dst_dev_addr, all_code.size(),
                                  reinterpret_cast<uintptr_t>(patch_stage), DRAM_TO_HOST), "direct_mme readback plan ECBs");
            check(synStreamSynchronize(patch_stream), "direct_mme sync readback plan ECBs");
            if (std::memcmp(patch_stage, all_code.data(), all_code.size()) != 0)
                throw std::runtime_error("direct_mme: plan ECB readback verification mismatch");
        }
    }
};

bool DirectRuntime::ensure_scal_config() {
    if (const char* cur = std::getenv("SCAL_CONFIG_FILE_PATH"); cur && cur[0] != '\0') return true;
    if (const char* cur = std::getenv("SCAL_CFG_FILE_PATH"); cur && cur[0] != '\0') {
        setenv("SCAL_CONFIG_FILE_PATH", cur, 1);
        return true;
    }
    const char* cands[] = {
        "/opt/gaudiinfer/tools/scal-g2e-direct.json",
        G2E_SOURCE_DIR "/tools/scal-g2e-direct.json",
        "tools/scal-g2e-direct.json",
    };
    for (const char* p : cands) {
        if (std::filesystem::exists(p)) {
            setenv("SCAL_CONFIG_FILE_PATH", p, 1);
            setenv("SCAL_CFG_FILE_PATH", p, 1);
            return true;
        }
    }
    return false;
}

DirectRuntime::DirectRuntime(const Device& dev) : impl_(std::make_unique<Impl>()) {
    auto& s = *impl_;
    set_options(Options{});  // the defaults, unless the environment asks otherwise
    s.dev_id = dev.id();
    synDeviceInfoV2 info{};
    check(synDeviceGetInfoV2(s.dev_id, &info), "direct_mme synDeviceGetInfoV2");
    s.fd = int(info.fd);
    s.sram_base = info.sramBaseAddress;
    s.sram_size = info.sramSize;

    if (scal_get_handle_from_fd(s.fd, &s.scal) || !s.scal)
        throw std::runtime_error("direct_mme: scal_get_handle_from_fd failed");
    // SCAL's usable SRAM size: with tpc_barrier_in_sram=true SCAL keeps its 2 KiB TPC barrier kernel at
    // sram_end - 2 KiB and scal_get_sram_size() reports 48 MiB - 2 KiB; with false all 48.00 MiB is free.
    s.sram_usable = s.sram_size;
    if (const uint32_t scal_sram = scal_get_sram_size(s.scal); scal_sram > 0 && uint64_t(scal_sram) < s.sram_size)
        s.sram_usable = scal_sram;
    if (scal_get_stream_handle_by_name(s.scal, "mme_attention0", &s.stream))
        throw std::runtime_error("direct_mme: stream 'mme_attention0' not found (set SCAL_CFG_FILE_PATH=tools/scal-g2e-direct.json)");
    if (scal_get_pool_handle_by_name(s.scal, "host_shared", &s.host_pool) ||
        scal_get_pool_handle_by_name(s.scal, "hbm_shared", &s.hbm_pool) ||
        scal_get_pool_handle_by_name(s.scal, "hbm_binary", &s.bin_pool))
        throw std::runtime_error("direct_mme: missing SCAL memory pools");
    if (scal_allocate_buffer(s.host_pool, kOwnRing, &s.cmd_buf))
        throw std::runtime_error("direct_mme: cannot allocate 64 KiB command buffer");
    scal_buffer_info_t mem{};
    if (scal_buffer_get_info(s.cmd_buf, &mem) || !mem.host_address)
        throw std::runtime_error("direct_mme: cmd_buf info failed");
    std::memset(mem.host_address, 0, kOwnRing);
    if (scal_stream_set_commands_buffer(s.stream, s.cmd_buf))
        throw std::runtime_error("direct_mme: scal_stream_set_commands_buffer failed");

    struct { scal_stream_info_t info; unsigned char pad[512]; } st{};
    if (scal_stream_get_info(s.stream, &st.info))
        throw std::runtime_error("direct_mme: scal_stream_get_info failed");
    s.sub_align = st.info.submission_alignment;
    s.cmd_align = st.info.command_alignment;
    s.commands = mem.host_address;

    if (scal_get_host_fence_counter_handle_by_name(s.scal, "mme_attention_fence0", &s.fence))
        throw std::runtime_error("direct_mme: fence 'mme_attention_fence0' not found");
    struct { scal_host_fence_counter_info_t info; unsigned char pad[512]; } fi{};
    if (scal_host_fence_counter_get_info(s.fence, &fi.info) || fi.info.sm != 2)
        throw std::runtime_error("direct_mme: bad host fence info");
    s.host_sob = fi.info.so_index;
    s.host_counter = fi.info.ctr;
    s.host_req_counter = fi.info.request_counter;

    scal_so_pool_handle_t sos = nullptr;
    scal_monitor_pool_handle_t mons = nullptr;
    if (scal_get_so_pool_handle_by_name(s.scal, "mme_attention_sos", &sos) ||
        scal_get_so_monitor_handle_by_name(s.scal, "mme_attention_monitors", &mons))
        throw std::runtime_error("direct_mme: private SOB/monitor pools not found");
    struct { scal_so_pool_info info; char pad[512]; } si{};
    struct { scal_monitor_pool_info info; char pad[512]; } mi{};
    if (scal_so_pool_get_info(sos, &si.info) || scal_monitor_pool_get_info(mons, &mi.info))
        throw std::runtime_error("direct_mme: SOB/monitor pool info failed");
    if (si.info.smIndex != 2 || mi.info.smIndex != 2 || si.info.size < 8 || mi.info.size < 64)
        throw std::runtime_error("direct_mme: unexpected SOB/monitor pool geometry");
    s.sob_base = si.info.baseIdx;
    s.sob_count = si.info.size;
    s.mon_base = mi.info.baseIdx;
    s.mon_count = mi.info.size;
    s.host_sob_addr64 = (kMme0FenceAddr & ~0x3fffffffull) | (uint32_t(mmDCORE2_SYNC_MNGR_OBJS_BASE + s.host_sob * 4));
    s.chain_sob = s.sob_base + s.sob_count - 1;
    s.chain_mon_fence = s.mon_base + s.mon_count - 4;
    s.chain_mon_host = s.mon_base + s.mon_count - 3;
    s.chain_mon_both = s.mon_base + s.mon_count - 2;

    // 2-iteration NOP+SOB warmup (validates SCAL stream + host fence counter immediately).
    const uint32_t cbytes = s.sub_align < 16 ? 16 : s.sub_align;
    for (uint32_t run = 0; run < 2; ++run) {
        auto* nop = reinterpret_cast<sched_arc_cmd_nop_t*>(static_cast<char*>(s.commands) + run * cbytes);
        nop->opcode = SCHED_COMPUTE_ARC_CMD_NOP;
        nop->padding_count = 0;
        auto* mk = reinterpret_cast<sched_arc_cmd_lbw_write_t*>(reinterpret_cast<char*>(nop) + 4);
        mk->opcode = SCHED_COMPUTE_ARC_CMD_LBW_WRITE;
        mk->dst_addr = uint32_t(mmDCORE2_SYNC_MNGR_OBJS_BASE + s.host_sob * 4);
        mk->src_data = 0x80000001u;
        if (cbytes > 16) {
            auto* pad = reinterpret_cast<sched_arc_cmd_nop_t*>(reinterpret_cast<char*>(nop) + 16);
            pad->opcode = SCHED_COMPUTE_ARC_CMD_NOP;
            pad->padding_count = (cbytes - 16) / 4 - 1;
        }
        s.pi = (run + 1) * cbytes;
        s.submit();
        s.wait_fence(1);
    }
    s.run_start = s.pi;

    // Probe and pre-configure on-device SOB -> Scheduler ARC FENCE_WAIT chaining (eliminates inter-plan host waits).
    if (std::getenv("G2E_DIRECT_NO_CHAIN") == nullptr && s.sob_count >= 16 && s.mon_count >= 1024 &&
        st.info.scheduler_handle) {
        scal_control_core_info_t ci{};
        if (scal_control_core_get_info(st.info.scheduler_handle, &ci) == SCAL_SUCCESS &&
            ci.dccm_message_queue_address != 0) {
            s.sched_dccm_mq = ci.dccm_message_queue_address;
            sched_mon_exp_fence_t fmsg{};
            fmsg.opcode = MON_EXP_FENCE_UPDATE;
            fmsg.fence_id = Impl::kChainFenceId;
            const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
            s.begin();
            s.cfg_chain_monitor_1(s.chain_mon_fence, s.chain_sob, s.sched_dccm_mq, fmsg.raw);
            s.cfg_chain_monitor_1(s.chain_mon_host, s.chain_sob, s.host_sob_addr64, 0x80000001u);
            s.cfg_chain_monitor_2(s.chain_mon_both, s.chain_sob, s.sched_dccm_mq, fmsg.raw,
                                  s.host_sob_addr64, 0x80000001u);
            // Self-test 1: chain_sob -> mon_fence -> FENCE_WAIT(63) -> chain_sob -> mon_both -> FENCE_WAIT(63) -> mon_host
            s.lbw(base + s.chain_sob * 4, 0);
            s.arm_only(s.chain_mon_fence, s.chain_sob, 1);
            s.lbw(base + s.chain_sob * 4, 0x80000001u);
            s.fence_wait(Impl::kChainFenceId, 1);
            s.lbw(base + s.chain_sob * 4, 0);
            s.arm_only(s.chain_mon_both, s.chain_sob, 1);
            s.lbw(base + s.chain_sob * 4, 0x80000001u);
            s.fence_wait(Impl::kChainFenceId, 1);
            s.lbw(base + s.chain_sob * 4, 0);
            s.arm_only(s.chain_mon_host, s.chain_sob, 1);
            s.lbw(base + s.chain_sob * 4, 0x80000001u);
            s.submit();
            s.wait_fence(2, 500000);
            s.chain_enabled = true;
        }
    }
}

DirectRuntime::~DirectRuntime() = default;

void DirectRuntime::set_options(const Options& o) {
    auto& s = *impl_;
    s.opt = o;
    if (const int v = env_switch("G2E_DIRECT_SPIN"); v >= 0) s.opt.spin_wait = v;
    if (const int v = env_switch("G2E_DIRECT_MME_DIFF"); v >= 0) s.opt.mme_diff = v;
    if (const int v = env_switch("G2E_DIRECT_PDMA_PATCH"); v >= 0) s.opt.pdma_patches = v;
}

const DirectRuntime::Options& DirectRuntime::options() const { return impl_->opt; }

uint64_t DirectRuntime::sram_base() const { return impl_->sram_base; }
uint64_t DirectRuntime::sram_size() const { return impl_->sram_size; }
uint64_t DirectRuntime::sram_usable() const { return impl_->sram_usable; }
uint32_t DirectRuntime::sob_base() const { return impl_->sob_base; }
uint32_t DirectRuntime::sob_count() const { return impl_->sob_count; }
uint32_t DirectRuntime::monitor_base() const { return impl_->mon_base; }
uint32_t DirectRuntime::monitor_count() const { return impl_->mon_count; }
uint32_t DirectRuntime::host_sob() const { return impl_->host_sob; }
bool DirectRuntime::chain_enabled() const { return impl_->chain_enabled; }

uint64_t DirectRuntime::load_tpc_elf(const std::string& obj_path) {
    auto& s = *impl_;
    if (auto it = s.isa_cache.find(obj_path); it != s.isa_cache.end()) return it->second;
    const auto isa = extract_elf_text(obj_path);
    const uint64_t cap = (uint64_t(isa.size()) + 65535) / 65536 * 65536;  // 64 KB granules
    if (isa.empty() || isa.size() > 4 * 65536)
        throw std::runtime_error("direct_mme: TPC ISA size out of range: " + std::to_string(isa.size()));
    scal_buffer_handle_t buf = nullptr;
    if (scal_allocate_aligned_buffer(s.bin_pool, cap, 256, &buf))
        throw std::runtime_error("direct_mme: hbm_binary alloc failed");
    scal_buffer_info_t bi{};
    if (scal_buffer_get_info(buf, &bi) || !bi.device_address)
        throw std::runtime_error("direct_mme: hbm_binary info failed");
    void* host = nullptr;
    synStreamHandle upload = nullptr;
    check(synHostMalloc(s.dev_id, cap, 0, &host), "direct_mme isa synHostMalloc");
    check(synStreamCreateGeneric(&upload, s.dev_id, 0), "direct_mme isa synStreamCreateGeneric");
    std::memset(host, 0, cap);
    std::memcpy(host, isa.data(), isa.size());
    check(synMemCopyAsync(upload, reinterpret_cast<uintptr_t>(host), cap, bi.device_address, HOST_TO_DRAM),
          "direct_mme upload TPC ISA");
    check(synStreamSynchronize(upload), "direct_mme sync TPC ISA");
    synStreamDestroy(upload);
    synHostFree(s.dev_id, host, 0);
    s.isa_cache[obj_path] = bi.device_address;
    return bi.device_address;
}

void DirectRuntime::copy_sync(uint64_t src, uint64_t dst, uint64_t bytes) const {
    synStreamHandle st = nullptr;
    check(synStreamCreateGeneric(&st, impl_->dev_id, 0), "direct_mme copy stream");
    check(synMemCopyAsync(st, src, bytes, dst, DRAM_TO_DRAM), "direct_mme copy");
    check(synStreamSynchronize(st), "direct_mme copy sync");
    synStreamDestroy(st);
}

void DirectRuntime::download_sync(uint64_t src_dev_addr, void* dst_host, size_t bytes) {
    pdma_d2h_sync(src_dev_addr, dst_host, bytes);
}

void DirectRuntime::pdma_d2h_sync(uint64_t src_dev_addr, void* dst_host, size_t bytes) {
    auto& s = *impl_;
    s.ensure_pdma_buf(bytes);
    s.begin();
    s.emit_pdma_1(SCAL_PDMA_RX_GROUP, PDMA_DIR_DEV2HOST, src_dev_addr, s.pdma_dev, uint32_t(bytes), s.host_sob);
    s.submit();
    s.wait_fence(1, 500000);
    std::memcpy(dst_host, s.pdma_host, bytes);
}

void DirectRuntime::pdma_h2d_sync(const void* src_host, uint64_t dst_dev_addr, size_t bytes) {
    auto& s = *impl_;
    s.ensure_pdma_buf(bytes);
    std::memcpy(s.pdma_host, src_host, bytes);
    s.begin();
    s.emit_pdma_1(SCAL_PDMA_TX_DATA_GROUP, PDMA_DIR_HOST2DEV, s.pdma_dev, dst_dev_addr, uint32_t(bytes), s.host_sob);
    s.submit();
    s.wait_fence(1, 500000);
}

void DirectRuntime::pdma_h2d_scatter_sync(std::span<const PdmaH2D> xfers) {
    if (xfers.empty()) return;
    auto& s = *impl_;
    size_t total = 0;
    for (const auto& x : xfers) total += (size_t(x.bytes) + 7) & ~size_t(7);
    s.ensure_pdma_buf(total);
    std::vector<sched_arc_pdma_commands_params_t> ps;
    ps.reserve(xfers.size());
    size_t off = 0;
    for (const auto& x : xfers) {
        if (!x.bytes) continue;
        std::memcpy(static_cast<char*>(s.pdma_host) + off, x.src, x.bytes);
        ps.push_back({x.bytes, s.pdma_dev + off, x.dst});
        off += (size_t(x.bytes) + 7) & ~size_t(7);
    }
    if (ps.empty()) return;
    s.begin();
    uint32_t sigs = 0;
    for (size_t i = 0; i < ps.size(); i += 2) {
        const uint32_t cnt = uint32_t(std::min<size_t>(2, ps.size() - i));
        s.emit_pdma_batch(SCAL_PDMA_TX_DATA_GROUP, PDMA_DIR_HOST2DEV, &ps[i], cnt, s.host_sob);
        ++sigs;
    }
    s.submit();
    s.wait_fence(sigs, 500000);
}

void DirectRuntime::pdma_d2h_scatter_sync(std::span<const PdmaD2H> xfers) {
    if (xfers.empty()) return;
    auto& s = *impl_;
    size_t total = 0;
    for (const auto& x : xfers) total += (size_t(x.bytes) + 7) & ~size_t(7);
    s.ensure_pdma_buf(total);
    std::vector<sched_arc_pdma_commands_params_t> ps;
    std::vector<std::tuple<size_t, void*, uint32_t>> cp;
    ps.reserve(xfers.size());
    cp.reserve(xfers.size());
    size_t off = 0;
    for (const auto& x : xfers) {
        if (!x.bytes) continue;
        ps.push_back({x.bytes, x.src, s.pdma_dev + off});
        cp.push_back({off, x.dst, x.bytes});
        off += (size_t(x.bytes) + 7) & ~size_t(7);
    }
    if (ps.empty()) return;
    s.begin();
    uint32_t sigs = 0;
    for (size_t i = 0; i < ps.size(); i += 2) {
        const uint32_t cnt = uint32_t(std::min<size_t>(2, ps.size() - i));
        s.emit_pdma_batch(SCAL_PDMA_RX_GROUP, PDMA_DIR_DEV2HOST, &ps[i], cnt, s.host_sob);
        ++sigs;
    }
    s.submit();
    s.wait_fence(sigs, 500000);
    for (const auto& [o, dst, nb] : cp) std::memcpy(dst, static_cast<const char*>(s.pdma_host) + o, nb);
}

std::vector<float> DirectRuntime::interleave_f32_scales(std::span<const float> src) {
    if ((src.size() % 128) != 0) throw std::invalid_argument("interleave_f32_scales: size must be multiple of 128");
    std::vector<float> out(src.size());
    for (size_t base = 0; base < src.size(); base += 128) {
        for (size_t k = 0; k < 64; ++k) {
            out[base + k] = src[base + 2 * k];
            out[base + 64 + k] = src[base + 2 * k + 1];
        }
    }
    return out;
}

struct DirectDynMmeSet::Impl {
    DirectRuntime& rt;
    struct DynStage {
        bool active = false;
        bool has_wait = false;
        uint32_t wait_target = 0;
        uint32_t sig_credits = 0;
        std::vector<PreparedEcb> ecbs;
    };
    struct DiffKey {
        const MmeTemplate* prev_t;
        const MmeTemplate* cur_t;
        uint8_t prev_act;
        uint8_t cur_act;
        uint8_t eng;
        bool operator<(const DiffKey& o) const {
            return std::tie(prev_t, cur_t, prev_act, cur_act, eng) <
                   std::tie(o.prev_t, o.cur_t, o.prev_act, o.cur_act, o.eng);
        }
    };
    struct DiffBlob {
        std::vector<char> raw;
        int32_t sob_byte_off = -1;  // byte offset inside `raw` of kSobWord (0x14c) if present
    };
    std::vector<DynStage> stages;
    uint32_t max_ecbs = 0;
    size_t max_code_bytes = 0;
    scal_buffer_handle_t meta_buf = nullptr;
    scal_buffer_handle_t code_buf = nullptr;
    void* meta_host = nullptr;
    uint64_t meta_core = 0;
    uint64_t code_dev = 0;
    uint32_t ecb_used = 0;
    size_t code_used = 0;
    std::shared_ptr<DirectRuntime::Impl::PoolUse> pool_use;
    alignas(64) char prog_buf[2][8192];
    std::map<DiffKey, DiffBlob> diff_cache;

    Impl(DirectRuntime& r, uint32_t num_stages, uint32_t max_e, size_t max_c)
        : rt(r), stages(num_stages), max_ecbs(max_e), max_code_bytes((max_c + 255) & ~size_t(255)) {
        auto& s = rt.impl();
        const int rc_h = scal_allocate_aligned_buffer(s.host_pool, size_t(max_ecbs) * 512, 256, &meta_buf);
        const int rc_d = rc_h ? 0 : scal_allocate_aligned_buffer(s.hbm_pool, max_code_bytes, 256, &code_buf);
        if (rc_h || rc_d)
            throw std::runtime_error("direct_mme: DirectDynMmeSet alloc failed");
        scal_buffer_info_t mi{}, ci{};
        if (scal_buffer_get_info(meta_buf, &mi) || scal_buffer_get_info(code_buf, &ci) ||
            !mi.host_address || !mi.core_address || (mi.core_address % 256) != 0)
            throw std::runtime_error("direct_mme: DirectDynMmeSet buffer info failed");
        meta_host = mi.host_address;
        meta_core = mi.core_address;
        code_dev = ci.device_address;
        pool_use = s.pool_use;
        pool_use->code += max_code_bytes;
        pool_use->meta += size_t(max_ecbs) * 512;
        // Pre-populate invariant 512B ECB headers for all slots once (flush_chunk only patches 2 sizes + 1 offset!).
        std::memset(meta_host, 0, size_t(max_ecbs) * 512);
        for (uint32_t slot = 0; slot < max_ecbs; ++slot) {
            char* md = static_cast<char*>(meta_host) + size_t(slot) * 512;
            for (uint32_t part = 0; part < 2; ++part) {
                auto* sz = reinterpret_cast<eng_arc_cmd_list_size_t*>(md + 256 * part);
                sz->cmd_type = ECB_CMD_LIST_SIZE;
                sz->topology_start = 1;
                sz->list_size = 256;
            }
            auto* dyn = reinterpret_cast<eng_arc_cmd_nop_t*>(md + 4);
            dyn->cmd_type = ECB_CMD_NOP;
            dyn->switch_cq = 1;
            dyn->yield = 1;
            dyn->padding = 62;
            auto* entries = reinterpret_cast<eng_arc_cmd_static_desc_v2_t*>(md + 260);
            for (uint32_t i = 0; i < 2; ++i) {
                entries[i].cmd_type = ECB_CMD_STATIC_DESC_V2;
                entries[i].cpu_index = i;
                entries[i].addr_index = 6;
                entries[i].yield = (i == 1) ? 1 : 0;
            }
            const uint32_t used = 4 + 2 * uint32_t(sizeof(*entries));
            auto* pad = reinterpret_cast<eng_arc_cmd_nop_t*>(md + 256 + used);
            pad->cmd_type = ECB_CMD_NOP;
            pad->padding = (256 - used) / 4 - 1;
        }
    }

    ~Impl() {
        if (pool_use) {
            pool_use->code -= max_code_bytes;
            pool_use->meta -= size_t(max_ecbs) * 512;
        }
        if (code_buf) scal_free_buffer(code_buf);
        if (meta_buf) scal_free_buffer(meta_buf);
    }

    static const MmeDesc& find_desc(const MmeTemplate& tmpl, uint32_t act, uint32_t eng) {
        for (const auto& ds : tmpl.descs)
            if (ds.act == act && ds.engine == eng) return ds;
        throw std::runtime_error("direct_mme: dyn MME missing (act, eng)");
    }

    // Pre-encoded QMAN register-write blob for words [8..181] transitioning (prev_t, prev_act) -> (cur_t, cur_act).
    // When prev_t == nullptr, emits full words [8..181] (including kSobWord); otherwise emits only differing spans.
    const DiffBlob& get_diff(const MmeTemplate* prev_t, uint8_t prev_act,
                             const MmeTemplate* cur_t, uint8_t cur_act, uint8_t eng, bool diff_on) {
        const DiffKey key{diff_on ? prev_t : nullptr, cur_t, diff_on ? prev_act : uint8_t(0), cur_act, eng};
        if (auto it = diff_cache.find(key); it != diff_cache.end()) return it->second;
        constexpr uint32_t kAddrWords = 8, kSobWord = (0x14c - 8) / 4;
        const auto& cur_w = find_desc(*cur_t, cur_act, eng).words;
        DiffBlob blob;
        if (!key.prev_t) {
            push_regs_bulk(blob.raw, uint16_t(0xb008 + 4 * kAddrWords), cur_w.data() + kAddrWords, 182 - kAddrWords, false);
        } else {
            const auto& prev_w = find_desc(*key.prev_t, key.prev_act, eng).words;
            auto need = [&](uint32_t i) { return i != kSobWord && cur_w[i] != prev_w[i]; };
            for (uint32_t i = kAddrWords; i < 182;) {
                if (!need(i)) { ++i; continue; }
                uint32_t j = i + 1;
                for (uint32_t k = j; k < 182 && k < j + 2; ++k)
                    if (need(k)) { j = k + 1; k = j - 1; }
                push_regs_bulk(blob.raw, uint16_t(0xb008 + 4 * i), cur_w.data() + i, j - i, false);
                i = j;
            }
        }
        const int64_t so = find_reg_payload(blob.raw, 0, uint32_t(0xb008 + 4 * kSobWord));
        blob.sob_byte_off = so >= 0 ? int32_t(so) : -1;
        return diff_cache.emplace(key, std::move(blob)).first->second;
    }
};

DirectDynMmeSet::DirectDynMmeSet(DirectRuntime& rt, uint32_t num_stages, uint32_t max_ecbs, size_t max_code_bytes)
    : impl_(std::make_unique<Impl>(rt, num_stages, max_ecbs, max_code_bytes)) {}
DirectDynMmeSet::~DirectDynMmeSet() = default;

void DirectDynMmeSet::reset() {
    auto& d = *impl_;
    d.rt.impl().ensure_pdma_buf(d.max_code_bytes);
    d.ecb_used = 0;
    d.code_used = 0;
    for (auto& st : d.stages) {
        st.active = false;
        st.has_wait = false;
        st.wait_target = 0;
        st.sig_credits = 0;
        st.ecbs.clear();
    }
}

uint32_t DirectDynMmeSet::set_stage(uint32_t stage_idx, std::span<const DirectMmeJob> jobs,
                                    bool has_wait, uint32_t wait_target, uint32_t fence_id, uint32_t sig_sob_abs) {
    auto& d = *impl_;
    if (stage_idx >= d.stages.size()) throw std::out_of_range("direct_mme: dyn stage index out of range");
    auto& st = d.stages[stage_idx];
    st.ecbs.clear();
    st.sig_credits = 0;
    if (jobs.empty()) {
        st.active = false;
        st.has_wait = false;
        st.wait_target = 0;
        return 0;
    }
    st.active = true;
    st.has_wait = has_wait;
    st.wait_target = wait_target;

    const bool diff_on = d.rt.impl().opt.mme_diff;
    static const bool do_val = std::getenv("G2E_DIRECT_DYN_VALIDATE") != nullptr;
    auto& s = d.rt.impl();
    char* const pdma_base = static_cast<char*>(s.pdma_host);
    const uint32_t sob_addr = uint32_t(mmDCORE2_SYNC_MNGR_OBJS_BASE + sig_sob_abs * 4);

    // Pre-built constant QMAN packet headers (zero per-job struct zeroing or vector resizing).
    packet_wreg_bulk addr_hdr{};
    addr_hdr.size64 = 4;  // 8 uint32 address words (0xb008..0xb024)
    addr_hdr.opcode = PACKET_WREG_BULK;
    addr_hdr.msg_barrier = 1;
    addr_hdr.reg_offset = 0xb008;
    packet_wreg32 exec_pkt{};
    exec_pkt.opcode = PACKET_WREG_32;
    exec_pkt.msg_barrier = 1;
    exec_pkt.reg_offset = 0xb004;
    exec_pkt.value = 0x3ff;
    packet_fence fence_pkt{};
    fence_pkt.opcode = PACKET_FENCE;
    fence_pkt.msg_barrier = 1;
    fence_pkt.id = fence_id;
    fence_pkt.target_val = 1;
    fence_pkt.dec_val = 1;
    packet_nop fin_pkt{};
    fin_pkt.opcode = PACKET_NOP;
    fin_pkt.msg_barrier = 1;
    fin_pkt.swtc = 1;

    size_t sz[2] = {0, 0};
    bool fence_added[2] = {false, false};
    const MmeTemplate* last_tmpl[2] = {nullptr, nullptr};
    uint8_t last_act[2] = {0, 0};
    // Small 1-entry hot cache for consecutive alternating (prev -> cur) blob lookups (e.g. gt512 act0 <-> act1).
    const MmeTemplate* cache_pt[2] = {nullptr, nullptr};
    const MmeTemplate* cache_ct[2] = {nullptr, nullptr};
    uint8_t cache_pa[2] = {0, 0}, cache_ca[2] = {0, 0};
    const Impl::DiffBlob* cache_b[2] = {nullptr, nullptr};

    auto flush_chunk = [&] {
        if (!sz[0] && !sz[1]) return;
        std::memcpy(d.prog_buf[0] + sz[0], &fin_pkt, 8); sz[0] += 8;
        std::memcpy(d.prog_buf[1] + sz[1], &fin_pkt, 8); sz[1] += 8;
        if (do_val) {
            std::vector<char> v0(d.prog_buf[0], d.prog_buf[0] + sz[0]);
            std::vector<char> v1(d.prog_buf[1], d.prog_buf[1] + sz[1]);
            if (!validate_static_program(v0, SCAL_MME_COMPUTE_GROUP) ||
                !validate_static_program(v1, SCAL_MME_COMPUTE_GROUP))
                throw std::runtime_error("direct_mme: dyn MME static QMAN program validation failed");
        }
        const uint32_t off0 = 0;
        const uint32_t off1 = uint32_t((sz[0] + 63) & ~size_t(63));
        const size_t chunk_bytes = (size_t(off1) + sz[1] + 255) & ~size_t(255);
        if (d.ecb_used >= d.max_ecbs || d.code_used + chunk_bytes > d.max_code_bytes)
            throw std::runtime_error("direct_mme: DirectDynMmeSet capacity exceeded (ecbs=" +
                                     std::to_string(d.ecb_used) + ", bytes=" + std::to_string(d.code_used + chunk_bytes) + ")");
        char* dst = pdma_base + d.code_used;
        std::memcpy(dst + off0, d.prog_buf[0], sz[0]);
        if (off1 > sz[0]) std::memset(dst + sz[0], 0, off1 - sz[0]);
        std::memcpy(dst + off1, d.prog_buf[1], sz[1]);
        if (chunk_bytes > off1 + sz[1]) std::memset(dst + off1 + sz[1], 0, chunk_bytes - (off1 + sz[1]));

        // Patch only variable size/offset fields in pre-initialized 512B ECB meta (zero memset!).
        char* md = static_cast<char*>(d.meta_host) + size_t(d.ecb_used) * 512;
        auto* entries = reinterpret_cast<eng_arc_cmd_static_desc_v2_t*>(md + 260);
        entries[0].size = uint32_t(sz[0]);
        entries[0].addr_offset = off0;
        entries[1].size = uint32_t(sz[1]);
        entries[1].addr_offset = off1;

        PreparedEcb ecb{};
        ecb.meta_core_addr = d.meta_core + uint64_t(d.ecb_used) * 512ull;
        ecb.code_dev_addr = d.code_dev + d.code_used;
        ecb.group = SCAL_MME_COMPUTE_GROUP;
        st.ecbs.push_back(std::move(ecb));

        ++d.ecb_used;
        d.code_used += chunk_bytes;
        sz[0] = sz[1] = 0;
        last_tmpl[0] = last_tmpl[1] = nullptr;
    };

    for (const auto& job : jobs) {
        const auto& tmpl = *job.tmpl;
        st.sig_credits += tmpl.signal_count;
        const uint64_t addr4[4] = {job.c1, job.c0, job.a, job.b};
        for (const auto& desc : tmpl.descs) {
            const uint32_t e = desc.engine;
            if (e == 0 && (sz[0] > 7300 || sz[1] > 7300)) flush_chunk();
            char* dst = d.prog_buf[e] + sz[e];
            if (has_wait && !fence_added[e]) {
                std::memcpy(dst, &fence_pkt, 8);
                dst += 8;
                fence_added[e] = true;
            }
            // Write 8 address words (40 bytes: 8B header + 32B payload)
            std::memcpy(dst, &addr_hdr, 8);
            std::memcpy(dst + 8, addr4, 32);
            dst += 40;
            const uint8_t da = uint8_t(desc.act);
            if (!(diff_on && last_tmpl[e] == job.tmpl && last_act[e] == da)) {
                const Impl::DiffBlob* blob;
                if (cache_pt[e] == last_tmpl[e] && cache_pa[e] == last_act[e] &&
                    cache_ct[e] == job.tmpl && cache_ca[e] == da && cache_b[e]) {
                    blob = cache_b[e];
                } else {
                    blob = &d.get_diff(last_tmpl[e], last_act[e], job.tmpl, da, uint8_t(e), diff_on);
                    cache_pt[e] = last_tmpl[e]; cache_pa[e] = last_act[e];
                    cache_ct[e] = job.tmpl;     cache_ca[e] = da;
                    cache_b[e] = blob;
                }
                const size_t bn = blob->raw.size();
                if (bn) {
                    std::memcpy(dst, blob->raw.data(), bn);
                    if (blob->sob_byte_off >= 0) std::memcpy(dst + blob->sob_byte_off, &sob_addr, 4);
                    dst += bn;
                }
                last_tmpl[e] = job.tmpl;
                last_act[e] = da;
            }
            std::memcpy(dst, &exec_pkt, 8);
            dst += 8;
            sz[e] = size_t(dst - d.prog_buf[e]);
        }
    }
    flush_chunk();
    return st.sig_credits;
}

void DirectDynMmeSet::upload() {
    auto& d = *impl_;
    if (!d.code_used) return;
    auto& s = d.rt.impl();
    s.begin();
    s.emit_pdma_1(SCAL_PDMA_TX_DATA_GROUP, PDMA_DIR_HOST2DEV, s.pdma_dev, d.code_dev, uint32_t(d.code_used), s.host_sob);
    s.submit();
    s.wait_fence(1, 500000);
}

uint32_t DirectDynMmeSet::stage_credits(uint32_t stage_idx) const {
    return impl_->stages.at(stage_idx).sig_credits;
}
size_t DirectDynMmeSet::last_code_bytes() const { return impl_->code_used; }
uint32_t DirectDynMmeSet::last_ecb_count() const { return impl_->ecb_used; }

struct DirectPlan::Impl {
    DirectRuntime& rt;
    struct PatchLoc {
        uint32_t stage = 0;
        uint32_t chunk = 0;
        uint32_t engine = 0;
        uint32_t off_lo = 0;
        uint32_t off_hi = 0;
    };
    struct SrfPatchLoc {
        uint32_t stage = 0;
        uint32_t chunk = 0;
        uint32_t engine = 0;
        uint32_t off = 0;
    };
    struct Stage {
        uint32_t group = 0;  // SCAL_MME_COMPUTE_GROUP or SCAL_TPC_COMPUTE_GROUP
        std::vector<std::vector<std::vector<char>>> chunks;  // one or more ECB chunks
        std::vector<std::vector<char>> cur;
        uint32_t cnt[2] = {0, 0};
        bool fence_added[2] = {false, false};
        int wait_sob = -1;
        uint32_t wait_target = 0;
        uint32_t fence_id = 3;
        int dyn_wait_slot = -1;
        const DirectDynMmeSet* dyn_set = nullptr;
        uint32_t dyn_stage_idx = 0;
        int opt_id = -1;
        int sig_sob_rel = -1;
        uint32_t sig_credits = 0;
        uint32_t engine_mask = 0;  // for MME: bit 0/1; for TPC: number of active TPCs
        uint32_t mon_first = 0;
        std::vector<PreparedEcb> ecbs;

        // the descriptor words last written by this chunk's program, per engine (diff encoding, see add_mme)
        std::vector<uint32_t> last[2];

        void flush_mme_chunk() {
            if (!cnt[0] && !cnt[1]) return;
            for (uint32_t e = 0; e < 2; ++e) push_finish(cur[e], false);
            chunks.push_back(std::move(cur));
            cur.assign(2, {});
            cnt[0] = cnt[1] = 0;
            last[0].clear();
            last[1].clear();
        }
    };
    std::vector<Stage> stages;
    std::map<int, std::vector<PatchLoc>> patches;
    std::map<int, std::vector<SrfPatchLoc>> srf_patches;
    uint32_t max_sob_used = 0;
    uint32_t host_credits = 0;
    // Split monitor pools so MME and TPC monitors NEVER collide across different plans (100% mon_cfg cache hits!):
    //   MME monitors: [0 .. 95]    (48 MME stages x 2 engines)
    //   TPC monitors: [96 .. 1023] (38 TPC stages x 24 engines)
    static constexpr uint32_t kMmeMonLimit = 96;
    uint32_t next_mme_mon = 0;
    uint32_t next_tpc_mon = kMmeMonLimit;
    // Lanes (set_lane): two plans that run concurrently (run_waves) use disjoint SOBs and monitors.
    //   lane 1: SOBs [0, 64),   MME monitors [0, 48),  TPC monitors [96, 744)
    //   lane 2: SOBs [64, 128), MME monitors [48, 96), TPC monitors [744, 1392)
    uint32_t mme_chunk_limit = 9;  // MME descriptors per engine in one ECB chunk
    bool mme_stage_break = false;
    int next_opt_id = -1;
    int next_dyn_wait = -1;
    uint64_t opt_mask = ~uint64_t(0);
    std::array<uint32_t, 16> dyn_wait_targets{};
    uint32_t lane = 0;
    uint32_t fence_id = 3;  // the engines' QMAN fence of this plan's waits (lane 2: fence 2, see set_lane)
    uint32_t sob_off = 0;
    uint32_t mme_mon_end = kMmeMonLimit;
    uint32_t tpc_mon_end = 0;  // 0: the pool's end
    scal_buffer_handle_t plan_meta = nullptr;
    scal_buffer_handle_t plan_code = nullptr;
    bool prepared = false;
    bool any_dirty = false;
    std::vector<PreparedEcb*> dirty_ecbs;

    explicit Impl(DirectRuntime& r, uint32_t /*init_mon*/) : rt(r) {}
    uint64_t code_bytes = 0, meta_bytes = 0;
    std::shared_ptr<DirectRuntime::Impl::PoolUse> pool_use;
    ~Impl() {
        if (pool_use) {
            pool_use->code -= code_bytes;
            pool_use->meta -= meta_bytes;
            --pool_use->plans;
        }
        if (plan_code) scal_free_buffer(plan_code);
        if (plan_meta) scal_free_buffer(plan_meta);
    }
};

DirectPlan::DirectPlan(DirectRuntime& rt, uint32_t init_mon) : impl_(std::make_unique<Impl>(rt, init_mon)) {}
DirectPlan::~DirectPlan() = default;

void DirectPlan::set_lane(uint32_t lane) {
    auto& p = *impl_;
    if (!p.stages.empty() || lane > 2) throw std::logic_error("direct_mme: set_lane before the first stage (0..2)");
    p.lane = lane;
    if (lane == 1) {
        p.sob_off = 0; p.next_mme_mon = 0; p.mme_mon_end = 48; p.next_tpc_mon = Impl::kMmeMonLimit; p.tpc_mon_end = 744;
    } else if (lane == 2) {
        p.sob_off = 64; p.next_mme_mon = 48; p.mme_mon_end = 96; p.next_tpc_mon = 744; p.tpc_mon_end = 1392;
        // A fence of its own: the waiting stages of an engine take from one counter per fence id, so with a shared
        // fence a signal of one lane's plan could release a stage of the other's.
        p.fence_id = 2;
    }
}

size_t DirectPlan::stage_count() const { return impl_->stages.size(); }

void DirectPlan::set_mme_chunk_limit(uint32_t n) {
    if (n < 1 || n > 9) throw std::invalid_argument("direct_mme: MME chunk limit 1..9");
    impl_->mme_chunk_limit = n;
}

void DirectPlan::end_mme_stage() { impl_->mme_stage_break = true; }

void DirectPlan::set_opt_id(int id) {
    if (id < -1 || id >= 64) throw std::invalid_argument("direct_mme: opt_id must be -1 or 0..63");
    if (impl_->next_opt_id != id) impl_->mme_stage_break = true;
    impl_->next_opt_id = id;
}

void DirectPlan::set_opt_mask(uint64_t mask) { impl_->opt_mask = mask; }

void DirectPlan::set_next_dyn_wait(int slot) {
    if (slot < -1 || slot >= 16) throw std::invalid_argument("direct_mme: dyn_wait slot must be -1 or 0..15");
    impl_->next_dyn_wait = slot;
}

void DirectPlan::set_dyn_wait(int slot, uint32_t target) {
    if (slot < 0 || slot >= 16) throw std::invalid_argument("direct_mme: dyn_wait slot out of range");
    impl_->dyn_wait_targets[size_t(slot)] = target;
}

void DirectPlan::add_dyn_mme(const DirectDynMmeSet* dyn_set, uint32_t dyn_stage_idx, int wait_sob, int sig_sob) {
    auto& p = *impl_;
    if (p.prepared) throw std::logic_error("direct_mme: plan already prepared");
    if (!dyn_set || dyn_stage_idx >= dyn_set->impl().stages.size())
        throw std::invalid_argument("direct_mme: bad dyn_set/stage_idx");
    if (sig_sob < 0) throw std::logic_error("direct_mme: dynamic MME stage must signal a private SOB (not host)");
    p.max_sob_used = std::max(p.max_sob_used, uint32_t(sig_sob) + 1);
    if (wait_sob >= 0) p.max_sob_used = std::max(p.max_sob_used, uint32_t(wait_sob) + 1);
    Impl::Stage st;
    st.group = SCAL_MME_COMPUTE_GROUP;
    st.wait_sob = wait_sob;
    st.wait_target = 0;
    st.dyn_set = dyn_set;
    st.dyn_stage_idx = dyn_stage_idx;
    st.sig_sob_rel = sig_sob;
    st.engine_mask = 3u;
    if (wait_sob >= 0) {
        st.mon_first = p.next_mme_mon;
        p.next_mme_mon += 2;
    }
    p.stages.push_back(std::move(st));
    p.mme_stage_break = true;
}

void DirectPlan::add_mme(const MmeTemplate& tmpl, uint64_t a, uint64_t b, uint64_t c0, uint64_t c1,
                         int wait_sob, uint32_t wait_target, int sig_sob, std::array<int, 4> patch_tags,
                         int override_fence) {
    auto& p = *impl_;
    if (p.prepared) throw std::logic_error("direct_mme: plan already prepared");
    if (p.next_opt_id >= 0 && wait_sob >= 0)
        throw std::logic_error("direct_mme: optional MME stage cannot carry wait_sob");
    const uint32_t st_fence = override_fence >= 0 ? uint32_t(override_fence) : p.fence_id;
    const uint32_t done_sob = p.rt.impl().chain_enabled ? p.rt.impl().chain_sob : p.rt.host_sob();
    const uint32_t sob_idx = sig_sob < 0 ? done_sob : (p.rt.sob_base() + p.sob_off + uint32_t(sig_sob));
    if (sig_sob < 0) p.host_credits += tmpl.signal_count;
    else p.max_sob_used = std::max(p.max_sob_used, uint32_t(sig_sob) + 1);
    if (wait_sob >= 0) p.max_sob_used = std::max(p.max_sob_used, uint32_t(wait_sob) + 1);

    const bool append_prev = (!p.mme_stage_break && wait_sob < 0 && !p.stages.empty() &&
                              p.stages.back().group == SCAL_MME_COMPUTE_GROUP &&
                              p.stages.back().dyn_set == nullptr &&
                              p.stages.back().engine_mask == tmpl.engine_mask &&
                              p.stages.back().opt_id == p.next_opt_id &&
                              (p.next_opt_id < 0 || p.stages.back().sig_sob_rel == sig_sob));
    p.mme_stage_break = false;
    if (!append_prev) {
        Impl::Stage st;
        st.group = SCAL_MME_COMPUTE_GROUP;
        st.cur.assign(2, {});
        st.opt_id = p.next_opt_id;
        st.sig_sob_rel = sig_sob;
        st.wait_sob = wait_sob;
        st.wait_target = wait_target;
        st.fence_id = st_fence;
        st.engine_mask = tmpl.engine_mask;
        if (wait_sob >= 0) {
            st.mon_first = p.next_mme_mon;
            p.next_mme_mon += 2;
        }
        p.stages.push_back(std::move(st));
    }
    const uint32_t stage_idx = uint32_t(p.stages.size() - 1);
    auto& st = p.stages.back();
    st.sig_credits += tmpl.signal_count;

    for (const auto& d : tmpl.descs) {
        if (d.engine > 1) throw std::runtime_error("direct_mme: invalid MME engine index");
        if (st.cnt[d.engine] >= p.mme_chunk_limit) st.flush_mme_chunk();
        if (st.wait_sob >= 0 && !st.fence_added[d.engine]) {
            push_fence(st.cur[d.engine], st.fence_id);
            st.fence_added[d.engine] = true;
        }
        const uint32_t chunk_idx = uint32_t(st.chunks.size());
        const size_t scan_from = st.cur[d.engine].size();
        auto w = d.words;
        auto set = [&](uint32_t off, uint32_t val) { w[(off - 8) / 4] = val; };
        set(0x08, uint32_t(c1)); set(0x0c, uint32_t(c1 >> 32));
        set(0x10, uint32_t(c0)); set(0x14, uint32_t(c0 >> 32));
        set(0x18, uint32_t(a));  set(0x1c, uint32_t(a >> 32));
        set(0x20, uint32_t(b));  set(0x24, uint32_t(b >> 32));
        set(0x14c, uint32_t(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob_idx * 4));
        // The descriptor registers keep their values between executions: after the first descriptor of a chunk's
        // program only the words that changed are written (Options::mme_diff), plus always the operand addresses
        // (the runtime patches them per descriptor) and the signal's SOB. A descriptor of the same template as the
        // one before then takes about 12 words instead of 182: the plans' code pool (about 116 MiB usable) holds the
        // prefill plans of both chunk slots and the decode plans. The execution time is the same.
        const bool diff_on = p.rt.impl().opt.mme_diff;
        auto& lastw = st.last[d.engine];
        if (!diff_on || lastw.size() != w.size()) {
            push_regs_bulk(st.cur[d.engine], 0xb008, w.data(), 182, false);
        } else {
            constexpr uint32_t kAddrWords = 8, kSobWord = (0x14c - 8) / 4;
            push_regs_bulk(st.cur[d.engine], 0xb008, w.data(), kAddrWords, false);
            auto need = [&](uint32_t i) { return w[i] != lastw[i] || i == kSobWord; };
            for (uint32_t i = kAddrWords; i < 182;) {
                if (!need(i)) { ++i; continue; }
                uint32_t j = i + 1;  // the run [i, j): up to the last changed word before two unchanged ones
                for (uint32_t k = j; k < 182 && k < j + 2; ++k)
                    if (need(k)) { j = k + 1; k = j - 1; }
                push_regs_bulk(st.cur[d.engine], uint16_t(0xb008 + 4 * i), w.data() + i, j - i, false);
                i = j;
            }
        }
        lastw.assign(w.begin(), w.end());
        auto rec_mme = [&](int tag, uint32_t reg_lo) {
            if (tag < 0) return;
            const int64_t lo = find_reg_payload(st.cur[d.engine], scan_from, reg_lo);
            const int64_t hi = find_reg_payload(st.cur[d.engine], scan_from, reg_lo + 4);
            if (lo < 0 || hi < 0) throw std::runtime_error("direct_mme: MME patch reg not found");
            p.patches[tag].push_back({stage_idx, chunk_idx, d.engine, uint32_t(lo), uint32_t(hi)});
        };
        rec_mme(patch_tags[0], 0xb018);
        rec_mme(patch_tags[1], 0xb020);
        rec_mme(patch_tags[2], 0xb010);
        rec_mme(patch_tags[3], 0xb008);
        push_reg(st.cur[d.engine], 0xb004, 0x3ff, false);
        ++st.cnt[d.engine];
    }
}

void DirectPlan::add_mme_pair(const MmeTemplate& tmpl,
                              uint64_t a0, uint64_t b0, uint64_t c0,
                              uint64_t a1, uint64_t b1, uint64_t c1,
                              int wait_sob, uint32_t wait_target, int sig_sob,
                              std::array<int, 4> patch0, std::array<int, 4> patch1) {
    auto& p = *impl_;
    if (p.prepared) throw std::logic_error("direct_mme: plan already prepared");
    uint32_t e0_signals = 0;
    for (const auto& d : tmpl.descs)
        if (d.engine == 0 && d.signals()) ++e0_signals;
    const uint32_t pair_signals = e0_signals * 2u;
    const uint32_t done_sob = p.rt.impl().chain_enabled ? p.rt.impl().chain_sob : p.rt.host_sob();
    const uint32_t sob_idx = sig_sob < 0 ? done_sob : (p.rt.sob_base() + p.sob_off + uint32_t(sig_sob));
    if (sig_sob < 0) p.host_credits += pair_signals;
    else p.max_sob_used = std::max(p.max_sob_used, uint32_t(sig_sob) + 1);
    if (wait_sob >= 0) p.max_sob_used = std::max(p.max_sob_used, uint32_t(wait_sob) + 1);

    constexpr uint32_t kBothMask = 3u;
    const bool append_prev = (wait_sob < 0 && !p.stages.empty() &&
                              p.stages.back().group == SCAL_MME_COMPUTE_GROUP &&
                              p.stages.back().engine_mask == kBothMask);
    if (!append_prev) {
        Impl::Stage st;
        st.group = SCAL_MME_COMPUTE_GROUP;
        st.cur.assign(2, {});
        st.wait_sob = wait_sob;
        st.wait_target = wait_target;
        st.fence_id = p.fence_id;
        st.engine_mask = kBothMask;
        if (wait_sob >= 0) {
            st.mon_first = p.next_mme_mon;
            p.next_mme_mon += 2;
        }
        p.stages.push_back(std::move(st));
    }
    const uint32_t stage_idx = uint32_t(p.stages.size() - 1);
    auto& st = p.stages.back();
    st.sig_credits += tmpl.signal_count;

    for (const auto& d : tmpl.descs) {
        if (d.engine != 0) continue;
        if (st.cnt[0] >= 9 || st.cnt[1] >= 9) st.flush_mme_chunk();
        auto emit_one = [&](uint32_t eng, uint64_t a, uint64_t b, uint64_t c, const std::array<int, 4>& pt) {
            if (st.wait_sob >= 0 && !st.fence_added[eng]) {
                push_fence(st.cur[eng], p.fence_id);
                st.fence_added[eng] = true;
            }
            const uint32_t chunk_idx = uint32_t(st.chunks.size());
            const size_t scan_from = st.cur[eng].size();
            auto w = d.words;
            auto set = [&](uint32_t off, uint32_t val) { w[(off - 8) / 4] = val; };
            set(0x08, 0); set(0x0c, 0);
            set(0x10, uint32_t(c)); set(0x14, uint32_t(c >> 32));
            set(0x18, uint32_t(a)); set(0x1c, uint32_t(a >> 32));
            set(0x20, uint32_t(b)); set(0x24, uint32_t(b >> 32));
            set(0x14c, uint32_t(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob_idx * 4));
            push_regs_bulk(st.cur[eng], 0xb008, w.data(), 182, false);
            st.last[eng].clear();  // not tracked here: the next add_mme of this chunk writes its descriptor whole
            auto rec_mme = [&](int tag, uint32_t reg_lo) {
                if (tag < 0) return;
                const int64_t lo = find_reg_payload(st.cur[eng], scan_from, reg_lo);
                const int64_t hi = find_reg_payload(st.cur[eng], scan_from, reg_lo + 4);
                if (lo < 0 || hi < 0) throw std::runtime_error("direct_mme: MME pair patch reg not found");
                p.patches[tag].push_back({stage_idx, chunk_idx, eng, uint32_t(lo), uint32_t(hi)});
            };
            rec_mme(pt[0], 0xb018);
            rec_mme(pt[1], 0xb020);
            rec_mme(pt[2], 0xb010);
            rec_mme(pt[3], 0xb008);
            push_reg(st.cur[eng], 0xb004, 0x3ff, false);
            ++st.cnt[eng];
        };
        emit_one(0, a0, b0, c0, patch0);
        emit_one(1, a1, b1, c1, patch1);
    }
}

void DirectPlan::add_tpc_grid(uint64_t isa, std::span<const DirectTpcTensor> tensors, std::span<const uint32_t> srfs,
                              std::array<uint32_t, 5> grid_sizes, uint32_t split_dim, uint32_t tpcs,
                              int wait_sob, uint32_t wait_target, int sig_sob, uint32_t tpc_cmd,
                              std::span<const int> tensor_patch_tags, uint32_t split_base,
                              std::span<const int> srf_patch_tags) {
    auto& p = *impl_;
    if (p.prepared) throw std::logic_error("direct_mme: plan already prepared");
    if (tensors.empty() || tensors.size() > 16 || srfs.size() > 32 || split_dim >= 5 || !grid_sizes[split_dim] ||
        !tpcs || tpcs > 24 || (!tensor_patch_tags.empty() && tensor_patch_tags.size() != tensors.size()) ||
        (!srf_patch_tags.empty() && srf_patch_tags.size() != srfs.size()))
        throw std::invalid_argument("direct_mme: bad TPC parameters");
    const uint32_t total_work = grid_sizes[split_dim];
    const uint32_t active_tpcs = std::min(tpcs, total_work);
    const uint32_t done_sob = p.rt.impl().chain_enabled ? p.rt.impl().chain_sob : p.rt.host_sob();
    const uint32_t sob_idx = sig_sob < 0 ? done_sob : (p.rt.sob_base() + p.sob_off + uint32_t(sig_sob));
    if (sig_sob < 0) p.host_credits += active_tpcs;
    else p.max_sob_used = std::max(p.max_sob_used, uint32_t(sig_sob) + 1);
    if (wait_sob >= 0) p.max_sob_used = std::max(p.max_sob_used, uint32_t(wait_sob) + 1);

    const uint32_t stage_idx = uint32_t(p.stages.size());
    Impl::Stage st;
    st.group = SCAL_TPC_COMPUTE_GROUP;
    st.wait_sob = wait_sob;
    st.wait_target = wait_target;
    st.dyn_wait_slot = p.next_dyn_wait;
    p.next_dyn_wait = -1;
    st.opt_id = p.next_opt_id;
    st.sig_sob_rel = sig_sob;
    st.sig_credits = active_tpcs;
    st.engine_mask = active_tpcs;
    if (wait_sob >= 0) {
        st.mon_first = p.next_tpc_mon;
        p.next_tpc_mon += 24;
    }

    const uint32_t t0_reg = uint16_t(0xb000 + offsetof(gaudi2::block_tpc, qm_tensor_0));
    std::vector<std::vector<char>> progs(24);
    for (uint32_t i = 0; i < 24; ++i) {
        if (i < active_tpcs) {
            const uint32_t first = total_work * i / active_tpcs;
            const uint32_t work = total_work * (i + 1) / active_tpcs - first;
            if (wait_sob >= 0) push_fence(progs[i], p.fence_id);
            gaudi2::Gaudi2TpcDesc desc{};
            for (size_t ti = 0; ti < tensors.size(); ++ti) {
                const auto& in = tensors[ti];
                auto& t = desc.m_tensors[ti];
                t.base_addr_low._raw = uint32_t(in.addr);
                t.base_addr_high._raw = uint32_t(in.addr >> 32);
                t.padding_value._raw = in.padding;
                t.tensor_config.data_type = tpc_hw_dtype(in.dtype);
                t.tensor_config.valid_dim_mask = 0x1f;
                t.tensor_config.last_dim = in.dims > 0 ? (in.dims - 1) : 0;
                uint32_t st0 = in.strides[0] ? in.strides[0] : 1u;
                uint32_t st1 = in.strides[1] ? in.strides[1] : st0 * in.sizes[0];
                uint32_t st2 = in.strides[2] ? in.strides[2] : st1 * in.sizes[1];
                uint32_t st3 = in.strides[3] ? in.strides[3] : st2 * in.sizes[2];
                uint32_t st4 = in.strides[4] ? in.strides[4] : st3 * in.sizes[3];
                t.dim_0_size.v = in.sizes[0]; t.dim_0_stride.v = st0;
                t.dim_1_size.v = in.sizes[1]; t.dim_1_stride.v = st1;
                t.dim_2_size.v = in.sizes[2]; t.dim_2_stride.v = st2;
                t.dim_3_size.v = in.sizes[3]; t.dim_3_stride.v = st3;
                t.dim_4_size.v = in.sizes[4]; t.dim_4_stride.v = st4;
            }
            desc.m_so.message._raw = 0x80000001u;
            desc.m_so.addr.v = uint32_t(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob_idx * 4);
            auto& d = desc.m_desc;
            d.kernel_base_address_low._raw = uint32_t(isa);
            d.kernel_base_address_high._raw = uint32_t(isa >> 32);
            uint32_t base[5] = {0, 0, 0, 0, 0};
            uint32_t sz[5] = {grid_sizes[0], grid_sizes[1], grid_sizes[2], grid_sizes[3], grid_sizes[4]};
            base[split_dim] = split_base + first;
            sz[split_dim] = work;
            d.tid_base_dim_0.v = base[0]; d.tid_size_dim_0.v = sz[0];
            d.tid_base_dim_1.v = base[1]; d.tid_size_dim_1.v = sz[1];
            d.tid_base_dim_2.v = base[2]; d.tid_size_dim_2.v = sz[2];
            d.tid_base_dim_3.v = base[3]; d.tid_size_dim_3.v = sz[3];
            d.tid_base_dim_4.v = base[4]; d.tid_size_dim_4.v = sz[4];
            for (size_t k = 0; k < srfs.size(); ++k) d.srf[k].v = srfs[k];
            d.kernel_config.aso_evict_l0 = 1;
            d.kernel_config.num_valid_srfs = 32;
            d.kernel_config.rd_rate_limit_rst_token = 8;
            d.kernel_config.wr_rate_limit_rst_token = 6;
            d.kernel_config.irf_32bit_compatibility = 1;
            const auto* t_words = reinterpret_cast<const uint32_t*>(&desc.m_tensors[0]);
            const uint32_t t_cnt = uint32_t(tensors.size() * sizeof(desc.m_tensors[0]) / 4);
            push_regs_bulk(progs[i], t0_reg, t_words, t_cnt, true);
            const auto* tail_words = reinterpret_cast<const uint32_t*>(&desc.m_so);
            const uint32_t tail_off = uint32_t(offsetof(gaudi2::block_tpc, qm_tensor_0) + offsetof(gaudi2::Gaudi2TpcDesc, m_so));
            const uint32_t tail_cnt = uint32_t((sizeof(desc.m_so) + sizeof(desc.m_desc)) / 4);
            const size_t tail_scan = progs[i].size();
            push_regs_bulk(progs[i], uint16_t(0xb000 + tail_off), tail_words, tail_cnt, false);
            if (i == 0 && !srf_patch_tags.empty()) {
                const uint32_t srf0_reg = uint32_t(0xb000 + offsetof(gaudi2::block_tpc, qm_tensor_0) +
                                                   offsetof(gaudi2::Gaudi2TpcDesc, m_desc) +
                                                   offsetof(gaudi2::block_tpc_non_tensor_descriptor, srf));
                for (size_t k = 0; k < srfs.size(); ++k) {
                    if (srf_patch_tags[k] >= 0) {
                        const int64_t off = find_reg_payload(progs[i], tail_scan, srf0_reg + uint32_t(k * 4));
                        if (off < 0) throw std::runtime_error("direct_mme: TPC SRF patch reg not found");
                        for (uint32_t eng = 0; eng < active_tpcs; ++eng)
                            p.srf_patches[srf_patch_tags[k]].push_back({stage_idx, 0, eng, uint32_t(off)});
                    }
                }
            }
            push_reg(progs[i], uint16_t(0xb000 + offsetof(gaudi2::block_tpc, tpc_cmd)), tpc_cmd, true);
            push_reg(progs[i], uint16_t(0xb000 + offsetof(gaudi2::block_tpc, tpc_execute)), 1, false);
        } else {
            if (wait_sob >= 0) push_fence(progs[i], p.fence_id);
            push_reg(progs[i], uint16_t(0xb000 + offsetof(gaudi2::block_tpc, tpc_cmd)), tpc_cmd, true);
        }
        push_finish(progs[i], true);
    }
    if (!tensor_patch_tags.empty()) {
        for (size_t ti = 0; ti < tensors.size(); ++ti) {
            if (tensor_patch_tags[ti] >= 0) {
                const uint32_t reg_lo = t0_reg + uint32_t(ti * sizeof(gaudi2::block_tpc_tensor));
                const int64_t lo = find_reg_payload(progs[0], 0, reg_lo);
                const int64_t hi = find_reg_payload(progs[0], 0, reg_lo + 4);
                if (lo < 0 || hi < 0) throw std::runtime_error("direct_mme: TPC patch reg not found");
                for (uint32_t eng = 0; eng < active_tpcs; ++eng)
                    p.patches[tensor_patch_tags[ti]].push_back({stage_idx, 0, eng, uint32_t(lo), uint32_t(hi)});
            }
        }
    }
    st.chunks.push_back(std::move(progs));
    p.stages.push_back(std::move(st));
}

void DirectPlan::add_tpc(uint64_t isa, std::span<const DirectTpcTensor> tensors, std::span<const uint32_t> srfs,
                         uint32_t rows, uint32_t tpcs, int wait_sob, uint32_t wait_target, int sig_sob,
                         uint32_t tpc_cmd, std::span<const int> tensor_patch_tags,
                         std::span<const int> srf_patch_tags) {
    add_tpc_grid(isa, tensors, srfs, {1, rows, 1, 1, 1}, 1, tpcs, wait_sob, wait_target, sig_sob, tpc_cmd,
                 tensor_patch_tags, 0, srf_patch_tags);
}

void DirectPlan::patch_addr(int tag, uint64_t addr) {
    auto& p = *impl_;
    if (!p.prepared) throw std::logic_error("direct_mme: patch_addr before prepare()");
    auto it = p.patches.find(tag);
    if (it == p.patches.end()) return;
    const uint32_t lo = uint32_t(addr);
    const uint32_t hi = uint32_t(addr >> 32);
    for (const auto& loc : it->second) {
        auto& ecb = p.stages[loc.stage].ecbs[loc.chunk];
        const size_t base_off = size_t(ecb.engine_offsets[loc.engine]);
        std::memcpy(ecb.shadow.data() + base_off + loc.off_lo, &lo, 4);
        std::memcpy(ecb.shadow.data() + base_off + loc.off_hi, &hi, 4);
        if (!ecb.dirty) {
            ecb.dirty = true;
            p.dirty_ecbs.push_back(&ecb);
        }
    }
    p.any_dirty = true;
}

void DirectPlan::patch_srf(int tag, uint32_t val) {
    auto& p = *impl_;
    if (!p.prepared) throw std::logic_error("direct_mme: patch_srf before prepare()");
    auto it = p.srf_patches.find(tag);
    if (it == p.srf_patches.end()) return;
    for (const auto& loc : it->second) {
        auto& ecb = p.stages[loc.stage].ecbs[loc.chunk];
        const size_t base_off = size_t(ecb.engine_offsets[loc.engine]);
        std::memcpy(ecb.shadow.data() + base_off + loc.off, &val, 4);
        if (!ecb.dirty) {
            ecb.dirty = true;
            p.dirty_ecbs.push_back(&ecb);
        }
    }
    p.any_dirty = true;
}

void DirectPlan::prepare() {
    auto& p = *impl_;
    if (p.prepared) return;
    auto& s = p.rt.impl();
    const uint32_t max_sobs = s.chain_enabled ? (s.sob_count - 1u) : s.sob_count;
    const uint32_t max_mons = s.chain_enabled ? (s.mon_count - 4u) : s.mon_count;
    if (p.sob_off + p.max_sob_used > max_sobs)
        throw std::runtime_error("direct_mme: plan exceeds private SOB pool (" + std::to_string(p.sob_off + p.max_sob_used) +
                                 " of " + std::to_string(max_sobs) + ")");
    if (p.lane && p.max_sob_used > 64)
        throw std::runtime_error("direct_mme: plan exceeds its lane's SOBs (" + std::to_string(p.max_sob_used) + ")");
    if (p.next_mme_mon > p.mme_mon_end || p.next_tpc_mon > (p.tpc_mon_end ? std::min(p.tpc_mon_end, max_mons) : max_mons))
        throw std::runtime_error("direct_mme: plan exceeds private monitor pool (mme=" +
                                 std::to_string(p.next_mme_mon) + ", tpc=" + std::to_string(p.next_tpc_mon) + ")");
    if (p.host_credits == 0)
        throw std::runtime_error("direct_mme: plan has no stage signaling host SOB");
    // Single-allocation plan preparation: 1 host_shared meta buffer + 1 hbm_shared code buffer + 1 DMA upload/readback!
    size_t num_ecbs = 0;
    size_t total_code = 0;
    for (auto& st : p.stages) {
        if (st.group == SCAL_MME_COMPUTE_GROUP) st.flush_mme_chunk();
        st.ecbs.reserve(st.chunks.size());
        for (const auto& ch : st.chunks) {
            PreparedEcb ecb{};
            ecb.group = st.group;
            DirectRuntime::Impl::build_ecb_bytes(st.group, ch, ecb.shadow, ecb.engine_offsets);
            total_code += ecb.shadow.size();
            ++num_ecbs;
            st.ecbs.push_back(std::move(ecb));
        }
    }
    const int rc_h = scal_allocate_aligned_buffer(s.host_pool, num_ecbs * 512, 256, &p.plan_meta);
    const int rc_d = rc_h ? 0 : scal_allocate_aligned_buffer(s.hbm_pool, total_code, 256, &p.plan_code);
    if (rc_h || rc_d) {
        throw std::runtime_error("direct_mme: plan alloc failed (rc_host=" + std::to_string(rc_h) +
                                 ", rc_hbm=" + std::to_string(rc_d) + ", ecbs=" + std::to_string(num_ecbs) +
                                 ", bytes=" + std::to_string(total_code) + "; " + std::to_string(s.pool_use->plans) + " plans hold " +
                                 std::to_string(s.pool_use->code >> 20) + " MiB of code)");
    }
    p.code_bytes = total_code;
    p.meta_bytes = num_ecbs * 512ull;
    p.pool_use = s.pool_use;
    s.pool_use->code += total_code;
    s.pool_use->meta += p.meta_bytes;
    ++s.pool_use->plans;
    if (static const bool dbg = std::getenv("G2E_DIRECT_POOL") != nullptr; dbg)
        std::fprintf(stderr, "[direct] plan %u: %zu ECBs, %zu KiB code; pools: %llu MiB code, %llu MiB meta\n", s.pool_use->plans, num_ecbs,
                     total_code >> 10, (unsigned long long)(s.pool_use->code >> 20), (unsigned long long)(s.pool_use->meta >> 20));
    scal_buffer_info_t mi{}, ci{};
    if (scal_buffer_get_info(p.plan_meta, &mi) || scal_buffer_get_info(p.plan_code, &ci) ||
        !mi.host_address || !mi.core_address || (mi.core_address % 256) != 0)
        throw std::runtime_error("direct_mme: plan buffer info failed");

    std::vector<char> all_code(total_code, 0);
    size_t ecb_idx = 0, code_off = 0;
    // Pack MME ECBs contiguously first, then TPC ECBs: allows flush_patches_and_upload to coalesce
    // all dirty MME ECBs of a plan into ONE single synMemCopyAsync (10-30x fewer PDMA calls)!
    for (int pass = 0; pass < 2; ++pass) {
        const uint32_t want_group = (pass == 0) ? SCAL_MME_COMPUTE_GROUP : SCAL_TPC_COMPUTE_GROUP;
        for (auto& st : p.stages) {
            if (st.group != want_group) continue;
            for (size_t k = 0; k < st.ecbs.size(); ++k) {
                auto& ecb = st.ecbs[k];
                ecb.meta_core_addr = mi.core_address + ecb_idx * 512ull;
                ecb.code_dev_addr = ci.device_address + code_off;
                DirectRuntime::Impl::fill_ecb_meta(static_cast<char*>(mi.host_address) + ecb_idx * 512ull,
                                                   st.chunks[k], ecb.engine_offsets);
                std::memcpy(all_code.data() + code_off, ecb.shadow.data(), ecb.shadow.size());
                code_off += ecb.shadow.size();
                ++ecb_idx;
            }
            st.chunks.clear();
        }
    }
    s.upload_and_verify(all_code, ci.device_address);
    p.prepared = true;

    // Pre-configure all monitor static payload/config registers in one lightweight LBW pass (keeps run() tiny!).
    // We configure without arming by resetting SOBs to 0 and letting run() arm only on demand; wait:
    // if total initial monitor config fits in 64 KiB (up to ~35 TPC stages), first run() configures automatically!
    // For larger plans, pre-configure monitor payloads in batches of <= 300 monitors here.
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    uint32_t batch_lbws = 0;
    s.begin();
    auto flush_cfg = [&] {
        if (!batch_lbws) return;
        s.lbw(base + s.host_sob * 4, 0x80000001u);
        s.submit();
        s.wait_fence(1);
        s.begin();
        batch_lbws = 0;
    };
    for (const auto& st : p.stages) {
        if (st.wait_sob < 0) continue;
        const uint32_t sob = s.sob_base + p.sob_off + uint32_t(st.wait_sob);
        if (st.group == SCAL_MME_COMPUTE_GROUP) {
            uint32_t m = s.mon_base + st.mon_first;
            auto cfg_one = [&](uint64_t dst) {
                if (batch_lbws > 2500) flush_cfg();
                const uint32_t sl = m % 2048;
                const uint64_t key = dst ^ (uint64_t((sob / 8) >> 8) << 1) ^ 1ull;
                if (s.mon_cfg[sl] != key || s.mon_wr[sl] != 0) {
                    packet_msg_short f{};
                    f.mon_config_register.msb_sid = (sob / 8) >> 8;
                    s.lbw(base + 0x10000 + m * 4, f.value);
                    s.lbw(base + 0x8000 + m * 4, uint32_t(dst));
                    s.lbw(base + 0xa000 + m * 4, uint32_t(dst >> 32));
                    s.lbw(base + 0xc000 + m * 4, 1);
                    s.mon_cfg[sl] = key;
                    s.mon_wr[sl] = 0;
                    batch_lbws += 4;
                }
                ++m;
            };
            if (st.engine_mask & 1u) cfg_one(fence_reg(kMme0FenceAddr, st.fence_id));
            if (st.engine_mask & 2u) cfg_one(fence_reg(kMme2FenceAddr, st.fence_id));
        } else {
            for (uint32_t i = 0; i < 24; i += 4) {
                if (batch_lbws > 2500) flush_cfg();
                const uint32_t cnt = 4u;
                for (uint32_t j = 0; j < cnt; ++j) {
                    const uint32_t m = s.mon_base + st.mon_first + i + j, sl = m % 2048;
                    const uint64_t dst = tpc_fence_addr(i + j, p.fence_id);
                    const uint8_t wr = j ? 0 : uint8_t(cnt - 1);
                    const uint64_t key = dst ^ (uint64_t((sob / 8) >> 8) << 1) ^ 1ull;
                    if (s.mon_cfg[sl] == key && s.mon_wr[sl] == wr) continue;
                    packet_msg_short f{};
                    f.mon_config_register.msb_sid = (sob / 8) >> 8;
                    f.mon_config_register.wr_num = wr;
                    s.lbw(base + 0x10000 + m * 4, f.value);
                    s.lbw(base + 0x8000 + m * 4, uint32_t(dst));
                    s.lbw(base + 0xa000 + m * 4, uint32_t(dst >> 32));
                    s.lbw(base + 0xc000 + m * 4, 1);
                    s.mon_cfg[sl] = key;
                    s.mon_wr[sl] = wr;
                    batch_lbws += 4;
                }
            }
        }
    }
    flush_cfg();
}

static void emit_sob_resets(DirectRuntime::Impl& s, const DirectPlan::Impl& p) {
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    for (uint32_t i = 0; i < p.max_sob_used; ++i) s.lbw(base + (s.sob_base + p.sob_off + i) * 4, 0);
}

static uint32_t compute_skip_sig(const DirectPlan::Impl& p, std::array<uint32_t, 64>& skip_sig) {
    skip_sig.fill(0);
    uint32_t skip_host = 0;
    if (p.opt_mask == ~uint64_t(0)) return 0;
    for (const auto& st : p.stages) {
        if (st.opt_id < 0 || ((p.opt_mask >> st.opt_id) & 1ull)) continue;
        if (st.sig_sob_rel < 0) skip_host += st.sig_credits;
        else if (st.sig_sob_rel < 64) skip_sig[size_t(st.sig_sob_rel)] += st.sig_credits;
    }
    return skip_host;
}

static void emit_stage(DirectRuntime::Impl& s, const DirectPlan::Impl& p, const DirectPlan::Impl::Stage& st,
                       const std::array<uint32_t, 64>& skip_sig) {
    if (st.dyn_set != nullptr) {
        const auto& ds = st.dyn_set->impl().stages[st.dyn_stage_idx];
        if (!ds.active) return;
        if (ds.has_wait && st.wait_sob >= 0) {
            const uint32_t sob = s.sob_base + p.sob_off + uint32_t(st.wait_sob);
            uint32_t m = s.mon_base + st.mon_first;
            s.arm_monitor(m++, sob, ds.wait_target, fence_reg(kMme0FenceAddr, p.fence_id));
            s.arm_monitor(m++, sob, ds.wait_target, fence_reg(kMme2FenceAddr, p.fence_id));
        }
        for (const auto& ecb : ds.ecbs) s.queue_ecb(ecb);
        return;
    }
    if (st.opt_id >= 0 && !((p.opt_mask >> st.opt_id) & 1ull)) return;
    if (st.wait_sob >= 0) {
        const uint32_t sob = s.sob_base + p.sob_off + uint32_t(st.wait_sob);
        const uint32_t target = (st.dyn_wait_slot >= 0)
                                    ? p.dyn_wait_targets[size_t(st.dyn_wait_slot)]
                                    : (st.wait_target - (st.wait_sob < 64 ? skip_sig[size_t(st.wait_sob)] : 0u));
        if (st.group == SCAL_MME_COMPUTE_GROUP) {
            uint32_t m = s.mon_base + st.mon_first;
            if (st.engine_mask & 1u) s.arm_monitor(m++, sob, target, fence_reg(kMme0FenceAddr, st.fence_id));
            if (st.engine_mask & 2u) s.arm_monitor(m++, sob, target, fence_reg(kMme2FenceAddr, st.fence_id));
        } else {
            for (uint32_t i = 0; i < 24; i += 4) {
                uint64_t dsts[4];
                for (uint32_t j = 0; j < 4; ++j) dsts[j] = tpc_fence_addr(i + j, p.fence_id);
                s.arm_monitor_multi(s.mon_base + st.mon_first + i, 4, sob, target, dsts);
            }
        }
    }
    for (const auto& ecb : st.ecbs) s.queue_ecb(ecb);
}

static void emit_plan_stages(DirectRuntime::Impl& s, const DirectPlan::Impl& p, const std::array<uint32_t, 64>& skip_sig) {
    emit_sob_resets(s, p);
    for (const auto& st : p.stages) emit_stage(s, p, st, skip_sig);
}

static uint32_t est_plan_cmd_bytes(const DirectPlan::Impl& p) {
    uint32_t bytes = (p.max_sob_used + 2u) * 16u + 128u;
    for (const auto& st : p.stages) {
        if (st.dyn_set != nullptr) {
            const auto& ds = st.dyn_set->impl().stages[st.dyn_stage_idx];
            if (!ds.active) continue;
            if (ds.has_wait && st.wait_sob >= 0) bytes += 48u;
            bytes += uint32_t(ds.ecbs.size()) * 64u;
            continue;
        }
        if (st.opt_id >= 0 && !((p.opt_mask >> st.opt_id) & 1ull)) continue;
        if (st.wait_sob >= 0) bytes += (st.group == SCAL_MME_COMPUTE_GROUP ? 48u : 128u);
        bytes += uint32_t(st.ecbs.size()) * 64u;
    }
    return bytes;
}

double DirectPlan::run() {
    auto& p = *impl_;
    if (!p.prepared) prepare();
    auto& s = p.rt.impl();
    if (p.any_dirty) {
        DirectPlan* self = this;
        p.rt.flush_patches(std::span<DirectPlan* const>(&self, 1));
    }
    std::array<uint32_t, 64> skip_sig;
    const uint32_t eff_host = p.host_credits - compute_skip_sig(p, skip_sig);
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    s.begin();
    if (s.chain_enabled) {
        s.lbw(base + s.chain_sob * 4, 0);
        s.arm_only(s.chain_mon_host, s.chain_sob, eff_host);
    }
    emit_plan_stages(s, p, skip_sig);
    const double t0 = now_us();
    s.submit();
    s.wait_fence(s.chain_enabled ? 1u : eff_host);
    return now_us() - t0;
}

double DirectRuntime::run_chain(std::span<DirectPlan* const> plans) {
    auto& s = *impl_;
    if (plans.empty()) return 0.0;
    if (!s.chain_enabled || plans.size() == 1) {
        double tot = 0.0;
        for (DirectPlan* pl : plans)
            if (pl) tot += pl->run();
        return tot;
    }
    bool any_dirty = false;
    for (DirectPlan* pl : plans) {
        if (!pl) continue;
        if (!pl->impl().prepared) pl->prepare();
        if (pl->impl().any_dirty) any_dirty = true;
    }
    if (any_dirty) flush_patches(plans);

    // Filter non-null plans for clean final-element indexing.
    std::vector<DirectPlan*> active;
    active.reserve(plans.size());
    for (DirectPlan* pl : plans)
        if (pl) active.push_back(pl);
    if (active.empty()) return 0.0;

    // Pipelined 20 KiB chunks inside the 64 KiB ring (max 2 chunks = <= 40 KiB in flight -> >= 24 KiB safety cushion).
    constexpr uint32_t kChunkBudget = 20480u;
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    const double t0 = now_us();
    uint32_t in_flight = 0;
    s.begin();
    uint32_t chunk_start_pi = s.pi;

    for (size_t i = 0; i < active.size(); ++i) {
        auto& p = active[i]->impl();
        const bool is_final = (i + 1 == active.size());
        const uint32_t cur_chunk = s.pi - chunk_start_pi;
        const uint32_t need_this = est_plan_cmd_bytes(p);
        const uint32_t need_next = is_final ? 0u : est_plan_cmd_bytes(active[i + 1]->impl());
        const bool is_chunk_end = !is_final && (cur_chunk + need_this + need_next > kChunkBudget);

        std::array<uint32_t, 64> skip_sig;
        const uint32_t eff_host = p.host_credits - compute_skip_sig(p, skip_sig);
        s.lbw(base + s.chain_sob * 4, 0);
        const uint32_t cmon = is_final ? s.chain_mon_host : (is_chunk_end ? s.chain_mon_both : s.chain_mon_fence);
        s.arm_only(cmon, s.chain_sob, eff_host);
        emit_plan_stages(s, p, skip_sig);
        if (!is_final) s.fence_wait(Impl::kChainFenceId, 1);

        if (is_final || is_chunk_end) {
            s.submit();
            ++in_flight;
            s.begin();
            chunk_start_pi = s.pi;
            if (!is_final && in_flight >= 2) {
                s.wait_fence(1);
                --in_flight;
            }
        }
    }
    if (in_flight > 0) s.wait_fence(in_flight);
    return now_us() - t0;
}

uint32_t DirectRuntime::submit_chain(std::span<DirectPlan* const> plans) {
    auto& s = *impl_;
    if (plans.empty()) return 0u;
    if (!s.chain_enabled) {
        run_chain(plans);
        return 0u;
    }
    std::vector<DirectPlan*> active;
    active.reserve(plans.size());
    for (DirectPlan* pl : plans) {
        if (!pl) continue;
        if (!pl->impl().prepared) pl->prepare();
        active.push_back(pl);
    }
    if (active.empty()) return 0u;

    constexpr uint32_t kChunkBudget = 20480u;
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    uint32_t in_flight = 0;
    s.begin();
    uint32_t chunk_start_pi = s.pi;
    for (size_t i = 0; i < active.size(); ++i) {
        auto& p = active[i]->impl();
        const bool is_final = (i + 1 == active.size());
        const uint32_t cur_chunk = s.pi - chunk_start_pi;
        const uint32_t need_this = est_plan_cmd_bytes(p);
        const uint32_t need_next = is_final ? 0u : est_plan_cmd_bytes(active[i + 1]->impl());
        const bool is_chunk_end = !is_final && (cur_chunk + need_this + need_next > kChunkBudget);

        std::array<uint32_t, 64> skip_sig;
        const uint32_t eff_host = p.host_credits - compute_skip_sig(p, skip_sig);
        s.lbw(base + s.chain_sob * 4, 0);
        const uint32_t cmon = is_final ? s.chain_mon_host : (is_chunk_end ? s.chain_mon_both : s.chain_mon_fence);
        s.arm_only(cmon, s.chain_sob, eff_host);
        emit_plan_stages(s, p, skip_sig);
        if (!is_final) s.fence_wait(Impl::kChainFenceId, 1);

        if (is_final || is_chunk_end) {
            s.submit();
            ++in_flight;
            if (!is_final) {
                s.begin();
                chunk_start_pi = s.pi;
                if (in_flight >= 2) {
                    s.wait_fence(1);
                    --in_flight;
                }
            }
        }
    }
    return in_flight;
}

void DirectRuntime::wait_chain(uint32_t in_flight) {
    if (in_flight > 0) impl_->wait_fence(in_flight);
}

// Steps of one or two plans back-to-back on the device. A step's two plans (lanes 1 and 2) run concurrently: their
// stages are dispatched interleaved in the given order (each engine executes its stages in dispatch order, so the
// order decides which plan an engine serves first), and the next step starts when both have finished.
double DirectRuntime::run_waves(std::span<const DirectWave> steps) {
    auto& s = *impl_;
    if (steps.empty()) return 0.0;
    if (!s.chain_enabled) throw std::runtime_error("direct_mme: run_waves needs the on-device chaining");
    std::vector<DirectPlan*> all;
    for (const auto& w : steps)
        for (DirectPlan* pl : {w.a, w.b})
            if (pl) {
                if (!pl->impl().prepared) pl->prepare();
                all.push_back(pl);
            }
    for (const auto& w : steps) {
        if (!w.a) throw std::invalid_argument("direct_mme: a wave step needs its first plan");
        if (w.b && (w.a->impl().lane == 0 || w.b->impl().lane == 0 || w.a->impl().lane == w.b->impl().lane))
            throw std::invalid_argument("direct_mme: the two plans of a wave step need different lanes");
    }
    bool any_dirty = false;
    for (DirectPlan* pl : all) any_dirty |= pl->impl().any_dirty;
    if (any_dirty) flush_patches(all);

    auto need = [&](const DirectWave& w) { return est_plan_cmd_bytes(w.a->impl()) + (w.b ? est_plan_cmd_bytes(w.b->impl()) : 0u) + 128u; };
    constexpr uint32_t kChunkBudget = 20480u;
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    const double t0 = now_us();
    std::vector<uint32_t> flight;  // bytes of the submitted chunks not yet known to be complete (oldest first)
    uint32_t flight_bytes = 0;
    s.begin();
    uint32_t chunk_start_pi = s.pi;
    for (size_t i = 0; i < steps.size(); ++i) {
        const auto& w = steps[i];
        const bool is_final = i + 1 == steps.size();
        const uint32_t cur_chunk = s.pi - chunk_start_pi;
        const uint32_t need_this = need(w), need_next = is_final ? 0u : need(steps[i + 1]);
        // the ring keeps the commands of the chunks still in flight: wait for the oldest when this step may not fit
        while (!flight.empty() && flight_bytes + cur_chunk + need_this + need_this / 4 + 2048 > kOwnRing - 2048) {
            s.wait_fence(1);
            flight_bytes -= flight.front();
            flight.erase(flight.begin());
        }
        const bool is_chunk_end = !is_final && (cur_chunk + need_this + need_next > kChunkBudget);
        const auto& pa = w.a->impl();
        std::array<uint32_t, 64> skip_a, skip_b;
        const uint32_t cred_a = pa.host_credits - compute_skip_sig(pa, skip_a);
        const uint32_t cred_b = w.b ? (w.b->impl().host_credits - compute_skip_sig(w.b->impl(), skip_b)) : 0u;
        const uint32_t credits = cred_a + cred_b;
        s.lbw(base + s.chain_sob * 4, 0);
        const uint32_t cmon = is_final ? s.chain_mon_host : (is_chunk_end ? s.chain_mon_both : s.chain_mon_fence);
        s.arm_only(cmon, s.chain_sob, credits);
        if (!w.b) {
            emit_plan_stages(s, pa, skip_a);
        } else {
            const auto& pb = w.b->impl();
            emit_sob_resets(s, pa);
            emit_sob_resets(s, pb);
            // the given order (bit 15: plan b; low bits: stage index), then whatever it left out, in plan order
            std::vector<uint8_t> seen_a(pa.stages.size(), 0), seen_b(pb.stages.size(), 0);
            for (uint16_t o : w.order) {
                const size_t k = o & 0x7fffu;
                if (!(o & 0x8000u) && k < pa.stages.size() && !seen_a[k]) { emit_stage(s, pa, pa.stages[k], skip_a); seen_a[k] = 1; }
                else if ((o & 0x8000u) && k < pb.stages.size() && !seen_b[k]) { emit_stage(s, pb, pb.stages[k], skip_b); seen_b[k] = 1; }
            }
            for (size_t k = 0; k < pa.stages.size(); ++k)
                if (!seen_a[k]) emit_stage(s, pa, pa.stages[k], skip_a);
            for (size_t k = 0; k < pb.stages.size(); ++k)
                if (!seen_b[k]) emit_stage(s, pb, pb.stages[k], skip_b);
        }
        if (!is_final) s.fence_wait(Impl::kChainFenceId, 1);
        if (is_final || is_chunk_end) {
            s.submit();
            flight.push_back(s.pi - chunk_start_pi);
            flight_bytes += flight.back();
            s.begin();
            chunk_start_pi = s.pi;
            if (!is_final && flight.size() >= 2) {
                s.wait_fence(1);
                flight_bytes -= flight.front();
                flight.erase(flight.begin());
            }
        }
    }
    if (!flight.empty()) s.wait_fence(flight.size());
    return now_us() - t0;
}

void DirectRuntime::flush_patches(std::span<DirectPlan* const> plans) {
    flush_patches_and_upload(plans, 0, nullptr, 0);
}

void DirectRuntime::flush_patches_and_upload_async(std::span<DirectPlan* const> plans, uint64_t dst_addr, const void* src, size_t bytes) {
    auto& s = *impl_;
    size_t total = bytes;
    for (DirectPlan* pl : plans) {
        if (!pl) continue;
        auto& p = pl->impl();
        if (!p.any_dirty) continue;
        for (PreparedEcb* ecb : p.dirty_ecbs) total += ecb->shadow.size();
    }
    if (!total) {
        for (DirectPlan* pl : plans) {
            if (!pl) continue;
            pl->impl().dirty_ecbs.clear();
            pl->impl().any_dirty = false;
        }
        return;
    }
    s.ensure_patch_buf(total);
    size_t off = 0;
    if (bytes && src) {
        std::memcpy(static_cast<char*>(s.patch_stage), src, bytes);
        check(synMemCopyAsync(s.patch_stream, reinterpret_cast<uintptr_t>(s.patch_stage),
                              bytes, dst_addr, HOST_TO_DRAM),
              "direct_mme ctrl upload");
        off += bytes;
    }
    for (DirectPlan* pl : plans) {
        if (!pl) continue;
        auto& p = pl->impl();
        if (!p.any_dirty) continue;
        std::sort(p.dirty_ecbs.begin(), p.dirty_ecbs.end(),
                  [](const PreparedEcb* a, const PreparedEcb* b) { return a->code_dev_addr < b->code_dev_addr; });
        size_t i = 0;
        while (i < p.dirty_ecbs.size()) {
            const uint64_t run_dst = p.dirty_ecbs[i]->code_dev_addr;
            const size_t run_off = off;
            uint64_t run_bytes = 0;
            while (i < p.dirty_ecbs.size() && p.dirty_ecbs[i]->code_dev_addr == run_dst + run_bytes) {
                auto* ecb = p.dirty_ecbs[i++];
                std::memcpy(static_cast<char*>(s.patch_stage) + off, ecb->shadow.data(), ecb->shadow.size());
                off += ecb->shadow.size();
                run_bytes += ecb->shadow.size();
                ecb->dirty = false;
            }
            check(synMemCopyAsync(s.patch_stream, reinterpret_cast<uintptr_t>(s.patch_stage) + run_off,
                                  run_bytes, run_dst, HOST_TO_DRAM),
                  "direct_mme batch patch upload");
        }
        p.dirty_ecbs.clear();
        p.any_dirty = false;
    }
}

void DirectRuntime::wait_patches() {
    check(synStreamSynchronize(impl_->patch_stream), "direct_mme batch patch sync");
}

// Synchronous flush: over the patch stream (Synapse), or with Options::pdma_patches as PDMA commands of the
// compute scheduler on the runtime's own stream (no Synapse stream synchronisation in the caller's hot path).
void DirectRuntime::flush_patches_and_upload(std::span<DirectPlan* const> plans, uint64_t dst_addr, const void* src, size_t bytes) {
    auto& s = *impl_;
    if (!s.opt.pdma_patches) {
        flush_patches_and_upload_async(plans, dst_addr, src, bytes);
        wait_patches();
        return;
    }
    size_t total = bytes;
    for (DirectPlan* pl : plans) {
        if (!pl) continue;
        auto& p = pl->impl();
        if (!p.any_dirty) continue;
        for (PreparedEcb* ecb : p.dirty_ecbs) total += ecb->shadow.size();
    }
    if (!total) {
        for (DirectPlan* pl : plans) {
            if (!pl) continue;
            pl->impl().dirty_ecbs.clear();
            pl->impl().any_dirty = false;
        }
        return;
    }
    s.ensure_pdma_buf(total);
    std::vector<sched_arc_pdma_commands_params_t> xfers;
    xfers.reserve(64);
    size_t off = 0;
    if (bytes && src) {
        std::memcpy(static_cast<char*>(s.pdma_host), src, bytes);
        xfers.push_back({uint32_t(bytes), s.pdma_dev, dst_addr});
        off += bytes;
    }
    for (DirectPlan* pl : plans) {
        if (!pl) continue;
        auto& p = pl->impl();
        if (!p.any_dirty) continue;
        std::sort(p.dirty_ecbs.begin(), p.dirty_ecbs.end(),
                  [](const PreparedEcb* a, const PreparedEcb* b) { return a->code_dev_addr < b->code_dev_addr; });
        size_t i = 0;
        while (i < p.dirty_ecbs.size()) {
            const uint64_t dst0 = p.dirty_ecbs[i]->code_dev_addr;
            const size_t src_off0 = off;
            uint64_t run_bytes = 0;
            while (i < p.dirty_ecbs.size() && p.dirty_ecbs[i]->code_dev_addr == dst0 + run_bytes) {
                PreparedEcb* ecb = p.dirty_ecbs[i++];
                std::memcpy(static_cast<char*>(s.pdma_host) + off, ecb->shadow.data(), ecb->shadow.size());
                off += ecb->shadow.size();
                run_bytes += ecb->shadow.size();
                ecb->dirty = false;
            }
            xfers.push_back({uint32_t(run_bytes), s.pdma_dev + src_off0, dst0});
        }
        p.dirty_ecbs.clear();
        p.any_dirty = false;
    }
    if (xfers.empty()) return;
    s.begin();
    uint32_t ncmds = 0;
    for (size_t i = 0; i < xfers.size(); i += 2) {
        const uint32_t cnt = (i + 2 <= xfers.size()) ? 2u : 1u;
        s.emit_pdma_batch(SCAL_PDMA_TX_DATA_GROUP, PDMA_DIR_HOST2DEV, &xfers[i], cnt, s.host_sob);
        ++ncmds;
    }
    s.submit();
    s.wait_fence(ncmds, 500000);
}

}  // namespace g2e
