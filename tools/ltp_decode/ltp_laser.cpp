#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// MSVC x64 std::string SSO layout (short strings only).
struct MsvcString {
  union {
    char buf[16];
    char* ptr;
  };
  std::uint64_t size;
  std::uint64_t cap;
};

static void msvc_string_set(MsvcString* s, const char* text) {
  const std::size_t n = std::strlen(text);
  std::memset(s, 0, sizeof(*s));
  s->size = n;
  s->cap = 15;
  std::memcpy(s->buf, text, n + 1);
}

struct StdVector {
  void* begin;
  void* end;
  void* cap_end;
};

using LaserCtor = void(__cdecl*)(void*);
using LaserInit = bool(__cdecl*)(void*);
using LaserInitMode = bool(__cdecl*)(void*, MsvcString, int);
using LaserOpen = bool(__cdecl*)(void*, MsvcString, bool, int);
using GetRawPts = int(__cdecl*)(void*, StdVector*);
using GetRawPtsN = int(__cdecl*)(void*, StdVector*, int);
using ParseFrame = void(__cdecl*)(void*, const char*, int, StdVector*, int);

static std::size_t vec_count(const StdVector* v, std::size_t elem) {
  if (!v->begin || !v->end) {
    return 0;
  }
  const auto b = reinterpret_cast<std::uintptr_t>(v->begin);
  const auto e = reinterpret_cast<std::uintptr_t>(v->end);
  if (e < b) {
    return 0;
  }
  return static_cast<std::size_t>((e - b) / elem);
}

static void dump_first_points(const StdVector* v, std::size_t elem, int max_n) {
  const std::size_t n = vec_count(v, elem);
  std::printf("vector count=%zu elem=%zu bytes\n", n, elem);
  if (n == 0 || !v->begin) {
    return;
  }
  const int show = (max_n < static_cast<int>(n)) ? max_n : static_cast<int>(n);
  const auto* base = reinterpret_cast<const float*>(v->begin);
  const int floats_per = static_cast<int>(elem / sizeof(float));
  for (int i = 0; i < show; ++i) {
    const float* p = base + i * floats_per;
    std::printf("  pt[%d]:", i);
    for (int j = 0; j < floats_per && j < 8; ++j) {
      std::printf(" %.4f", p[j]);
    }
    std::printf("\n");
  }
}

int main(int argc, char** argv) {
  const char* ltp = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\2026-03-24-123712_002.ltp";
  const wchar_t* copre = L"C:\\Program Files\\CHCNAV\\CoPre2\\Release";
  SetDllDirectoryW(copre);
  SetCurrentDirectoryW(copre);

  HMODULE h_lt = LoadLibraryW(L"LTLaser.dll");
  if (!h_lt) {
    std::printf("LTLaser.dll load fail\n");
    return 1;
  }

  auto ctor = reinterpret_cast<LaserCtor>(
      GetProcAddress(h_lt, "??0LTLaserReader@lt@@QEAA@XZ"));
  auto init = reinterpret_cast<LaserInit>(
      GetProcAddress(h_lt, "?init@LTLaserReader@lt@@QEAA_NXZ"));
  auto init_mode = reinterpret_cast<LaserInitMode>(GetProcAddress(
      h_lt,
      "?initByOpenMode@LTLaserReader@lt@@QEAA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@H@Z"));
  auto open = reinterpret_cast<LaserOpen>(GetProcAddress(
      h_lt,
      "?open@LTLaserReader@lt@@UEAA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@_NH@Z"));
  auto get_raw = reinterpret_cast<GetRawPts>(GetProcAddress(
      h_lt,
      "?getRawPoints@LTLaserReader@lt@@UEAAHAEAV?$vector@URawLaserPoint@lt@@V?$allocator@URawLaserPoint@lt@@@std@@@std@@@Z"));
  auto get_raw_n = reinterpret_cast<GetRawPtsN>(GetProcAddress(
      h_lt,
      "?getRawPoints@LTLaserReader@lt@@UEAAHAEAV?$vector@URawLaserPoint@lt@@V?$allocator@URawLaserPoint@lt@@@std@@@std@@H@Z"));
  auto parse_frame = reinterpret_cast<ParseFrame>(GetProcAddress(
      h_lt,
      "?parseFrame@LTLaserReader@lt@@QEAAXPEBDHAEAV?$vector@URawLaserPoint@lt@@V?$allocator@URawLaserPoint@lt@@@std@@@std@@H@Z"));

  if (!ctor || !open || !get_raw) {
    std::printf("missing exports ctor=%p open=%p get_raw=%p\n",
                reinterpret_cast<void*>(ctor),
                reinterpret_cast<void*>(open),
                reinterpret_cast<void*>(get_raw));
    return 2;
  }

  void* reader = operator new[](65536);
  ctor(reader);

  MsvcString path{};
  msvc_string_set(&path, ltp);

  if (init_mode) {
    for (int mode : {0, 1, 2, 3}) {
      std::printf("initByOpenMode(%d) -> %d\n", mode,
                  init_mode(reader, path, mode) ? 1 : 0);
    }
  } else if (init) {
    std::printf("init -> %d\n", init(reader) ? 1 : 0);
  }

  bool opened = false;
  if (open) {
    std::printf("calling open...\n");
    std::fflush(stdout);
    for (int flag : {0, 1}) {
      for (int mode : {0, 1, 2}) {
        opened = open(reader, path, flag != 0, mode);
        std::printf("open(flag=%d,mode=%d) -> %d\n", flag, mode, opened ? 1 : 0);
        if (opened) {
          break;
        }
      }
      if (opened) {
        break;
      }
    }
  }

  StdVector pts{};
  int rc = get_raw(reader, &pts);
  std::printf("getRawPoints rc=%d\n", rc);
  dump_first_points(&pts, 48, 3);

  if (vec_count(&pts, 48) == 0 && get_raw_n) {
    pts = {};
    rc = get_raw_n(reader, &pts, 1);
    std::printf("getRawPoints(n=1) rc=%d\n", rc);
    dump_first_points(&pts, 48, 3);
  }

  if (vec_count(&pts, 48) == 0 && parse_frame) {
    FILE* f = std::fopen(ltp, "rb");
    if (f) {
      std::fseek(f, 0, SEEK_END);
      const long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      char* data = static_cast<char*>(std::malloc(static_cast<std::size_t>(sz)));
      if (data && std::fread(data, 1, static_cast<std::size_t>(sz), f) ==
                      static_cast<std::size_t>(sz)) {
        const unsigned char sync[] = {0x5A, 0x23, 0x05, 0xA6};
        long idx = 512;
        for (; idx + 1461 < sz; ++idx) {
          if (std::memcmp(data + idx, sync, 4) == 0) {
            break;
          }
        }
        std::printf("first sync frame at offset %ld\n", idx);
        if (idx + 1461 < sz) {
          StdVector frame_pts{};
          parse_frame(reader, data + idx, 1461, &frame_pts, 0);
          std::printf("parseFrame(1461B @%ld)\n", idx);
          dump_first_points(&frame_pts, 48, 3);
        }
      }
      std::free(data);
      std::fclose(f);
    }
  }

  operator delete[](reader);
  return vec_count(&pts, 48) > 0 ? 0 : 3;
}
