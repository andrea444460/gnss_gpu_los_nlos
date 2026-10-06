#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

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

using HeaderCtor = void(__cdecl*)(void*);
using HeaderRead = bool(__cdecl*)(void*, MsvcString, bool);
using ReadHeader1xx = bool(__cdecl*)(MsvcString, void*, bool);
using LaserCtor = void(__cdecl*)(void*);
using LaserInit = bool(__cdecl*)(void*);
using LaserInitMode = bool(__cdecl*)(void*, MsvcString, int);
using LaserOpen = bool(__cdecl*)(void*, MsvcString, bool, int);
using GenRtCal = bool(__cdecl*)(void*, MsvcString, int, int, MsvcString);
using ParseFrame = void(__cdecl*)(void*, const char*, int, StdVector*, int);

static std::size_t vec_count(const StdVector* v, std::size_t elem) {
  if (!v->begin || !v->end) return 0;
  auto b = reinterpret_cast<std::uintptr_t>(v->begin);
  auto e = reinterpret_cast<std::uintptr_t>(v->end);
  return e >= b ? static_cast<std::size_t>((e - b) / elem) : 0;
}

static void dump_pts(const StdVector* v, std::size_t elem, int max_n) {
  const std::size_t n = vec_count(v, elem);
  std::printf("  vector count=%zu elem=%zu\n", n, elem);
  if (!n || !v->begin) return;
  const auto* base = reinterpret_cast<const float*>(v->begin);
  const int nf = static_cast<int>(elem / sizeof(float));
  const int show = max_n < static_cast<int>(n) ? max_n : static_cast<int>(n);
  for (int i = 0; i < show; ++i) {
    const float* p = base + i * nf;
    std::printf("    pt[%d]:", i);
    for (int j = 0; j < nf && j < 10; ++j) std::printf(" %.4f", p[j]);
    std::printf("\n");
  }
}

int main(int argc, char** argv) {
  const char* ltp = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\2026-03-24-123712_002.ltp";
  const char* ep = (argc > 2) ? argv[2]
                               : "c:\\Users\\Andrea\\Desktop\\AP7-007-CAR-SCAN1-WSM-20260307-s20020002J.EP";
  const wchar_t* copre = L"C:\\Program Files\\CHCNAV\\CoPre2\\Release";
  SetDllDirectoryW(copre);
  SetCurrentDirectoryW(copre);

  HMODULE h_lt = LoadLibraryW(L"LTLaser.dll");
  if (!h_lt) {
    std::printf("LTLaser.dll load fail\n");
    return 1;
  }

  auto header_ctor = reinterpret_cast<HeaderCtor>(
      GetProcAddress(h_lt, "??0LTLaserHeader@lt@@QEAA@XZ"));
  auto header_read = reinterpret_cast<HeaderRead>(GetProcAddress(
      h_lt, "?read@LTLaserHeader@lt@@QEAA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@_N@Z"));
  auto read_header1xx = reinterpret_cast<ReadHeader1xx>(GetProcAddress(
      h_lt,
      "?readHeader1xx@LTParseHeader1xx@lt@@SA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAVLTLaserHeader@2@_N@Z"));
  auto laser_ctor = reinterpret_cast<LaserCtor>(
      GetProcAddress(h_lt, "??0LTLaserReader@lt@@QEAA@XZ"));
  auto laser_init = reinterpret_cast<LaserInit>(
      GetProcAddress(h_lt, "?init@LTLaserReader@lt@@QEAA_NXZ"));
  auto laser_init_mode = reinterpret_cast<LaserInitMode>(GetProcAddress(
      h_lt,
      "?initByOpenMode@LTLaserReader@lt@@QEAA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@H@Z"));
  auto laser_open = reinterpret_cast<LaserOpen>(GetProcAddress(
      h_lt,
      "?open@LTLaserReader@lt@@UEAA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@_NH@Z"));
  auto gen_rt = reinterpret_cast<GenRtCal>(GetProcAddress(
      h_lt,
      "?genRealTimeCalParam@LTLaserReader@lt@@QEAA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@HH0@Z"));
  auto parse_frame = reinterpret_cast<ParseFrame>(GetProcAddress(
      h_lt,
      "?parseFrame@LTLaserReader@lt@@QEAAXPEBDHAEAV?$vector@URawLaserPoint@lt@@V?$allocator@URawLaserPoint@lt@@@std@@@std@@H@Z"));

  void* header = operator new[](131072);
  void* reader = operator new[](131072);
  header_ctor(header);
  laser_ctor(reader);

  MsvcString ep_path{};
  msvc_string_set(&ep_path, ep);
  if (read_header1xx) {
    for (int flag = 0; flag <= 1; ++flag) {
      std::printf("readHeader1xx(flag=%d) -> %d\n", flag,
                  read_header1xx(ep_path, header, flag != 0) ? 1 : 0);
    }
  }
  if (header_read) {
    for (int flag = 0; flag <= 1; ++flag) {
      std::printf("LTLaserHeader::read(flag=%d) -> %d\n", flag,
                  header_read(header, ep_path, flag != 0) ? 1 : 0);
    }
  }

  MsvcString ltp_path{};
  msvc_string_set(&ltp_path, ltp);
  if (gen_rt) {
    MsvcString extra{};
    msvc_string_set(&extra, ep);
    for (int a = 0; a <= 2; ++a) {
      for (int b = 0; b <= 2; ++b) {
        bool ok = gen_rt(reader, ltp_path, a, b, extra);
        std::printf("genRealTimeCalParam(%d,%d,ep) -> %d\n", a, b, ok ? 1 : 0);
      }
    }
  }
  if (laser_init) {
    std::printf("init -> %d\n", laser_init(reader) ? 1 : 0);
  }
  if (laser_init_mode) {
    for (int mode = 0; mode <= 5; ++mode) {
      std::printf("initByOpenMode(%d) -> %d\n", mode,
                  laser_init_mode(reader, ltp_path, mode) ? 1 : 0);
    }
  }
  bool opened = false;
  if (laser_open) {
    for (int flag = 0; flag <= 1 && !opened; ++flag) {
      for (int mode = 0; mode <= 3; ++mode) {
        opened = laser_open(reader, ltp_path, flag != 0, mode);
        std::printf("open(flag=%d,mode=%d) -> %d\n", flag, mode, opened ? 1 : 0);
        if (opened) break;
      }
    }
  }

  if (parse_frame) {
    FILE* f = std::fopen(ltp, "rb");
    if (f) {
      std::fseek(f, 0, SEEK_END);
      const long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      char* data = static_cast<char*>(std::malloc(static_cast<std::size_t>(sz)));
      if (data && std::fread(data, 1, static_cast<std::size_t>(sz), f) ==
                      static_cast<std::size_t>(sz)) {
        long idx = 512;
        const unsigned char prefix[] = {0x5A, 0x23, 0x05};
        while (idx + 1461 < sz &&
               (data[idx] != prefix[0] || data[idx + 1] != prefix[1] ||
                data[idx + 2] != prefix[2])) {
          ++idx;
        }
        std::printf("first frame @ %ld type=0x%02x\n", idx,
                    static_cast<unsigned>(data[idx + 3] & 0xFF));
        for (int fi = 0; fi < 3 && idx + 1461 < sz; ++fi) {
          StdVector pts{};
          parse_frame(reader, data + idx, 1461, &pts, 0);
          std::printf("parseFrame frame %d:\n", fi);
          const std::size_t elems[] = {32U, 40U, 48U, 56U, 64U};
          for (int ei = 0; ei < 5; ++ei) {
            const std::size_t elem = elems[ei];
            if (vec_count(&pts, elem) > 0) {
              dump_pts(&pts, elem, 3);
            }
          }
          idx += 1461;
        }
      }
      std::free(data);
      std::fclose(f);
    }
  }

  operator delete[](header);
  operator delete[](reader);
  return 0;
}
