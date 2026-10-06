#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

struct StdVector {
  void* begin;
  void* end;
  void* cap_end;
};

using LaserCtor = void(__cdecl*)(void*);
using ParseFrame = void(__cdecl*)(void*, const char*, int, StdVector*, int);

static std::size_t vec_bytes(const StdVector* v) {
  if (!v->begin || !v->end) return 0;
  auto b = reinterpret_cast<std::uintptr_t>(v->begin);
  auto e = reinterpret_cast<std::uintptr_t>(v->end);
  return e >= b ? static_cast<std::size_t>(e - b) : 0;
}

int main(int argc, char** argv) {
  const char* ltp = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\2026-03-24-123712_002.ltp";
  const int max_frames = (argc > 2) ? std::atoi(argv[2]) : 5;
  const wchar_t* copre = L"C:\\Program Files\\CHCNAV\\CoPre2\\Release";
  SetDllDirectoryW(copre);
  SetCurrentDirectoryW(copre);
  HMODULE h_lt = LoadLibraryW(L"LTLaser.dll");
  if (!h_lt) return 1;

  auto ctor = reinterpret_cast<LaserCtor>(
      GetProcAddress(h_lt, "??0LTLaserReader@lt@@QEAA@XZ"));
  auto parse_frame = reinterpret_cast<ParseFrame>(GetProcAddress(
      h_lt,
      "?parseFrame@LTLaserReader@lt@@QEAAXPEBDHAEAV?$vector@URawLaserPoint@lt@@V?$allocator@URawLaserPoint@lt@@@std@@@std@@H@Z"));
  if (!ctor || !parse_frame) {
    std::printf("missing exports\n");
    return 2;
  }

  FILE* f = std::fopen(ltp, "rb");
  if (!f) return 3;
  std::fseek(f, 0, SEEK_END);
  const long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  char* data = static_cast<char*>(std::malloc(static_cast<std::size_t>(sz)));
  if (!data || std::fread(data, 1, static_cast<std::size_t>(sz), f) !=
                  static_cast<std::size_t>(sz)) {
    std::fclose(f);
    return 4;
  }
  std::fclose(f);

  void* reader = operator new[](65536);
  ctor(reader);

  const unsigned char prefix[] = {0x5A, 0x23, 0x05};
  long pos = 512;
  int parsed = 0;
  int total_pts = 0;
  for (int fi = 0; fi < max_frames && pos + 1461 < sz; ++fi) {
    while (pos + 1461 < sz &&
           (data[pos] != prefix[0] || data[pos + 1] != prefix[1] ||
            data[pos + 2] != prefix[2])) {
      ++pos;
    }
    if (pos + 1461 >= sz) break;

    StdVector pts{};
    parse_frame(reader, data + pos, 1461, &pts, 0);
    const std::size_t nbytes = vec_bytes(&pts);
    std::printf("frame %d @%ld type=0x%02x bytes=%zu\n", fi, pos,
                static_cast<unsigned>(data[pos + 3] & 0xFF), nbytes);

    const std::size_t elems[] = {32U, 40U, 48U, 56U, 64U, 72U, 80U};
    for (std::size_t ei = 0; ei < sizeof(elems) / sizeof(elems[0]); ++ei) {
      const std::size_t elem = elems[ei];
      if (elem == 0 || nbytes % elem != 0) continue;
      const std::size_t n = nbytes / elem;
      if (n == 0 || n > 500) continue;
      const auto* base = reinterpret_cast<const float*>(pts.begin);
      const int nf = static_cast<int>(elem / sizeof(float));
      std::printf("  elem=%zu count=%zu first:", elem, n);
      for (int j = 0; j < nf && j < 8; ++j) std::printf(" %.4f", base[j]);
      std::printf("\n");
      total_pts += static_cast<int>(n);
      break;
    }
    pos += 1461;
    ++parsed;
  }

  std::printf("parsed_frames=%d total_pts_guess=%d\n", parsed, total_pts);
  operator delete[](reader);
  std::free(data);
  return total_pts > 0 ? 0 : 5;
}
