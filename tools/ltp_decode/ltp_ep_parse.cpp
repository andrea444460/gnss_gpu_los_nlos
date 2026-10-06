#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <new>

struct StdVector {
  void* begin;
  void* end;
  void* cap_end;
};

static size_t vec_count(const StdVector* v, size_t elem) {
  if (!v->begin || !v->end) return 0;
  auto b = (uintptr_t)v->begin;
  auto e = (uintptr_t)v->end;
  return e >= b ? (size_t)((e - b) / elem) : 0;
}

int main(int argc, char** argv) {
  const char* ltp = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\2026-03-24-123712_002.ltp";
  const char* ep = (argc > 2) ? argv[2]
                               : "c:\\Users\\Andrea\\Desktop\\AP7-007-CAR-SCAN1-WSM-20260307-s20020002J.EP";
  SetDllDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  SetCurrentDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  HMODULE h = LoadLibraryW(L"LTLaser.dll");
  if (!h) return 1;

  using HeaderCtor = void(__cdecl*)(void*);
  using CustomCtor = void(__cdecl*)(void*);
  using ReaderCtor = void(__cdecl*)(void*);
  using BufToHeader1xx = bool(__cdecl*)(void*, const char*, bool);
  using GetPackType = int(__cdecl*)(void*);
  using IsValid = bool(__cdecl*)(void*);
  using InitPackType = void*(__cdecl*)(void*, void*, int);
  using ParseFrame = void(__cdecl*)(void*, const char*, int, StdVector*, int);

  auto header_ctor = (HeaderCtor)GetProcAddress(h, "??0LTLaserHeader@lt@@QEAA@XZ");
  auto custom_ctor = (CustomCtor)GetProcAddress(h, "??0LTLaserCustom@lt@@QEAA@XZ");
  auto reader_ctor = (ReaderCtor)GetProcAddress(h, "??0LTLaserReader@lt@@QEAA@XZ");
  auto buf_to_header1xx = (BufToHeader1xx)GetProcAddress(
      h, "?bufToHeader1xx@LTParseHeader1xx@lt@@SA_NAEAVLTLaserHeader@2@PEBD_N@Z");
  auto get_pack_type = (GetPackType)GetProcAddress(h, "?getPackType@LTLaserHeader@lt@@QEAAHXZ");
  auto is_valid = (IsValid)GetProcAddress(h, "?isValid@LTLaserHeader@lt@@QEAA_NXZ");
  auto init_pack = (InitPackType)GetProcAddress(
      h, "?initPackType@LTLaserReader@lt@@SAPEAVLTPackType@2@AEAVLTLaserHeader@2@AEAVLTLaserCustom@2@H@Z");
  auto parse_frame = (ParseFrame)GetProcAddress(
      h, "?parseFrame@LTLaserReader@lt@@QEAAXPEBDHAEAV?$vector@URawLaserPoint@lt@@V?$allocator@URawLaserPoint@lt@@@std@@@std@@H@Z");

  FILE* ef = fopen(ep, "rb");
  fseek(ef, 0, SEEK_END);
  long esz = ftell(ef);
  fseek(ef, 0, SEEK_SET);
  char* edata = (char*)malloc((size_t)esz);
  fread(edata, 1, (size_t)esz, ef);
  fclose(ef);

  void* header = operator new[](262144);
  void* custom = operator new[](65536);
  void* reader = operator new[](262144);
  header_ctor(header);
  custom_ctor(custom);
  reader_ctor(reader);
  bool ep_ok = buf_to_header1xx(header, edata, true);
  printf("EP loaded %d valid=%d packType=%d\n", ep_ok ? 1 : 0,
         is_valid ? (is_valid(header) ? 1 : 0) : -1,
         get_pack_type ? get_pack_type(header) : -1);

  void* pack = nullptr;
  if (init_pack && get_pack_type) {
    int pt = get_pack_type(header);
    for (int mode = 0; mode <= 3; ++mode) {
      pack = init_pack(header, custom, mode);
      printf("initPackType(mode=%d) -> %p\n", mode, pack);
      if (pack) break;
    }
  }

  FILE* lf = fopen(ltp, "rb");
  fseek(lf, 0, SEEK_END);
  long lsz = ftell(lf);
  fseek(lf, 0, SEEK_SET);
  char* ldata = (char*)malloc((size_t)lsz);
  fread(ldata, 1, (size_t)lsz, lf);
  fclose(lf);
  long idx = 512;
  const unsigned char prefix[] = {0x5A, 0x23, 0x05};
  while (idx + 1461 < lsz &&
         (ldata[idx] != prefix[0] || ldata[idx + 1] != prefix[1] ||
          ldata[idx + 2] != prefix[2]))
    ++idx;

  if (parse_frame) {
    StdVector pts{};
    parse_frame(reader, ldata + idx, 1461, &pts, 0);
    printf("parseFrame bytes=%llu\n",
           (unsigned long long)((uintptr_t)pts.end - (uintptr_t)pts.begin));
    const size_t elems[] = {32, 40, 48, 56, 64};
    for (int ei = 0; ei < 5; ++ei) {
      size_t elem = elems[ei];
      size_t n = vec_count(&pts, elem);
      if (n) {
        printf("  elem=%zu count=%zu\n", elem, n);
        const float* base = (const float*)pts.begin;
        int nf = (int)(elem / sizeof(float));
        for (int i = 0; i < 3 && (size_t)i < n; ++i) {
          printf("    pt%d:", i);
          for (int j = 0; j < nf && j < 8; ++j) printf(" %.4f", base[i * nf + j]);
          printf("\n");
        }
      }
    }
  }

  free(edata);
  free(ldata);
  operator delete[](header);
  operator delete[](custom);
  operator delete[](reader);
  return 0;
}
