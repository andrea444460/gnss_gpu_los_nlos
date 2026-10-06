#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <new>

int main(int argc, char** argv) {
  const char* ep = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\AP7-007-CAR-SCAN1-WSM-20260307-s20020002J.EP";
  SetDllDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  SetCurrentDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  HMODULE h = LoadLibraryW(L"LTLaser.dll");
  if (!h) return 1;

  using HeaderCtor = void(__cdecl*)(void*);
  using BufToHeader1xx = bool(__cdecl*)(void*, const char*, bool);
  using BufToHeader = bool(__cdecl*)(void*, const char*, int, bool);

  using GetPackType = int(__cdecl*)(void*);
  using IsValid = bool(__cdecl*)(void*);
  using BufToCalib = void(__cdecl*)(void*, const char*);
  using HeaderToBuf = bool(__cdecl*)(void*, char*, int, bool);

  auto header_ctor = (HeaderCtor)GetProcAddress(h, "??0LTLaserHeader@lt@@QEAA@XZ");
  auto buf_to_header1xx = (BufToHeader1xx)GetProcAddress(
      h, "?bufToHeader1xx@LTParseHeader1xx@lt@@SA_NAEAVLTLaserHeader@2@PEBD_N@Z");
  auto buf_to_header = (BufToHeader)GetProcAddress(
      h, "?bufToHeader@LTLaserHeader@lt@@QEAA_NPEBDH_N@Z");
  auto buf_to_calib = (BufToCalib)GetProcAddress(
      h, "?bufToCalibParam@LTParseHeader1xx@lt@@SAXAEAVLTLaserHeader@2@PEBD@Z");
  auto header_to_buf = (HeaderToBuf)GetProcAddress(
      h, "?headerToBuf@LTLaserHeader@lt@@QEAA_NPEADH_N@Z");
  auto get_pack_type = (GetPackType)GetProcAddress(h, "?getPackType@LTLaserHeader@lt@@QEAAHXZ");
  auto is_valid = (IsValid)GetProcAddress(h, "?isValid@LTLaserHeader@lt@@QEAA_NXZ");

  FILE* f = fopen(ep, "rb");
  if (!f) return 2;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* data = (char*)malloc((size_t)sz);
  fread(data, 1, (size_t)sz, f);
  fclose(f);

  void* header = operator new[](262144);
  header_ctor(header);
  printf("EP size %ld\n", sz);

  if (buf_to_header1xx) {
    for (int flag = 0; flag <= 1; ++flag) {
      bool ok = buf_to_header1xx(header, data, flag != 0);
      printf("bufToHeader1xx(flag=%d) -> %d\n", flag, ok ? 1 : 0);
    }
  }
  if (buf_to_calib) {
    buf_to_calib(header, data);
    printf("bufToCalibParam done\n");
  }
  if (get_pack_type) {
    printf("packType after calib=%d valid=%d\n", get_pack_type(header),
           is_valid ? (is_valid(header) ? 1 : 0) : -1);
  }
  if (is_valid && is_valid(header)) {
    const float* hf = (const float*)header;
    printf("nonzero floats in header object:\n");
    for (int i = 0; i < 20000; ++i) {
      float v = hf[i];
      if (v > -180.0f && v < 180.0f && v != 0.0f &&
          (v > 0.001f || v < -0.001f)) {
        printf("  off %d: %.6f\n", i * 4, v);
      }
    }
  }

  free(data);
  operator delete[](header);
  return 0;
}
