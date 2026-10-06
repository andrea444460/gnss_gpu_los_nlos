#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <new>

struct MsvcString {
  union { char buf[16]; char* ptr; };
  unsigned long long size;
  unsigned long long cap;
};

static void msvc_string_set(MsvcString* s, const char* text) {
  const size_t n = strlen(text);
  memset(s, 0, sizeof(*s));
  s->size = n;
  s->cap = 15;
  memcpy(s->buf, text, n + 1);
}

int main(int argc, char** argv) {
  const char* ep = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\AP7-007-CAR-SCAN1-WSM-20260307-s20020002J.EP";
  SetDllDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  SetCurrentDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  HMODULE h = LoadLibraryW(L"LTLaser.dll");
  if (!h) { printf("dll fail\n"); return 1; }
  printf("dll ok\n"); fflush(stdout);

  using HeaderCtor = void(__cdecl*)(void*);
  using ReadHeader1xx = bool(__cdecl*)(MsvcString, void*, bool);
  using HeaderRead = bool(__cdecl*)(void*, MsvcString, bool);

  auto header_ctor = (HeaderCtor)GetProcAddress(h, "??0LTLaserHeader@lt@@QEAA@XZ");
  auto read_header1xx = (ReadHeader1xx)GetProcAddress(
      h, "?readHeader1xx@LTParseHeader1xx@lt@@SA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAVLTLaserHeader@2@_N@Z");
  auto header_read = (HeaderRead)GetProcAddress(
      h, "?read@LTLaserHeader@lt@@QEAA_NV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@_N@Z");

  void* header = operator new[](65536);
  header_ctor(header);
  printf("header ctor ok\n"); fflush(stdout);

  MsvcString ep_path{};
  msvc_string_set(&ep_path, ep);
  if (header_read) {
    bool ok = header_read(header, ep_path, true);
    printf("header read -> %d\n", ok ? 1 : 0);
    fflush(stdout);
  }
  if (read_header1xx) {
    bool ok = read_header1xx(ep_path, header, true);
    printf("readHeader1xx -> %d\n", ok ? 1 : 0);
    fflush(stdout);
  }
  operator delete[](header);
  return 0;
}
