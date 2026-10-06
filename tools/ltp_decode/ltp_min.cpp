#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>

using Ctor = void(__cdecl*)(void*);
using OpenFn = bool(__cdecl*)(void*, const char*);

int main(int argc, char** argv) {
  const char* ltp = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\2026-03-24-123712_002.ltp";
  const wchar_t* copre = L"C:\\Program Files\\CHCNAV\\CoPre2\\Release";
  SetDllDirectoryW(copre);
  SetCurrentDirectoryW(copre);
  LoadLibraryW(L"LTLaser.dll");
  HMODULE h = LoadLibraryW(L"LiDARLTDataSolver.dll");
  if (!h) {
    std::printf("load fail %lu\n", GetLastError());
    return 1;
  }
  auto ctor = reinterpret_cast<Ctor>(
      GetProcAddress(h, "??0RawLTLiDARDataReader@@QEAA@XZ"));
  auto open = reinterpret_cast<OpenFn>(
      GetProcAddress(h, "?openStream@RawLTLiDARDataReader@@UEAA_NPEBD@Z"));
  alignas(16) unsigned char obj[8192] = {};
  std::printf("ctor\n");
  ctor(obj);
  std::printf("open\n");
  const bool ok = open(obj, ltp);
  std::printf("open -> %s\n", ok ? "true" : "false");
  return ok ? 0 : 2;
}
