#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

using Ctor = void(__cdecl*)(void*);
using OpenFn = bool(__cdecl*)(void*, const char*);
using GetPtFn = int(__cdecl*)(void*, unsigned int, void*, unsigned int*);
using AttrsCtor = void(__cdecl*)(void*);
using CreatePtrFn = void(__cdecl*)(void*, const std::uint64_t*);
using FPtrFn = float*(__cdecl*)(void*, std::uint64_t);

int main(int argc, char** argv) {
  const char* ltp = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\2026-03-24-123712_002.ltp";
  const unsigned int max_frames =
      (argc > 2) ? static_cast<unsigned int>(std::atoi(argv[2])) : 20U;
  const wchar_t* copre = L"C:\\Program Files\\CHCNAV\\CoPre2\\Release";
  SetDllDirectoryW(copre);
  SetCurrentDirectoryW(copre);
  LoadLibraryW(L"LTLaser.dll");
  HMODULE h_ldr = LoadLibraryW(L"LiDARLTDataSolver.dll");
  HMODULE h_pcs = LoadLibraryW(L"PointCloudSolver.dll");
  if (!h_ldr || !h_pcs) {
    std::printf("dll load fail\n");
    return 1;
  }

  auto reader_ctor = reinterpret_cast<Ctor>(
      GetProcAddress(h_ldr, "??0RawLTLiDARDataReader@@QEAA@XZ"));
  auto open = reinterpret_cast<OpenFn>(
      GetProcAddress(h_ldr, "?openStream@RawLTLiDARDataReader@@UEAA_NPEBD@Z"));
  auto getpt = reinterpret_cast<GetPtFn>(GetProcAddress(
      h_ldr,
      "?getPointData@RawLTLiDARDataReader@@UEAA?AW4ReadResult@@IAEAURawLasPointAttributes@@PEAI@Z"));
  auto attrs_ctor = reinterpret_cast<AttrsCtor>(
      GetProcAddress(h_pcs, "??0RawLasPointAttributes@@QEAA@XZ"));
  auto create_ptr = reinterpret_cast<CreatePtrFn>(GetProcAddress(
      h_ldr, "?CreatePtr@RawLasPointAttributes@@QEAAXAEB_K@Z"));
  auto f_ptr = reinterpret_cast<FPtrFn>(GetProcAddress(
      h_pcs, "?fPointer@RawLasPointAttributes@@QEAAPEAMAEB_K@Z"));

  void* reader = operator new[](65536);
  void* attrs = operator new[](65536);
  reader_ctor(reader);
  std::printf("open -> %d\n", open(reader, ltp));
  attrs_ctor(attrs);
  std::uint64_t cap = 200000;
  create_ptr(attrs, &cap);

  unsigned int total = 0;
  for (unsigned int frame = 0; frame < max_frames; ++frame) {
    unsigned int n = 0;
    int rc = getpt(reader, frame, attrs, &n);
    std::printf("frame %u: rc=%d n=%u\n", frame, rc, n);
    if (rc != 0) {
      break;
    }
    if (n > 0 && f_ptr) {
      total = n;
      float* p = f_ptr(attrs, 0);
      if (p) {
        std::printf("first: %.4f %.4f %.4f %.4f\n", p[0], p[1], p[2], p[3]);
      }
      break;
    }
  }
  std::printf("total=%u cap=%llu\n", total,
              static_cast<unsigned long long>(cap));
  operator delete[](reader);
  operator delete[](attrs);
  return total > 0 ? 0 : 3;
}
