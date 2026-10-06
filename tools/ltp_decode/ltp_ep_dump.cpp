#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <new>

static float rf(void* hdr, int off) {
  return *(float*)((char*)hdr + off);
}

int main(int argc, char** argv) {
  const char* ep = (argc > 1) ? argv[1]
                               : "c:\\Users\\Andrea\\Desktop\\AP7-007-CAR-SCAN1-WSM-20260307-s20020002J.EP";
  SetDllDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  SetCurrentDirectoryW(L"C:\\Program Files\\CHCNAV\\CoPre2\\Release");
  HMODULE h = LoadLibraryW(L"LTLaser.dll");
  if (!h) return 1;

  using HeaderCtor = void(__cdecl*)(void*);
  using BufFn = void(__cdecl*)(void*, const char*);
  using BufToHeader1xx = bool(__cdecl*)(void*, const char*, bool);
  using GetVRot = int(__cdecl*)(void*);

  auto header_ctor = (HeaderCtor)GetProcAddress(h, "??0LTLaserHeader@lt@@QEAA@XZ");
  auto buf_to_header1xx = (BufToHeader1xx)GetProcAddress(
      h, "?bufToHeader1xx@LTParseHeader1xx@lt@@SA_NAEAVLTLaserHeader@2@PEBD_N@Z");
  auto buf_to_calib = (BufFn)GetProcAddress(
      h, "?bufToCalibParam@LTParseHeader1xx@lt@@SAXAEAVLTLaserHeader@2@PEBD@Z");
  auto buf_to_scan = (BufFn)GetProcAddress(
      h, "?bufToScanParam@LTParseHeader1xx@lt@@SAXAEAVLTLaserHeader@2@PEBD@Z");
  auto buf_to_sensor = (BufFn)GetProcAddress(
      h, "?bufToSensorInfo@LTParseHeader1xx@lt@@SAXAEAVLTLaserHeader@2@PEBD@Z");
  auto buf_to_base = (BufFn)GetProcAddress(
      h, "?bufToBaseInfo@LTParseHeader1xx@lt@@SAXAEAVLTLaserHeader@2@PEBD@Z");
  auto get_vrot = (GetVRot)GetProcAddress(h, "?getVRotSpeed@LTLaserHeader@lt@@QEAAHXZ");

  FILE* f = fopen(ep, "rb");
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* data = (char*)malloc((size_t)sz);
  fread(data, 1, (size_t)sz, f);
  fclose(f);

  void* header = operator new[](262144);
  header_ctor(header);
  buf_to_header1xx(header, data, true);
  if (buf_to_base) buf_to_base(header, data);
  if (buf_to_sensor) buf_to_sensor(header, data);
  if (buf_to_scan) buf_to_scan(header, data);
  if (buf_to_calib) buf_to_calib(header, data);

  printf("{\n");
  printf("  \"ok\": true,\n");
  printf("  \"vrot_speed\": %d,\n", get_vrot ? get_vrot(header) : 0);
  printf("  \"mount_roll_deg\": %.8g,\n", (double)rf(header, 712));
  printf("  \"mount_pitch_deg\": %.8g,\n", (double)rf(header, 716));
  printf("  \"mount_yaw_deg\": %.8g,\n", (double)rf(header, 724));
  printf("  \"lever_x_m\": %.8g,\n", (double)rf(header, 420));
  printf("  \"lever_y_m\": %.8g,\n", (double)rf(header, 424));
  printf("  \"lever_z_m\": %.8g,\n", (double)rf(header, 760));
  printf("  \"vert_fov_deg\": %.8g,\n", (double)rf(header, 2032));
  printf("  \"vert_fov_alt_deg\": %.8g,\n", (double)rf(header, 1800));
  printf("  \"horiz_half_deg\": %.8g\n", (double)rf(header, 940));
  printf("}\n");

  free(data);
  operator delete[](header);
  return 0;
}
