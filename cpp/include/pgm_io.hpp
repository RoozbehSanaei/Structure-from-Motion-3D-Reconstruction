#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sfm {

struct GrayImage {
  int w=0, h=0;
  std::vector<std::uint8_t> pix; // size w*h
  std::uint8_t& at(int x,int y){ return pix[y*w + x]; }
  std::uint8_t  at(int x,int y) const { return pix[y*w + x]; }
};

struct RGBImage {
  int w=0, h=0;
  std::vector<std::uint8_t> pix; // size 3*w*h, RGB
  std::uint8_t* at(int x,int y){ return &pix[3*(y*w + x)]; }
  const std::uint8_t* at(int x,int y) const { return &pix[3*(y*w + x)]; }
};

inline void skip_comments(std::istream& is){
  while(true){
    int c = is.peek();
    if (c=='#'){
      std::string line;
      std::getline(is, line);
      continue;
    }
    break;
  }
}

inline GrayImage read_pgm(const std::string& path){
  std::ifstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("Failed to open: " + path);
  std::string magic;
  f >> magic;
  if(magic != "P5") throw std::runtime_error("Only binary PGM (P5) supported: " + path);
  skip_comments(f);
  int w,h,maxv;
  f >> w; skip_comments(f);
  f >> h; skip_comments(f);
  f >> maxv;
  if(maxv != 255) throw std::runtime_error("Only 8-bit PGM supported: " + path);
  f.get(); // consume whitespace
  GrayImage im;
  im.w=w; im.h=h; im.pix.resize((size_t)w*(size_t)h);
  f.read(reinterpret_cast<char*>(im.pix.data()), (std::streamsize)im.pix.size());
  if(!f) throw std::runtime_error("PGM read failed: " + path);
  return im;
}

inline void write_ppm(const std::string& path, const RGBImage& im){
  std::ofstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("Failed to write: " + path);
  f << "P6\n" << im.w << " " << im.h << "\n255\n";
  f.write(reinterpret_cast<const char*>(im.pix.data()), (std::streamsize)im.pix.size());
}

inline RGBImage gray_to_rgb(const GrayImage& g){
  RGBImage out;
  out.w=g.w; out.h=g.h; out.pix.resize((size_t)3*out.w*out.h);
  for(int y=0;y<g.h;y++){
    for(int x=0;x<g.w;x++){
      const auto v = g.at(x,y);
      auto* p = &out.pix[3*(y*out.w + x)];
      p[0]=p[1]=p[2]=v;
    }
  }
  return out;
}

} // namespace sfm
