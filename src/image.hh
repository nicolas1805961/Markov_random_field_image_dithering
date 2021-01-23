#ifndef IMAGE_HH
#define IMAGE_HH

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <vector>
#include <FreeImage.h>
#include <math.h> 


namespace cmkv
{
  
  struct rgb8_t
  {
    rgb8_t(std::uint8_t c_r = 0u, std::uint8_t c_g = 0u, std::uint8_t c_b = 0u): r(c_r), g(c_g), b(c_b)
    {}

    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
  };

  bool operator==(const rgb8_t& lhs, const rgb8_t& rhs);

  
  template <typename T>
  struct image
  {
    image(unsigned c_width, unsigned c_height): width(c_width), height(c_height), data(std::make_shared<std::vector<T>>(c_width * c_height))
    {}

    image(const image&) = default;
    void operator=(const image&) = delete;
    
    const unsigned width;
    const unsigned height;
    std::shared_ptr<std::vector<T>> data;

    bool has(int x, int y) const noexcept
    {
      return x >= 0 and y >= 0 and x < static_cast<int>(width) and y < static_cast<int>(height);
    }

    T& operator()(int x, int y)
    {
      assert(has(x, y));
      return (*data)[y * width + x];
    }

    T operator()(int x, int y) const
    {
      assert(has(x, y));
      return (*data)[y * width + x];
    }

  };


  image<std::uint8_t> convert_gray(const image<rgb8_t>& img);
  image<rgb8_t> convert_rgb(const image<std::uint8_t>& img);

  
  namespace io
  {
    
    image<rgb8_t> read(const char* filename);

    void write(const char* filename, const image<rgb8_t>& img);
    void write(const char* filename, const image<std::uint8_t>& img);

  } // io

} // cmkv


#endif
