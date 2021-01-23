#include <cstring>
#include <iostream>
#include <stdexcept>

#include "image.hh"


namespace cmkv
{

  bool operator==(const rgb8_t& lhs, const rgb8_t& rhs)
  {
    return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b;
  }

  image<std::uint8_t> convert_gray(const image<rgb8_t>& img)
  {
    auto result = image<std::uint8_t>(img.width, img.height);
    for (std::size_t y = 0; y < img.height; ++y)
      for (std::size_t x = 0; x < img.width; ++x)
	{
	  float f = 0.2989f * float(img(x, y).r)
	    + 0.5870f * float(img(x, y).g)
	    + 0.1140f * float(img(x, y).b);
	  result(x, y) = static_cast<std::uint8_t>(f + 0.5f);
	}
    return result;
  }

  image<rgb8_t> convert_rgb(const image<std::uint8_t>& img)
  {
    auto result = image<rgb8_t>(img.width, img.height);
    for (std::size_t y = 0; y < img.height; ++y)
      for (std::size_t x = 0; x < img.width; ++x)
	{
	  auto v = img(x, y);
	  result(x, y) = {v, v, v};
	}
    return result;
  }
  
  namespace io
  {

    image<rgb8_t> read(const char* filename)
    {
      FIBITMAP* bitmap = FreeImage_Load(FIF_PNG, filename, 0);
      if (not bitmap)
        throw std::runtime_error("Image could not been read");

      const auto width  = FreeImage_GetWidth(bitmap);
      const auto height = FreeImage_GetHeight(bitmap);

      auto img = image<rgb8_t>(width, height);

      for (std::size_t y = 0; y < height; ++y)
      {
        RGBTRIPLE* line = (RGBTRIPLE*)FreeImage_GetScanLine(bitmap, y);
        for (std::size_t x = 0; x < width; ++x)
        {
	  auto& pix = img(x, y);
	  pix.r = line[x].rgbtRed;
          pix.g = line[x].rgbtGreen;
          pix.b = line[x].rgbtBlue;
        }
      }

      FreeImage_Unload(bitmap);
      return img;
    }

    void write(const char* filename, const image<rgb8_t>& img)
    {
      FIBITMAP* bitmap =
          FreeImage_AllocateT(FIT_BITMAP, img.width, img.height, 24);
      if (not bitmap)
        throw std::runtime_error("Could not allocate the new bitmap");

      for (std::size_t y = 0; y < img.height; ++y)
      {
        RGBTRIPLE* line = (RGBTRIPLE*)FreeImage_GetScanLine(bitmap, y);
        for (std::size_t x = 0; x < img.width; ++x)
        {
	  const auto& pix = img(x, y);
          line[x].rgbtRed   = pix.r;
          line[x].rgbtGreen = pix.g;
          line[x].rgbtBlue  = pix.b;
        }
      }

      const auto e = FreeImage_Save(FIF_PNG, bitmap, filename);
      if (not e)
        throw std::runtime_error("Could not save the bitmap");

      FreeImage_Unload(bitmap);
    }

    void write(const char* filename, const image<std::uint8_t>& img)
    {
      write(filename, convert_rgb(img));
    }
    
  } // io

} // cmkv
