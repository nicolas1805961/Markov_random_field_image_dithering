#include "minimize.hh"
#include <fstream>

namespace cmkv
{

  int get_threshold_value(unsigned int row, unsigned int col, std::vector<std::vector<std::uint8_t> > const &kernel)
  {
    unsigned int i = row % kernel.size();
    unsigned int j = col % kernel.size();
    return kernel[i][j];
  }

  float applyFunction(std::uint8_t candidate, image<std::uint8_t> const &x, image<std::uint8_t> const &y, unsigned int i, unsigned int j, std::vector<std::vector<std::uint8_t> > const &kernel)
  {
    std::uint8_t regularizer = 0;
    std::uint8_t data_consistency = 0;

    if (i > 0)
      regularizer += (candidate != x(j, i-1));
    if (i < x.height - 1)
      regularizer += (candidate != x(j, i+1));
    if (j > 0)
      regularizer += (candidate != x(j-1, i));
    if (j < x.width - 1)
      regularizer += (candidate != x(j+1, i));

    if (candidate == 0)
      data_consistency = y(j, i);
    else
      data_consistency = 255 - y(j, i) + get_threshold_value(i, j, kernel);

    return 1 * data_consistency + 0 * regularizer;
  }

  image<std::uint8_t> minimize(const image<std::uint8_t>& img)
  {
    /*std::vector<std::vector<std::uint8_t> > kernel{
        {0, 128},
        {192, 64}};
    std::vector<std::vector<std::uint8_t> > kernel{
        {0, 128, 32, 160},
        {192, 64, 224, 96},
        {48, 176, 16, 144},
        {240, 112, 208, 80}};*/
    std::vector<std::vector<std::uint8_t> > kernel{
        {0, 128, 32, 160, 8, 136, 40, 168},
        {192, 64, 224, 96, 200, 72, 232, 104},
        {48, 176, 16, 144, 56, 184, 24, 152},
        {240, 112, 208, 80, 248, 120, 216, 88},
        {12, 140, 44, 172, 4, 132, 36, 164},
        {204, 76, 236, 108, 196, 68, 228, 100},
        {60, 188, 28, 156, 52, 180, 20, 148},
        {252, 124, 220, 92, 244, 116, 212, 84}};
    auto result = image<std::uint8_t>(img.width, img.height);
    Rand random;
    float T = 4;
    unsigned int iterations = 1000000;
    for (size_t iter = 0; iter < iterations; iter++)
    {
      unsigned int row = random.randint(0, img.height - 1);
      unsigned int col = random.randint(0, img.width - 1);
      int candidate = random.randint(0, 1);
      float energyNewState = applyFunction(candidate, result, img, row, col, kernel);
      float energyCurrentState = applyFunction(result(col, row), result, img, row, col, kernel);
      float energy = energyNewState - energyCurrentState;
      float p = expf(std::min(0.0f, -energy) / T);
      if (random.rand() < p)
        result(col, row) = candidate * 255;
      T *= 0.99999;
    }
    return result;
  }

} // cmkv
