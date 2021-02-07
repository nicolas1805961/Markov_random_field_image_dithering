#include "minimize.hh"
#include <fstream>

namespace cmkv
{

  float applyFunction(std::uint8_t candidate, image<std::uint8_t> const &x, image<std::uint8_t> const &y, unsigned int i, unsigned int j, unsigned int threshold)
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
      data_consistency = 255 - y(j, i) + threshold;

    return 1 * data_consistency + 1 * regularizer;
  }

  float update_temp(float r)
  {
    return pow(0.99, exp(8 * r));
  }

  int get_threshold_value(unsigned int row, unsigned int col, std::vector<std::vector<std::uint8_t>> const& kernel)
  {
    unsigned int i = row % kernel.size();
    unsigned int j = col % kernel.size();
    return kernel[i][j];
  }

  image<std::uint8_t> minimize(const image<std::uint8_t>& img)
  {
    std::vector<std::vector<std::uint8_t> > kernel{
        {0, 128, 32, 159},
        {191, 64, 223, 96},
        {48, 175, 16, 143},
        {239, 112, 207, 80}};
    auto result = image<std::uint8_t>(img.width, img.height);
    Rand random;
    float T = 4;
    unsigned int iterations = 1000000;
    std::ofstream myfile;
    myfile.open("temperature.txt");
    for (size_t iter = 0; iter < iterations; iter++)
    {
      myfile << T << "\n";
      unsigned int row = random.randint(0, img.height - 1);
      unsigned int col = random.randint(0, img.width - 1);
      int thresh = get_threshold_value(row, col, kernel);
      //int candidate = result(col, row) == 0 ? 1: 0;
      int candidate = random.randint(0, 1);
      float energyNewState = applyFunction(candidate, result, img, row, col, thresh);
      float energyCurrentState = applyFunction(result(col, row), result, img, row, col, thresh);
      float energy = energyNewState - energyCurrentState;
      float p = expf(std::min(0.0f, -energy) / T);
      if (random.rand() < p)
        result(col, row) = candidate * 255;
      T *= 0.99999;
      //T *= update_temp(static_cast<float>(iter) / static_cast<float>(iterations));
    }
    myfile.close();
    return result;
  }

} // cmkv
