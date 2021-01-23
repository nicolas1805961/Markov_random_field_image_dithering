#include "minimize.hh"

namespace cmkv
{

  float get_ratio(std::uint8_t candidate, image<std::uint8_t> const &x, image<std::uint8_t> const &y, unsigned int i, unsigned int j, unsigned int threshold)
  {
    std::uint8_t regularizer_denominator = 0;
    std::uint8_t regularizer_numerator = 0;
    std::uint8_t data_consistency_denominator = 0;
    std::uint8_t data_consistency_numerator = 0;

    if (i > 0)
    {
      regularizer_denominator += (x(j, i) != x(j, i-1));
      regularizer_numerator += (candidate != x(j, i-1));
    }
    if (i < x.height - 1)
    {
      regularizer_denominator += (x(j, i) != x(j, i+1));
      regularizer_numerator += (candidate != x(j, i+1));
    }
    if (j > 0)
    {
      regularizer_denominator += (x(j, i) != x(j-1, i));
      regularizer_numerator += (candidate != x(j-1, i));
    }
    else if (j < x.width - 1)
    {
      regularizer_denominator += (x(j, i) != x(j+1, i));
      regularizer_numerator += (candidate != x(j+1, i));
    }

    if (x(j, i) == 0)
      data_consistency_denominator = y(j, i);
    else
      data_consistency_denominator = 255 - y(j, i) + threshold;
    if (candidate == 0)
      data_consistency_numerator = y(j, i);
    else
      data_consistency_numerator = 255 - y(j, i) + threshold;

    float u_denominator = 1 * data_consistency_denominator + 0 * regularizer_denominator;
    float u_numerator = 1 * data_consistency_numerator + 0 * regularizer_numerator;

    return u_denominator - u_numerator;
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
    std::vector<std::vector<std::uint8_t>> kernel{
      {64, 128},
      {192, 0}
    };
    auto result = image<std::uint8_t>(img.width, img.height);
    Rand random;
    float T = 4;
    unsigned int count = 0;
    unsigned int iterations = 10000000;
    for (size_t iter = 0; iter < iterations; iter++)
    {
      unsigned int row = random.randint(0, img.height - 1);
      unsigned int col = random.randint(0, img.width - 1);
      int thresh = get_threshold_value(row, col, kernel);
      int candidate = random.randint(0, 1);
      float value = get_ratio(candidate, result, img, row, col, thresh);
      float p = expf(std::min(0.0f, value) / T);
      if (random.rand() < p)
      {
        count++;
        result(col, row) = candidate * 255;
      }
      T *= update_temp(static_cast<float>(iter) / static_cast<float>(iterations));
    }
    // FIXME: replace the code below by yours!
    /*auto result = image<std::uint8_t>(img.width, img.height);
    for (std::size_t y = 0; y < img.height; y++)
      for (std::size_t x = 0; x < img.width; x++)
	result(x, y) = img(x, y) > 127u ? 255u : 0u;*/
    return result;
  }

} // cmkv
