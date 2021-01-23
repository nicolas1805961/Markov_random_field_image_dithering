#include <iostream>

#include "image.hh"
#include "minimize.hh"


int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cerr << argv[0] << " input.png output.png\n";
    return EXIT_FAILURE;
  }

  FreeImage_Initialise();

  auto input = cmkv::convert_gray(cmkv::io::read(argv[1]));
  auto output = cmkv::minimize(input);
  cmkv::io::write(argv[2], output);

  FreeImage_DeInitialise();
}
