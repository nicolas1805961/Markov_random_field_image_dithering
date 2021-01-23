#pragma once

#include <random>
#include <iostream>

class Rand
{
private:
    std::mt19937 generator;

public :
    Rand();
    float rand();
    int randint(int start, int stop);
    ~Rand();
};


