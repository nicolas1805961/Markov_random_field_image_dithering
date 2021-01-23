#include "Rand.hh"

Rand::Rand() : generator(std::random_device()())
{}

float Rand::rand() 
{
    std::uniform_real_distribution<> distribution(0.0, 1.0);
    return distribution(generator);
}

int Rand::randint(int start, int stop) 
{
    std::uniform_int_distribution<> distribution(start, stop);
    return distribution(generator);
}

Rand::~Rand()
{
}