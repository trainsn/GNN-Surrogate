#pragma once
#define _USE_MATH_DEFINES

#include <cmath>
const float eps = 1e-4;

class Cell {
public:
	float x, y, z;
	float lat, lon;
	float d;
	bool real;
	Cell();
	//Cell(float lat, float lon, float d, bool real);
	Cell(float xx, float yy, float zz, float d, bool real);
};