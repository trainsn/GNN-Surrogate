#pragma once

#include "cell.h"

using namespace std;

class Edge {
public:
	float deltaHighLat, deltaLowLat, deltaWest, deltaEast;
	bool isHorizontal, isUp, isDown;
	float weight;

	Edge();
	Edge(Cell left, Cell right, float weight);
};