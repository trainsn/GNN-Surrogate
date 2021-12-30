#include "edge.h"

double cross(float x1, float y1, float x2, float y2) {
	return x1 * y2 - x2 * y1;
}

Edge::Edge(){
	deltaHighLat = deltaLowLat = deltaWest = deltaEast = 0.0;
	isHorizontal = isUp = isDown = false;
	weight = 0.0;
}

Edge::Edge(Cell left, Cell right, float weight){
	deltaHighLat = deltaLowLat = deltaWest = deltaEast = 0.0;
	isHorizontal = isUp = isDown = false;
	this->weight = weight;

	if (abs(left.d - right.d) < eps) {
		isHorizontal = true;

		bool toHighLat = (abs(right.lat) - abs(left.lat)) > 0;
		if (toHighLat) {
			deltaHighLat = abs(right.lat) - abs(left.lat);
		}
		else {
			deltaLowLat = abs(left.lat) - abs(right.lat);
		}

		float deltaX = right.x - left.x;
		float deltaY = right.y - left.y;
		bool toEast = cross(left.x, left.y, deltaX, deltaY) > 0;
		if (toEast) {
			deltaEast = sqrt(deltaX * deltaX + deltaY * deltaY);
		}
		else {
			deltaWest = sqrt(deltaX * deltaX + deltaY * deltaY);
		}

		//float deltaLen = sqrt(deltaHighLat * deltaHighLat + deltaLowLat * deltaLowLat + deltaWest * deltaWest + deltaEast * deltaEast);
		float deltaLen = deltaHighLat + deltaLowLat + deltaWest + deltaEast;
		deltaHighLat /= deltaLen;
		deltaLowLat /= deltaLen;
		deltaWest /= deltaLen;
		deltaEast /= deltaLen;
	}
	else if (right.d > left.d) {
		isDown = true;
	}
	else if (left.d > right.d) {
		isUp = true;
	}
}


