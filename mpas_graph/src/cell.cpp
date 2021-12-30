#include "cell.h"

Cell::Cell(){
	real = false;
}

//Cell::Cell(float lat, float lon, float d, bool real) : lat(lat), lon(lon), d(d), real(real) {
//	x = cos(lat) * cos(lon);
//	y = cos(lat) * sin(lon);
//	z = sin(lat);
//}

Cell::Cell(float xx, float yy, float zz, float d, bool real) : x(xx), y(yy), z(zz), d(d), real(real) {
	lat = asin(z);
	lon = atan(y / x);
	if (lon < 0)
		lon += M_PI;
	if (y < 0)
		lon += M_PI;
}


