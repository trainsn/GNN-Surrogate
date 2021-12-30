#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <map>
#include <assert.h>
#include "cnpy.h"
#include "cell.h"
#include "mesh.h"

using namespace std;

int main(int argc, char **argv) {
    char input_root[1024];
    sprintf(input_root, argv[1]);
    char output_root[1024];
    sprintf(output_root, argv[2]);
    
    char filename[1024];
	sprintf(filename, argv[3]);
	fprintf(stderr, "%s\n", argv[3]);

	string filename_s = filename;
	int pos_first_dash = filename_s.find("_");
	string fileid = filename_s.substr(0, pos_first_dash);
	
	char nc_path[1024];
	sprintf(nc_path, "%s/%s", input_root, filename);
	loadMeshFromNetCDF(nc_path);
	
	char npy_path[1024];
	sprintf(npy_path, "%s/%s_temperature.npy", output_root, fileid.c_str());

	cells.reserve(nCells * nVertLevels);
	for (int i = 0; i < nCells; i++) {
		double x = cos(latCell[i]) * cos(lonCell[i]);
		double y = cos(latCell[i]) * sin(lonCell[i]);
		double z = sin(latCell[i]);
		double depth = 0.0;
		for (int j = 0; j < nVertLevels; j++) {
			Cell cell = Cell(x, y, z, depth, true);
			if (j >= maxLevelCell[i])
				cell.real = false;
			assert(abs(cell.lat - latCell[i]) < eps && abs(cell.lon - lonCell[i]) < eps);
			depth += maxThickness[j];
			cells.push_back(cell);
		}
	}

	fstream readFile;
	readFile.open("../res/perm.dat", ios::binary | ios::in);
	streampos fileSize;
	readFile.seekg(0, std::ios::end);
	fileSize = readFile.tellg();
	readFile.seekg(0, std::ios::beg);

	vector<int> indices;
	indices.resize(fileSize / sizeof(int));
	// read the data:
	readFile.read((char*)(&indices[0]), fileSize);
	/*for (int i = 0; i < indices.size(); i++)
		assert(indices[i] == perm[i]);*/
	
	vector<Cell> cellsNew;
	cellsNew.reserve(indices.size());
	for (int i = 0; i < indices.size(); i++) {
		cellsNew.push_back(cells[indices[i]]);
	}
	vector<double> posCell;
	for (int i = 0; i < cellsNew.size(); i++) {
		posCell.push_back(cellsNew[i].x);
		posCell.push_back(cellsNew[i].y);
		posCell.push_back(cellsNew[i].z);
		posCell.push_back(cellsNew[i].lat);
		posCell.push_back(cellsNew[i].lon);
		posCell.push_back(cellsNew[i].d);
		posCell.push_back(cellsNew[i].real);
	}
	// save it to file 
// 	cnpy::npy_save("../res/latLonNodes0.npy", &posCell[0], { (size_t)(posCell.size() / 7), (size_t)7 }, "w");

	vector<double> temperatureNew;
	temperatureNew.reserve(indices.size());
	for (int i = 0; i < indices.size(); i++) {
		temperatureNew.push_back(temperature[indices[i]]);
	}
	cnpy::npy_save(npy_path, &temperatureNew[0], { (size_t)(temperatureNew.size())}, "w");

	for (int i = 0; i < indices.size(); i++) {
		assert(abs(temperatureNew[i] - temperature[indices[i]]) < eps );
	}
}