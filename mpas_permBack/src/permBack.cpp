#include <iostream>
#include <fstream>
#include <vector>
#include "def.h"
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
	sprintf(nc_path, "%s/%s", output_root, filename); //    /fs/ess/PAS0027/MPAS1/Inter1/

	char bin_path[1024];
	sprintf(bin_path, "%s/%s_temperature_fake.bin", input_root, fileid.c_str());   //  /fs/ess/PAS0027/mpas_graph/test/
	
    loadMeshFromNetCDF(nc_path);

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

	fstream readFileRes;
    readFileRes.open(bin_path, ios::binary | ios::in);
	streampos fileSizeRes;
	readFileRes.seekg(0, std::ios::end);
	fileSizeRes = readFileRes.tellg();
	readFileRes.seekg(0, std::ios::beg);

	vector<double> fakeTemperature(temperature);
	vector<double> permedTemperature;
	permedTemperature.resize(fileSizeRes / sizeof(double));
	// read the data
	readFileRes.read((char*)(&permedTemperature[0]), fileSizeRes);

	for (int i = 0; i < indices.size(); i++) {
		fakeTemperature[indices[i]] = permedTemperature[i];
	}

	double loss = 0.0;
	for (int i = 0; i < fakeTemperature.size(); i++) {
		loss += abs(fakeTemperature[i] - temperature[i]);
	}
	cout << loss / indices.size() << endl;

	const size_t start_time_cell_vertLevel[3] = { 0, 0, 0 }, size_time_cell_vertLevel[3] = { Time, nCells, nVertLevels };
	NC_SAFE_CALL(nc_put_vara_double(ncid, varid_temperature, start_time_cell_vertLevel, size_time_cell_vertLevel, &fakeTemperature[0]));
	NC_SAFE_CALL(nc_close(ncid));
}
