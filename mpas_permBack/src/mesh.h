#pragma once
#include <netcdf.h>
#include <vector>
#include <map>
#include "cell.h"
#include "def.h"

using namespace std;

size_t nCells, nEdges, nVertices, nVertLevels, maxEdges, vertexDegree, Time;
vector<double> latVertex, lonVertex, xVertex, yVertex, zVertex;
vector<Cell> cells;
vector<double> xyzCell, latCell, lonCell, bottomDepth;
vector<int> indexToVertexID, indexToCellID, indexToEdgeID;
vector<int> verticesOnEdge, cellsOnEdge,
cellsOnVertex, edgesOnVertex,
verticesOnCell, nEdgesOnCell, maxLevelCell;
vector<double> temperature, salinity, thickness, maxThickness;

map<int, int> vertexIndex, cellIndex;

int ncid;
int dimid_cells, dimid_edges, dimid_vertices, dimid_vertLevels, dimid_maxEdges,
dimid_vertexDegree, dimid_Time;
int varid_latVertex, varid_lonVertex, varid_xVertex, varid_yVertex, varid_zVertex,
varid_latCell, varid_lonCell, varid_xCell, varid_yCell, varid_zCell, varid_bottomDepth,
varid_edgesOnVertex, varid_cellsOnVertex,
varid_indexToVertexID, varid_indexToCellID, varid_indexToEdgeID,
varid_nEdgesOnCell, varid_cellsOncell, varid_verticesOnCell, varid_maxLevelCell,
varid_verticesOnEdge, varid_cellsOnEdge,
varid_temperature, varid_salinity, varid_thickness;

void loadMeshFromNetCDF(const string& filename) {
	NC_SAFE_CALL(nc_open(filename.c_str(), NC_WRITE, &ncid));

	NC_SAFE_CALL(nc_inq_dimid(ncid, "nCells", &dimid_cells));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "nEdges", &dimid_edges));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "nVertices", &dimid_vertices));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "nVertLevels", &dimid_vertLevels));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "maxEdges", &dimid_maxEdges));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "vertexDegree", &dimid_vertexDegree));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "Time", &dimid_Time));

	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_cells, &nCells));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_edges, &nEdges));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_vertices, &nVertices));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_vertLevels, &nVertLevels));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_maxEdges, &maxEdges));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_vertexDegree, &vertexDegree));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_Time, &Time));

	NC_SAFE_CALL(nc_inq_varid(ncid, "indexToVertexID", &varid_indexToVertexID));
	NC_SAFE_CALL(nc_inq_varid(ncid, "indexToCellID", &varid_indexToCellID));
	NC_SAFE_CALL(nc_inq_varid(ncid, "indexToEdgeID", &varid_indexToEdgeID));
	NC_SAFE_CALL(nc_inq_varid(ncid, "latCell", &varid_latCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "lonCell", &varid_lonCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "xCell", &varid_xCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "yCell", &varid_yCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "zCell", &varid_zCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "bottomDepth", &varid_bottomDepth));
	NC_SAFE_CALL(nc_inq_varid(ncid, "nEdgesOnCell", &varid_nEdgesOnCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "maxLevelCell", &varid_maxLevelCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "latVertex", &varid_latVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "lonVertex", &varid_lonVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "xVertex", &varid_xVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "yVertex", &varid_yVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "zVertex", &varid_zVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "edgesOnVertex", &varid_edgesOnVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnVertex", &varid_cellsOnVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnCell", &varid_cellsOncell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "verticesOnCell", &varid_verticesOnCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "verticesOnEdge", &varid_verticesOnEdge));
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnEdge", &varid_cellsOnEdge));
	NC_SAFE_CALL(nc_inq_varid(ncid, "temperature", &varid_temperature));
	NC_SAFE_CALL(nc_inq_varid(ncid, "salinity", &varid_salinity));
	NC_SAFE_CALL(nc_inq_varid(ncid, "layerThickness", &varid_thickness));

	const size_t start_cells[1] = { 0 }, size_cells[1] = { nCells };

	latCell.resize(nCells);
	lonCell.resize(nCells);
	indexToCellID.resize(nCells);
	nEdgesOnCell.resize(nCells);
	maxLevelCell.resize(nCells);
	bottomDepth.resize(nCells);

	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_latCell, start_cells, size_cells, &latCell[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_lonCell, start_cells, size_cells, &lonCell[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_indexToCellID, start_cells, size_cells, &indexToCellID[0]));
	for (int i = 0; i < nCells; i++) {
		cellIndex[indexToCellID[i]] = i;
		// fprintf(stderr, "%d, %d\n", i, indexToCellID[i]);
	}
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_nEdgesOnCell, start_cells, size_cells, &nEdgesOnCell[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_maxLevelCell, start_cells, size_cells, &maxLevelCell[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_bottomDepth, start_cells, size_cells, &bottomDepth[0]));

	std::vector<double> coord_cells;
	coord_cells.resize(nCells);
	xyzCell.resize(nCells * 3);
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_xCell, start_cells, size_cells, &coord_cells[0]));
	for (int i = 0; i < nCells; i++)
		xyzCell[i * 3] = coord_cells[i];
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_yCell, start_cells, size_cells, &coord_cells[0]));
	for (int i = 0; i < nCells; i++)
		xyzCell[i * 3 + 1] = coord_cells[i];
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_zCell, start_cells, size_cells, &coord_cells[0]));
	for (int i = 0; i < nCells; i++)
		xyzCell[i * 3 + 2] = coord_cells[i];

	const size_t start_vertices[1] = { 0 }, size_vertices[1] = { nVertices };
	latVertex.resize(nVertices);
	lonVertex.resize(nVertices);
	xVertex.resize(nVertices);
	yVertex.resize(nVertices);
	zVertex.resize(nVertices);
	indexToVertexID.resize(nVertices);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_indexToVertexID, start_vertices, size_vertices, &indexToVertexID[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_latVertex, start_vertices, size_vertices, &latVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_lonVertex, start_vertices, size_vertices, &lonVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_xVertex, start_vertices, size_vertices, &xVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_yVertex, start_vertices, size_vertices, &yVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_zVertex, start_vertices, size_vertices, &zVertex[0]));

	//for (int i = 0; i < nVertices; i++) {
	//	double x = max_rho * cos(latVertex[i]) * cos(lonVertex[i]);
	//	double y = max_rho * cos(latVertex[i]) * sin(lonVertex[i]);
	//	double z = max_rho * sin(latVertex[i]);
	//	assert(abs(x - xVertex[i]) < eps && abs(y - yVertex[i]) < eps && abs(z - zVertex[i]) < eps);
	//}

	for (int i = 0; i < nVertices; i++) {
		vertexIndex[indexToVertexID[i]] = i;
		// fprintf(stderr, "%d, %d\n", i, indexToVertexID[i]);
	}

	const size_t start_edges[1] = { 0 }, size_edges[1] = { nEdges };
	indexToEdgeID.resize(nEdges);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_indexToEdgeID, start_edges, size_edges, &indexToEdgeID[0]));

	const size_t start_edges2[2] = { 0, 0 }, size_edges2[2] = { nEdges, 2 };
	verticesOnEdge.resize(nEdges * 2);
	cellsOnEdge.resize(nEdges * 2);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_verticesOnEdge, start_edges2, size_edges2, &verticesOnEdge[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_cellsOnEdge, start_edges2, size_edges2, &cellsOnEdge[0]));

	//for (int i=0; i<nEdges; i++) 
	//   fprintf(stderr, "%d, %d\n", verticesOnEdge[i*2], verticesOnEdge[i*2+1]);

	const size_t start_vertex_cell[2] = { 0, 0 }, size_vertex_cell[2] = { nVertices, 3 };
	cellsOnVertex.resize(nVertices * 3);
	edgesOnVertex.resize(nVertices * 3);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_cellsOnVertex, start_vertex_cell, size_vertex_cell, &cellsOnVertex[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_edgesOnVertex, start_vertex_cell, size_vertex_cell, &edgesOnVertex[0]));

	const size_t start_cell_vertex[2] = { 0, 0 }, size_cell_vertex[2] = { nCells, maxEdges };
	verticesOnCell.resize(nCells * maxEdges);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_verticesOnCell, start_cell_vertex, size_cell_vertex, &verticesOnCell[0]));

	const size_t start_time_cell_vertLevel[3] = { 0, 0, 0 }, size_time_cell_vertLevel[3] = { Time, nCells, nVertLevels };
	temperature.resize(Time * nCells * nVertLevels);
	salinity.resize(Time * nCells * nVertLevels);
	thickness.resize(Time * nCells * nVertLevels);

	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_temperature, start_time_cell_vertLevel, size_time_cell_vertLevel, &temperature[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_salinity, start_time_cell_vertLevel, size_time_cell_vertLevel, &salinity[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_thickness, start_time_cell_vertLevel, size_time_cell_vertLevel, &thickness[0]));
	maxThickness.reserve(nVertLevels);
	for (int j = 0; j < nVertLevels; j++) {
		double maxThick = 0;
		for (int i = 0; i < nCells; i++) {
			if (thickness[i * nVertLevels + j] > maxThick)
				maxThick = thickness[i * nVertLevels + j];
		}
		maxThickness.push_back(maxThick);
	}

	//NC_SAFE_CALL(nc_close(ncid));

	fprintf(stderr, "%zu, %zu, %zu, %zu\n", nCells, nEdges, nVertices, nVertLevels);
}