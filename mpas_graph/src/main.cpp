#include <stdlib.h>
#include <netcdf.h>
#include <vector>
#include <map>
#include <queue>
#include <fstream>
#include <Eigen/Sparse>
#include "def.h"
#include "graph.h"
#include "coarsening.h"

using namespace std;
using namespace Eigen;

size_t nCells, nEdges, nVertices, nVertLevels, maxEdges, vertexDegree, Time;
vector<double> latVertex, lonVertex, xVertex, yVertex, zVertex;
vector<Cell> cells;
vector<double> latCell, lonCell, bottomDepth;
vector<int> indexToVertexID, indexToCellID, indexToEdgeID;
vector<int> verticesOnEdge, cellsOnEdge,
cellsOnVertex, edgesOnVertex,
cellsOnCell, verticesOnCell, nEdgesOnCell, maxLevelCell;
vector<double> thickness, maxThickness;

map<int, int> vertexIndex, cellIndex;

const int minNumEdges = 1;
const int maxNumEdges = 7;
const double PI = 3.14159265358979323846;

void loadMeshFromNetCDF(const string& filename) {
	int ncid;
	int dimid_cells, dimid_edges, dimid_vertices, dimid_vertLevels, dimid_maxEdges,
		dimid_vertexDegree, dimid_Time;
	int varid_latVertex, varid_lonVertex, varid_xVertex, varid_yVertex, varid_zVertex,
		varid_latCell, varid_lonCell, varid_xCell, varid_yCell, varid_zCell, varid_bottomDepth,
		varid_edgesOnVertex, varid_cellsOnVertex,
		varid_indexToVertexID, varid_indexToCellID, varid_indexToEdgeID,
		varid_nEdgesOnCell, varid_cellsOnCell, varid_verticesOnCell, varid_maxLevelCell,
		varid_verticesOnEdge, varid_cellsOnEdge, varid_thickness;

	NC_SAFE_CALL(nc_open(filename.c_str(), NC_NOWRITE, &ncid));

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
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnCell", &varid_cellsOnCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "verticesOnCell", &varid_verticesOnCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "verticesOnEdge", &varid_verticesOnEdge));
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnEdge", &varid_cellsOnEdge));
	NC_SAFE_CALL(nc_inq_varid(ncid, "restingThickness", &varid_thickness));

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
		//fprintf(stderr, "%d, %d\n", i, indexToCellID[i]);
	}
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_nEdgesOnCell, start_cells, size_cells, &nEdgesOnCell[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_maxLevelCell, start_cells, size_cells, &maxLevelCell[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_bottomDepth, start_cells, size_cells, &bottomDepth[0]));
	/*freopen("depth.csv", "w", stdout);
	for (int i = 0; i < nCells; i++) {
		printf("%.2lf,%d,%.2lf\n", bottomDepth[i], maxLevelCell[i], bottomDepth[i] / maxLevelCell[i]);
	}
	fclose(stdout);*/

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
	cellsOnCell.resize(nCells * maxEdges);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_verticesOnCell, start_cell_vertex, size_cell_vertex, &verticesOnCell[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_cellsOnCell, start_cell_vertex, size_cell_vertex, &cellsOnCell[0]));

	const size_t start_cell_vertLevel[2] = { 0, 0 }, size_cell_vertLevel[2] = { nCells, nVertLevels };
	thickness.resize(nCells * nVertLevels);

	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_thickness, start_cell_vertLevel, size_cell_vertLevel, &thickness[0]));
	maxThickness.reserve(nVertLevels);
	for (int j = 0; j < nVertLevels; j++) {
		float maxThick = 0; 
		for (int i = 0; i < nCells; i++) {
			if (thickness[i * nVertLevels + j] > maxThick)
				maxThick = thickness[i * nVertLevels + j];
		}
		maxThickness.push_back(maxThick);
	}

	NC_SAFE_CALL(nc_close(ncid));

	fprintf(stderr, "%zu, %zu, %zu, %zu\n", nCells, nEdges, nVertices, nVertLevels);
}

int main(int argc, char **argv) {
	loadMeshFromNetCDF(argv[1]); // such as /fs/project/PAS0027/MPAS/Results/0031_1.00949.nc

	vector<int> indicesByBFS;	// stores the original index by edge order
	indicesByBFS.reserve(nCells);
	map<int, int> BFSOrdFromIndices;

	vector<bool> visited(nCells, false);
	queue<int> q;
	// first add all boundary cells 
	for (int i = 0; i < nCells; i++) {
		int idx1 = i;
		for (int j = 0; j < nEdgesOnCell[idx1]; j++) {
			if (!cellsOnCell[idx1 * maxEdges + j]) {
				q.push(i);
				visited[i] = true;
				indicesByBFS.push_back(i);
				break;
			}
		}	
	}

	while (!q.empty()) {
		int idx1 = q.front();
		q.pop();
		for (int j = 0; j < nEdgesOnCell[idx1]; j++) {
			if (cellsOnCell[idx1 * maxEdges + j]) {
				int idx2 = cellIndex[cellsOnCell[idx1 * maxEdges + j]];	// original index 
				if (!visited[idx2]) {
					q.push(idx2);
					visited[idx2] = true;
					indicesByBFS.push_back(idx2);
				}
			}
		}
	}
	assert(indicesByBFS.size() == nCells);

	for (int i = 0; i < nCells; i++) {
		int idx = indicesByBFS[i];
		BFSOrdFromIndices[idx] = i;
	}

// 	nVertLevels = 1;
	cells.reserve(nCells * nVertLevels);
	for (int i = 0; i < nCells; i++) {
		int idx = indicesByBFS[i];	// original index 
		float x = cos(latCell[idx]) * cos(lonCell[idx]);
		float y = cos(latCell[idx]) * sin(lonCell[idx]);
		float z = sin(latCell[idx]);
		float depth = 0.0;
		for (int j = 0; j < nVertLevels; j++) {
			Cell cell = Cell(x, y, z, depth, true);
			if (j >= maxLevelCell[idx])
				cell.real = false;
			assert(abs(cell.lat - latCell[idx]) < eps && abs(cell.lon - lonCell[idx]) < eps);
			depth += maxThickness[j];
			cells.push_back(cell);
		}
	}

	Graph oriGraph = Graph(cells, nVertLevels);
	for (int i =0; i < oriGraph.edgeNumAttr; i++)
		oriGraph.A[i].reserve(VectorXi::Constant(nCells * nVertLevels, maxEdges + 2));

	oriGraph.avgSqrDistH = 0.0;
	int totalEdgesH = 0;
	// add horizontal edges 
	for (int i = 0; i < nCells; i++) {
		int idx1 = indicesByBFS[i];	// original index 
		assert(BFSOrdFromIndices[idx1] == i);
		int mtIdx1 = i * nVertLevels;
		for (int k = 0; k < nEdgesOnCell[idx1]; k++) {
			if (cellsOnCell[idx1 * maxEdges + k]) {
				int idx2 = cellIndex[cellsOnCell[idx1 * maxEdges + k]];	// original index
				if (visited[idx2]) {
					int mtIdx2 = BFSOrdFromIndices[idx2] * nVertLevels;
					float sqrDistH = (cells[mtIdx1].x - cells[mtIdx2].x) * (cells[mtIdx1].x - cells[mtIdx2].x) +
						(cells[mtIdx1].y - cells[mtIdx2].y) * (cells[mtIdx1].y - cells[mtIdx2].y) +
						(cells[mtIdx1].z - cells[mtIdx2].z) * (cells[mtIdx1].z - cells[mtIdx2].z);
					oriGraph.avgSqrDistH += sqrDistH;
					totalEdgesH++;
					for (int j = 0; j < nVertLevels; j++) {
						if (cells[mtIdx1 + j].real && cells[mtIdx2 + j].real){
							Edge edge = Edge(cells[mtIdx1], cells[mtIdx2], sqrDistH);
							oriGraph.A[0].insert(mtIdx1 + j, mtIdx2 + j) = edge.deltaHighLat;
							oriGraph.A[1].insert(mtIdx1 + j, mtIdx2 + j) = edge.deltaLowLat;
							oriGraph.A[2].insert(mtIdx1 + j, mtIdx2 + j) = edge.deltaWest;
							oriGraph.A[3].insert(mtIdx1 + j, mtIdx2 + j) = edge.deltaEast;
							oriGraph.A[4].insert(mtIdx1 + j, mtIdx2 + j) = edge.isHorizontal;
							oriGraph.A[5].insert(mtIdx1 + j, mtIdx2 + j) = edge.isUp;
							oriGraph.A[6].insert(mtIdx1 + j, mtIdx2 + j) = edge.isDown;
							oriGraph.A[7].insert(mtIdx1 + j, mtIdx2 + j) = edge.weight;
						}							
					}
				}
			}
		}
	}
	oriGraph.avgSqrDistH /= totalEdgesH;

	for (int k = 0; k < oriGraph.A[7].outerSize(); k++) {
		for (SparseMatrix<float, RowMajor>::InnerIterator it(oriGraph.A[7], k); it; ++it) {
			int row = it.row();
			int col = it.col();
			int depthLayer1 = row % nVertLevels;
			int depthLayer2 = col % nVertLevels;
			float weight;
			if (depthLayer1 == depthLayer2) {
				float value = it.value();
				weight = exp(-value / (4.0 * oriGraph.avgSqrDistH));
			}
			it.valueRef() = weight;
		}
	}

	// add vertical edges 
	oriGraph.avgSqrDistV = 0.0;
	vector<float> sqrDistV(nVertLevels);
	for (int j = 0; j < nVertLevels; j++) {
		oriGraph.avgSqrDistV += maxThickness[j] * maxThickness[j];
	}
	oriGraph.avgSqrDistV /= nVertLevels;

	for (int j = 0; j < nVertLevels - 1; j++) {
		float weight = exp(-maxThickness[j] * maxThickness[j] / (4.0 * oriGraph.avgSqrDistV));
		for (int i = 0; i < nCells; i++) {
			int mtIdx1 = i * nVertLevels + j;
			int mtIdx2 = i * nVertLevels + j + 1;
			if (cells[mtIdx1].real && cells[mtIdx2].real) {
				Edge edge1 = Edge(cells[mtIdx1], cells[mtIdx2], weight);
				oriGraph.A[0].insert(mtIdx1, mtIdx2) = edge1.deltaHighLat;
				oriGraph.A[1].insert(mtIdx1, mtIdx2) = edge1.deltaLowLat;
				oriGraph.A[2].insert(mtIdx1, mtIdx2) = edge1.deltaWest;
				oriGraph.A[3].insert(mtIdx1, mtIdx2) = edge1.deltaEast;
				oriGraph.A[4].insert(mtIdx1, mtIdx2) = edge1.isHorizontal;
				oriGraph.A[5].insert(mtIdx1, mtIdx2) = edge1.isUp;
				oriGraph.A[6].insert(mtIdx1, mtIdx2) = edge1.isDown;
				oriGraph.A[7].insert(mtIdx1, mtIdx2) = edge1.weight;

				Edge edge2 = Edge(cells[mtIdx2], cells[mtIdx1], weight);
				oriGraph.A[0].insert(mtIdx2, mtIdx1) = edge2.deltaHighLat;
				oriGraph.A[1].insert(mtIdx2, mtIdx1) = edge2.deltaLowLat;
				oriGraph.A[2].insert(mtIdx2, mtIdx1) = edge2.deltaWest;
				oriGraph.A[3].insert(mtIdx2, mtIdx1) = edge2.deltaEast;
				oriGraph.A[4].insert(mtIdx2, mtIdx1) = edge2.isHorizontal;
				oriGraph.A[5].insert(mtIdx2, mtIdx1) = edge2.isUp;
				oriGraph.A[6].insert(mtIdx2, mtIdx1) = edge2.isDown;
				oriGraph.A[7].insert(mtIdx2, mtIdx1) = edge2.weight;
			}		
		}
	}
	for (int i = 0; i < oriGraph.edgeNumAttr; i++)
		assert(oriGraph.A[i].outerSize() == nCells * nVertLevels);
	
	//oriGraph.Serialize(oriGraph.A, "oriGraphAdjacency");
	//oriGraph.Deserialize(oriGraph.A, "oriGraphAdjacency");

	Coarsening coarsening = Coarsening(); 
	coarsening.coarsen(oriGraph, 9, 2);
	//coarsening.permTest();
	//cout << coarsening.graphs[6].A << endl;
	//cout << coarsening.graphs[6].upAsgn << endl;
	//cout << coarsening.graphs[5].avgPoolAsgn << endl;

	vector<int> perm(coarsening.numReals[0]);
	for (int i = 0; i < coarsening.numReals[0]; i++) {
		int ordInCells = coarsening.perms[0][i];
		int ordInBFS = ordInCells / nVertLevels;
		int ordRawH = indicesByBFS[ordInBFS];
		int ordRawV = ordInCells % nVertLevels;
		perm[i] = ordRawH * nVertLevels + ordRawV;
	}
	cnpy::npy_save("../res/perm.npy", &perm[0], { (size_t)(perm.size()) }, "w");
	fstream writeFile;
	writeFile.open("../res/perm.dat", ios::binary | ios::out);
	writeFile.write((const char *)&perm[0], sizeof(int) * perm.size());

	cnpy::npy_save("../res/graphSizes.npy", &coarsening.numReals[0], { (size_t)(coarsening.numReals.size()) }, "w");
}																														
