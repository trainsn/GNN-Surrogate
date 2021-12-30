#pragma once
#include <fstream>
#include <vector>
#include <Eigen/Sparse>
#include "cnpy.h"
#include "cell.h"
#include "edge.h"

using namespace std;
using namespace Eigen;

class Graph {
public:
	int size;
	int nDepths, nCells;
	vector<Cell> nodes;
	float avgSqrDistH, avgSqrDistV;
	static const int edgeNumAttr = 8;
	SparseMatrix<float, RowMajor> A[edgeNumAttr];	// 0 - deltaHighLat, 1 - deltaLowLat, 2 - deltaWest, 3 - deltaEast
										// 4 - isHorizontal, 5 - isUp, 6 - isDown, 7 - weight 
	SparseMatrix<float, RowMajor> upAsgn, avgPoolAsgn;
	Graph();
	Graph(vector<Cell> nodes, int nVertLevels);
	void Serialize(SparseMatrix<float, RowMajor>& m, const string& filename);
	void Deserialize(SparseMatrix<float, RowMajor>& m, const string& filename);
	void SaveMatrix(SparseMatrix<float, RowMajor> m[edgeNumAttr], const string& idxFilename, const string& valueFilename);
	void SaveMatrix(SparseMatrix<float, RowMajor>& m, const string& idxFilename, const string& valueFilename);
	void SaveNodes(const string& filename);
};