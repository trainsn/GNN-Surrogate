#pragma once
#include <stdio.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <Eigen/Sparse>
#include "graph.h"

using namespace std;
using namespace Eigen;

class Coarsening {
private:
	enum coarsenTypeEnum {HORIZONTAL, VERTICAL};
	vector<vector<int>> parents;
	map<int, vector<vector<int>>> inPerms;
	vector<coarsenTypeEnum> coarsenType;
	int nCells;
	void metis(Graph graph, int levels, int sub_levels);
	void metis_one_level(Graph& graph, vector<int>& cluster_id, Graph& coarsedGraph);
	void vertical_one_level(Graph& graph, vector<int>& cluster_id, Graph& coarsedGraph);
	void compute_perm(int sub_levels);
	void delete_fake_nodeA();
	void delete_fake_asgn();
	void perm_data(vector<float>& x, vector<int>& indices, vector <float>& xNew, int numReal);
	void perm_data(vector<Cell> & x, vector<int>& indices, vector<Cell>& xNew, int numReal);
	void perm_adjacency(SparseMatrix<float, RowMajor>& A, vector<int>& indices, int numReal);
	float EuclideanDist(int graphId, int front, int back);
public:
	vector<Graph> graphs;	// from fine to coarse
	vector<vector<int>> perms;
	vector<int> numReals;
	Coarsening();
	void coarsen(Graph graph, int levels, int sub_levels);
	void permTest();
};