#include "coarsening.h"

Coarsening::Coarsening(){
	inPerms[1] = { 
		{0} 
	};
	inPerms[2] = {
		{1, 0}, {0, 1}
	};
	inPerms[3] = {
		{1, 2, 0}, {2, 1, 0},
		{0, 2, 1}, {2, 0, 1},
		{0, 1, 2}, {1, 0, 2}
	};
	inPerms[4] = {
		{1, 2, 3, 0}, {1, 3, 2, 0}, {2, 1, 3, 0}, {2, 3, 1, 0}, {3, 1, 2, 0}, {3, 2, 1, 0},
		{0, 2, 3, 1}, {0, 3, 2, 1}, {2, 0, 3, 1}, {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 2, 0, 1},
		{0, 1, 3, 2}, {0, 3, 1, 2}, {1, 0, 3, 2}, {1, 3, 0, 2}, {3, 0, 1, 2}, {3, 1, 0, 2},
		{0, 1, 2, 3}, {0, 2, 1, 3}, {1, 0, 2, 3}, {1, 2, 0, 3}, {2, 0, 1, 3}, {2, 1, 0, 3}
	};
}

void Coarsening::coarsen(Graph graph, int levels, int sub_levels){
	cout << "start metis" << endl;
	metis(graph, levels, sub_levels);
	if (levels) {
		cout << "start compute permutation" << endl;
		compute_perm(sub_levels);
	}
		
	cout << "start delete fake nodes and edges" << endl;
	delete_fake_nodeA();
	for (int i = 0; i < graphs.size(); i++) {
		if (levels) {
			vector<Cell> tmpNodes;
			perm_data(graphs[i].nodes, perms[i], tmpNodes, numReals[i]);
			graphs[i].nodes = tmpNodes;

			for (int j = 0; j < graphs[i].edgeNumAttr; j++)
				perm_adjacency(graphs[i].A[j], perms[i], numReals[i]);
		}
	}

	cout << "start delete fake items in assignment matrix" << endl;
	delete_fake_asgn();

	for (int i = 0; i < graphs.size(); i++) {
		int M = graphs[i].A[0].rows();
		
		int MNew = graphs[i].A[0].rows();
		int nnz = graphs[i].A[0].nonZeros();

		graphs[i].size = MNew;
		printf("Layer %d: M_%d = |V| = %d nodes, |E| = %d edges, avgDeg = %.2lf\n", i, i, MNew, nnz, (float)nnz  / M);

		//save the nodes to file 
		stringstream ss;
		ss << "../res/latLonNodes" << i << ".npy";
		string filename = ss.str();
		graphs[i].SaveNodes(filename);
		// save the adjacency matrix to file
		ss.str("");
		ss << "../res/adjacencyIdx" << i << ".npy"; 
		string idxFilename = ss.str();
		ss.str("");
		ss << "../res/adjacencyValue" << i << ".npy";
		string valueFilename = ss.str();
		graphs[i].SaveMatrix(graphs[i].A, idxFilename, valueFilename);
		// save the upSample assignment matrix to file 
		if (i) {
			ss.str("");
			ss << "../res/upAsgnIdx" << i << ".npy";
			idxFilename = ss.str();
			ss.str("");
			ss << "../res/upAsgnValue" << i << ".npy";
			valueFilename = ss.str();
			graphs[i].SaveMatrix(graphs[i].upAsgn, idxFilename, valueFilename);
		}
		// save the avgPool assignment matrix to file 
		if (i < graphs.size() - 1) {
			ss.str("");
			ss << "../res/avgPoolAsgnIdx" << i << ".npy";
			idxFilename = ss.str();
			ss.str("");
			ss << "../res/avgPoolAsgnValue" << i << ".npy";
			valueFilename = ss.str();
			graphs[i].SaveMatrix(graphs[i].avgPoolAsgn, idxFilename, valueFilename);
		}
	}
}

//Coarsen a graph multiple times using the METIS algorithm.
//
//INPUT
//W : symmetric sparse weight(adjacency) matrix
//levels : the number of coarsened graphs
//
//OUTPUT
//graph[0] : original graph of size N_1
//graph[2] : coarser graph of size N_2 < N_1
//graph[levels] : coarsest graph of Size N_levels < ... < N_2 < N_1
//parents[i] is a vector of size N_i with entries ranging from 1 to N_{ i + 1 } which indicate the parents in the coarser graph[i + 1]
//
//NOTE
//if "graph" is a list of length k, then "parents" will be a list of length k - 1
void Coarsening::metis(Graph graph, int levels, int sub_levels){
	graphs.push_back(graph);
	
	for (int i = 0; i < levels; i++) {
		cout << "horizontal coarsen level " << i << endl;
		// horizontal coarsen 
		vector<int> parentH(graph.nCells);
		for (int j = 0; j < sub_levels; j++) {
			vector<int> cluster_id(graph.nCells);
			Graph coarsedGraph = Graph();
			metis_one_level(graph, cluster_id, coarsedGraph);
			graph = coarsedGraph;
			if (j == sub_levels - 1) {
				graphs.push_back(coarsedGraph);
			}
			if (!j)
				parentH = cluster_id;
			else {
				for (int k = 0; k < parentH.size(); k++) {
					parentH[k] = cluster_id[parentH[k]];
				}
			}			
		}
		vector<int> parentTmp;
		parentTmp.reserve(parentH.size() * graph.nDepths);
		for (int k = 0; k < parentH.size(); k++) {
			for (int j = 0; j < graph.nDepths; j++) {
				parentTmp.push_back(parentH[k] * graph.nDepths + j);
			}
		}
		parents.push_back(parentTmp);
		coarsenType.push_back(HORIZONTAL);

		// vertical coarsen 
		if (i % 3 != 2 && i < 9)  	// EC60to30 equator patch only need 6 verical coarsen
		{
			cout << "vertical coarsen level " << i << endl;
			vector<int> cluster_id(graph.nDepths);
			Graph coarsedGraph = Graph();
			vertical_one_level(graph, cluster_id, coarsedGraph);
			graphs.push_back(coarsedGraph);
			graph = coarsedGraph;

			vector<int> parentV(graph.nDepths);
			parentV = cluster_id;
			vector<int>().swap(parentTmp);
			parentTmp.reserve(parentV.size() *  graph.nCells);
			for (int k = 0; k < graph.nCells; k++) {
				for (int j = 0; j < parentV.size(); j++) {
					parentTmp.push_back(k * graph.nDepths + parentV[j]);
				}
			}
			parents.push_back(parentTmp);
			coarsenType.push_back(VERTICAL);
		}	
	}
}

void Coarsening::metis_one_level(Graph& graph, vector<int>& cluster_id, Graph& coarsedGraph) {
	vector<int> rr, cc;
	vector<float> vv;
	for (int k = 0; k < graph.A[graph.edgeNumAttr - 1].outerSize(); k += graph.nDepths) {		// we only need to consider the surface level, rest are the same 
		for (SparseMatrix<float, RowMajor>::InnerIterator it(graph.A[graph.edgeNumAttr - 1], k); it; ++it) {
			int row = it.row();
			int col = it.col();
			int depthLayer1 = row % graph.nDepths;
			int depthLayer2 = col % graph.nDepths;
			if (depthLayer1 == depthLayer2) {
				// pair the vertices and construct the root vector
				rr.push_back(row / graph.nDepths);
				cc.push_back(col / graph.nDepths);
				vv.push_back(it.value());
			}
		}
	}
	
	int nnz = rr.size();
	int N = graph.nCells;

	vector<bool> marked(N);
	vector<int> rowStart(N), rowLength(N);

	int count = 0;
	int clusterCount = 0;
	
	for (int i = 0; i < nnz; i++) {
		while (rr[i] > count) {
			rowStart[count + 1] = i;
			count++;
		}
		rowLength[count]++;
	}
	//assert(count == N - 1);

	for (int i = 0; i < N; i++) {
		int idx1 = i;
		if (!marked[idx1]) {
			float wMax = -65535.0;
			int rs = rowStart[idx1];
			marked[idx1] = true;
			int bestNeighbor = -1;
			for (int j = 0; j < rowLength[idx1]; j++) {
				int idx2 = cc[rs + j];
				float tVal = -65535.0;
				if (!marked[idx2]) {
					tVal = vv[rs + j];
				}
				if (tVal > wMax) {
					wMax = tVal;
					bestNeighbor = idx2;
				}
			}
			cluster_id[idx1] = clusterCount;
			float newX = graph.nodes[idx1 * graph.nDepths].x;
			float newY = graph.nodes[idx1 * graph.nDepths].y;
			float newZ = graph.nodes[idx1 * graph.nDepths].z;
			/*float newLat = graph.nodes[idx1 * graph.nDepths].lat;
			float newLon = graph.nodes[idx1 * graph.nDepths].lon;*/

			if (bestNeighbor > -1) {
				assert(graph.nodes[idx1 * graph.nDepths].real);
				assert(graph.nodes[bestNeighbor * graph.nDepths].real);
				cluster_id[bestNeighbor] = clusterCount;
				marked[bestNeighbor] = true;
				newX = (newX + graph.nodes[bestNeighbor * graph.nDepths].x) / 2;
				newY = (newY + graph.nodes[bestNeighbor * graph.nDepths].y) / 2;
				newZ = (newZ + graph.nodes[bestNeighbor * graph.nDepths].z) / 2;
				float newLen = sqrt(newX * newX + newY * newY + newZ * newZ);
				newX /= newLen;
				newY /= newLen;
				newZ /= newLen;
				/*newLat = (newLat + graph.nodes[bestNeighbor].lat) / 2;
				newLon = (newLon + graph.nodes[bestNeighbor].lon) / 2;*/
			}
			else {
				//cout << idx1 << ",";
			}
			for (int j = 0; j < graph.nDepths; j++) {
				int idx1 = i * graph.nDepths + j;
				int idx2 = bestNeighbor * graph.nDepths + j;
				float newD = graph.nodes[idx1].d;
				bool newReal = graph.nodes[idx1].real;
				if (bestNeighbor > -1)
					newReal = newReal || graph.nodes[idx2].real;
				coarsedGraph.nodes.push_back(Cell(newX, newY, newZ, newD, newReal));
				//coarsedGraph.nodes.push_back(Cell(newLat, newLon, true));				
			}
			clusterCount++;
		}
	}	
	assert(clusterCount >= N / 2);
	//cout << endl;

	coarsedGraph.size = clusterCount * graph.nDepths;
	for (int i = 0; i < coarsedGraph.edgeNumAttr; i++)
		coarsedGraph.A[i].resize(coarsedGraph.size, coarsedGraph.size);
	coarsedGraph.nDepths = graph.nDepths;
	coarsedGraph.nCells = clusterCount;
	coarsedGraph.avgSqrDistV = graph.avgSqrDistV;
	vector<Triplet<float> > triple[coarsedGraph.edgeNumAttr];
	for (int i = 0; i < nnz; i++) {
		if (cluster_id[rr[i]] != cluster_id[cc[i]]) {
			int mtIdx1 = cluster_id[rr[i]] * coarsedGraph.nDepths;
			int mtIdx2 = cluster_id[cc[i]] * coarsedGraph.nDepths;
			Cell node1 = coarsedGraph.nodes[mtIdx1];
			Cell node2 = coarsedGraph.nodes[mtIdx2];
			float sqrDistH = (node1.x - node2.x) * (node1.x - node2.x) + (node1.y - node2.y) * (node1.y - node2.y) + (node1.z - node2.z) * (node1.z - node2.z);
			for (int j = 0; j < coarsedGraph.nDepths; j++) {
				int mtIdx1 = cluster_id[rr[i]] * coarsedGraph.nDepths + j;
				int mtIdx2 = cluster_id[cc[i]] * coarsedGraph.nDepths + j;
				if (coarsedGraph.nodes[mtIdx1].real && coarsedGraph.nodes[mtIdx2].real) {
					Edge edge = Edge(coarsedGraph.nodes[mtIdx1], coarsedGraph.nodes[mtIdx2], sqrDistH);
					triple[0].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaHighLat));
					triple[1].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaLowLat));
					triple[2].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaWest));
					triple[3].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaEast));
					triple[4].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.isHorizontal));
					triple[5].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.isUp));
					triple[6].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.isDown));
					triple[7].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.weight));
				}	
			}
		}
	}

	// prestore all the vertical edges into a vector from the previous graph 
	vector<float> edgeWeightsV(graph.nDepths - 1);
	for (int k = 0; k < graph.nDepths; k++) {
		for (SparseMatrix<float, RowMajor>::InnerIterator it(graph.A[coarsedGraph.edgeNumAttr - 1], k); it; ++it) {
			int row = it.row();
			int col = it.col();
			int depthLayer1 = row % graph.nDepths;
			int depthLayer2 = col % graph.nDepths;
			if (depthLayer2 - depthLayer1 == 1) {
				edgeWeightsV[depthLayer1] = it.value();
			}
		}
	}

	for (int i = 0; i < coarsedGraph.nCells; i++) {
		for (int j = 0; j < coarsedGraph.nDepths - 1; j++) {
			int mtIdx1 = i * coarsedGraph.nDepths + j;
			int mtIdx2 = i * coarsedGraph.nDepths + j + 1;
			if (coarsedGraph.nodes[mtIdx1].real && coarsedGraph.nodes[mtIdx2].real) {
				Edge edge1 = Edge(coarsedGraph.nodes[mtIdx1], coarsedGraph.nodes[mtIdx2], edgeWeightsV[j]);
				triple[0].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaHighLat));
				triple[1].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaLowLat));
				triple[2].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaWest));
				triple[3].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaEast));
				triple[4].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.isHorizontal));
				triple[5].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.isUp));
				triple[6].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.isDown));
				triple[7].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.weight));

				Edge edge2 = Edge(coarsedGraph.nodes[mtIdx2], coarsedGraph.nodes[mtIdx1], edgeWeightsV[j]);
				triple[0].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaHighLat));
				triple[1].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaLowLat));
				triple[2].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaWest));
				triple[3].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaEast));
				triple[4].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.isHorizontal));
				triple[5].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.isUp));
				triple[6].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.isDown));
				triple[7].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.weight));
			}
		}
	}
	for (int i=0; i < coarsedGraph.edgeNumAttr; i++)
		coarsedGraph.A[i].setFromTriplets(triple[i].begin(), triple[i].end(), [](const float&, const float &b) { return b; });
	
	int totalEdgesH = 0;
	coarsedGraph.avgSqrDistH = 0.0;
	for (int k = 0; k < coarsedGraph.A[coarsedGraph.edgeNumAttr - 1].outerSize(); k += coarsedGraph.nDepths)
		for (SparseMatrix<float, RowMajor>::InnerIterator it(coarsedGraph.A[coarsedGraph.edgeNumAttr - 1], k); it; ++it) {
			int row = it.row();
			int col = it.col();
			int depthLayer1 = row % coarsedGraph.nDepths;
			int depthLayer2 = col % coarsedGraph.nDepths;
			if (depthLayer1 == depthLayer2) {
				float sqrDistH = it.value();
				coarsedGraph.avgSqrDistH += sqrDistH;
				totalEdgesH++;
			}			
		}
	if (totalEdgesH)
		coarsedGraph.avgSqrDistH /= totalEdgesH;
	else
		coarsedGraph.avgSqrDistH = 1.0;

	for (int k = 0; k < coarsedGraph.A[coarsedGraph.edgeNumAttr - 1].outerSize(); k++)
		for (SparseMatrix<float, RowMajor>::InnerIterator it(coarsedGraph.A[coarsedGraph.edgeNumAttr - 1], k); it; ++it) {
			int row = it.row();
			int col = it.col();
			int depthLayer1 = row % coarsedGraph.nDepths;
			int depthLayer2 = col % coarsedGraph.nDepths;
			if (depthLayer1 == depthLayer2) {
				/*Cell node1 = coarsedGraph.nodes[row];
				Cell node2 = coarsedGraph.nodes[col];
				float sqrDistH = (node1.x - node2.x) * (node1.x - node2.x) + (node1.y - node2.y) * (node1.y - node2.y) + (node1.z - node2.z) * (node1.z - node2.z);
				assert(abs(sqrDistH - it.value()) < eps);*/
				float weight = exp(-it.value() / (4.0 * coarsedGraph.avgSqrDistH));
				it.valueRef() = weight;
			}
		}
}

void Coarsening::vertical_one_level(Graph & graph, vector<int>& cluster_id, Graph & coarsedGraph){
	int  N = graph.nDepths;

	vector<bool> marked(N);
	int clusterCount = 0;
	vector<float> newDs;
	for (int j = 0; j < N; j++) {
		int idx1 = j;
		if (!marked[idx1]) {
			marked[idx1] = true;
			cluster_id[idx1] = clusterCount;
			float newD = graph.nodes[idx1].d;

			if (j < N - 1) {
				int idx2 = idx1 + 1;
				cluster_id[idx2] = clusterCount;
				marked[idx2] = true;
				newD = (newD + graph.nodes[idx2].d) / 2;
			}
			newDs.push_back(newD);
			clusterCount++;
		}	
	}

	// Add graph nodes 
	for (int i = 0; i < graph.nCells; i++) {
		int idx = i * graph.nDepths;
		float newX = graph.nodes[idx].x;
		float newY = graph.nodes[idx].y;
		float newZ = graph.nodes[idx].z;
		for (int j = 0; j < clusterCount; j++) {
			bool newReal = graph.nodes[idx + j * 2].real;
			if (j * 2 < N - 1) {
				newReal = newReal || graph.nodes[idx + j * 2 + 1].real;
			}
			coarsedGraph.nodes.push_back(Cell(newX, newY, newZ, newDs[j], newReal));
		}
	}

	coarsedGraph.size = clusterCount * graph.nCells;
	for (int i=0; i < coarsedGraph.edgeNumAttr; i++)
		coarsedGraph.A[i].resize(coarsedGraph.size, coarsedGraph.size);
	coarsedGraph.nCells = graph.nCells;
	coarsedGraph.nDepths = clusterCount;
	coarsedGraph.avgSqrDistH = graph.avgSqrDistH;
	assert(coarsedGraph.nDepths == (graph.nDepths + 1) / 2);
	vector<Triplet<float> > triple[coarsedGraph.edgeNumAttr];

	// add vertical edges 
	coarsedGraph.avgSqrDistV = 0.0;
	for (int j = 0; j < coarsedGraph.nDepths - 1; j++) {
		float sqrDistV = (newDs[j] - newDs[j + 1]) * (newDs[j] - newDs[j + 1]);
		coarsedGraph.avgSqrDistV += sqrDistV;
	}
	if (coarsedGraph.nDepths - 1)
		coarsedGraph.avgSqrDistV /= (coarsedGraph.nDepths - 1);
	else
		coarsedGraph.avgSqrDistV = 1.0;
	for (int j = 0; j < coarsedGraph.nDepths - 1; j++) {
		float sqrDistV = (newDs[j] - newDs[j + 1]) * (newDs[j] - newDs[j + 1]);
		float weight = exp(-sqrDistV / (4.0 * coarsedGraph.avgSqrDistV));
		for (int i = 0; i < coarsedGraph.nCells; i++) {
			int mtIdx1 = i * coarsedGraph.nDepths + j;
			int mtIdx2 = i * coarsedGraph.nDepths + j + 1;
			if (coarsedGraph.nodes[mtIdx1].real && coarsedGraph.nodes[mtIdx2].real) {
				Edge edge1 = Edge(coarsedGraph.nodes[mtIdx1], coarsedGraph.nodes[mtIdx2], weight);
				triple[0].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaHighLat));
				triple[1].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaLowLat));
				triple[2].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaWest));
				triple[3].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.deltaEast));
				triple[4].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.isHorizontal));
				triple[5].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.isUp));
				triple[6].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.isDown));
				triple[7].push_back(Triplet<float>(mtIdx1, mtIdx2, edge1.weight));

				Edge edge2 = Edge(coarsedGraph.nodes[mtIdx2], coarsedGraph.nodes[mtIdx1], weight);
				triple[0].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaHighLat));
				triple[1].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaLowLat));
				triple[2].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaWest));
				triple[3].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.deltaEast));
				triple[4].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.isHorizontal));
				triple[5].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.isUp));
				triple[6].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.isDown));
				triple[7].push_back(Triplet<float>(mtIdx2, mtIdx1, edge2.weight));
			}
		}
	}
	
	// add horizontal edges 
	for (int k = 0; k < graph.A[coarsedGraph.edgeNumAttr - 1].outerSize(); k += graph.nDepths) {		// we only need to consider the surface level, rest are the same 
		for (SparseMatrix<float, RowMajor>::InnerIterator it(graph.A[coarsedGraph.edgeNumAttr - 1], k); it; ++it) {
			int row = it.row();
			int col = it.col();
			int depthLayer1 = row % graph.nDepths;
			int depthLayer2 = col % graph.nDepths;
			int cellId1 = row / graph.nDepths;
			int cellId2 = col / graph.nDepths;
			float weight = it.value();
			if (depthLayer1 == depthLayer2) {
				for (int j = 0; j < coarsedGraph.nDepths; j++) {
					int mtIdx1 = cellId1 * coarsedGraph.nDepths + j;
					int mtIdx2 = cellId2 * coarsedGraph.nDepths + j;
					if (coarsedGraph.nodes[mtIdx1].real && coarsedGraph.nodes[mtIdx2].real) {
						Edge edge = Edge(coarsedGraph.nodes[mtIdx1], coarsedGraph.nodes[mtIdx2], weight);
						triple[0].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaHighLat));
						triple[1].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaLowLat));
						triple[2].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaWest));
						triple[3].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.deltaEast));
						triple[4].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.isHorizontal));
						triple[5].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.isUp));
						triple[6].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.isDown));
						triple[7].push_back(Triplet<float>(mtIdx1, mtIdx2, edge.weight));
					}					
				}
			}
		}
	}
	
	for (int i=0; i < coarsedGraph.edgeNumAttr; i++)
		coarsedGraph.A[i].setFromTriplets(triple[i].begin(), triple[i].end()); 
}

// Return a list of indices to reorder the adjacency and data matrices 
// so that the union of two neighbors from layer to layer forms a binary tree 
void Coarsening::compute_perm(int sub_levels){
	// Order of last layer is not changed (chosen by the clustering algorithm)
	int M_last = *max_element(parents.back().begin(), parents.back().end()) + 1;
	int M = M_last;	// number of super nodes
	vector<int> index(M_last);
	iota(begin(index), end(index), 0);
	perms.push_back(index);
	
	for (int i = parents.size() - 1; i >= 0; i--) {
		cout << "compute permutation level " << i << endl;
		int pool_size;
		if (coarsenType[i] == HORIZONTAL)
			pool_size = (int)pow(2, sub_levels);
		else if (coarsenType[i] == VERTICAL)
			pool_size = 2;
		vector<vector<int>> indices_nodes(M);
		vector<int> indices_layer;
		for (int k = 0; k < parents[i].size(); k++) {
			indices_nodes[parents[i][k]].push_back(k);
		}
		for (int j = 0; j < perms.back().size(); j++) {
			int idx = perms.back()[j];
			indices_layer.insert(indices_layer.end(), indices_nodes[idx].begin(), indices_nodes[idx].end());
		}
		perms.push_back(indices_layer);

		// set assignment matrix
		vector<Triplet<float> > tripleUp, tripleAvgPool;
		tripleUp.reserve(parents[i].size());
		tripleAvgPool.reserve(parents[i].size());

		if (i == parents.size() - 1) {
			graphs[i + 1].upAsgn.resize(parents[i].size(), M);
			graphs[i].avgPoolAsgn.resize(M, parents[i].size());
		}
		else {
			graphs[i + 1].upAsgn.resize(parents[i].size(), parents[i + 1].size());
			graphs[i].avgPoolAsgn.resize(parents[i + 1].size(), parents[i].size());
		}

		int nodeCounter = 0;
		for (int j = 0; j < M; j++) {
			int realNodes = 0;
			for (int k = 0; k < indices_nodes[j].size(); k++) {
				int idx = indices_nodes[j][k];
				if (graphs[i].nodes[idx].real)
					realNodes++;
			}
			int norm = realNodes;
			int numNodes = indices_nodes[j].size();
			for (int k = 0; k < numNodes; k++) {
				int idx = indices_nodes[j][k];
				if (graphs[i].nodes[idx].real) {
					tripleUp.push_back(Triplet<float>(idx, j, 1.0));
					tripleAvgPool.push_back(Triplet<float>(j, idx, 1.0 / norm));				
					realNodes--;
				}			
				nodeCounter++;
			}
			assert(!realNodes);
		}
		assert(nodeCounter == parents[i].size());
		graphs[i + 1].upAsgn.setFromTriplets(tripleUp.begin(), tripleUp.end());
		graphs[i].avgPoolAsgn.setFromTriplets(tripleAvgPool.begin(), tripleAvgPool.end());

		M = indices_layer.size();
	}

	// Sanity check 
	for (int i = 0; i < perms.size(); i++) {
		// The new ordering does not omit an index 
		vector<int> indices_layer = perms[i];
		sort(indices_layer.begin(), indices_layer.end());
		for (int j = 0; j < indices_layer.size(); j++)
			if (indices_layer[j] != j)
				assert(false);
	}

	reverse(perms.begin(), perms.end());
}

// delete fake nodes in graph.nodes, graphs.A by resetting perms
void Coarsening::delete_fake_nodeA(){
	vector<vector<int>> fakeNodes;
	fakeNodes.resize(graphs.size());
	numReals.reserve(graphs.size());
	for (int i = 0; i < graphs.size(); i++) {
		vector<int> tmpPerm;
		for (int j = 0; j < graphs[i].size; j++) {
			if (graphs[i].nodes[perms[i][j]].real) {
				tmpPerm.push_back(perms[i][j]);
			}
			else {
				fakeNodes[i].push_back(perms[i][j]);
			}
		}
		for (int j = 0; j < fakeNodes[i].size(); j++) {
			tmpPerm.push_back(fakeNodes[i][j]);
		}
		perms[i] = tmpPerm;
		numReals.push_back(graphs[i].size - fakeNodes[i].size());
	}
}

// delete fake nodes in upAsgn, avgPoolAsgn
void Coarsening::delete_fake_asgn(){
	vector<PermutationMatrix<Dynamic, Dynamic>> permMats;
	for (int i = 0; i < graphs.size(); i++) {
		PermutationMatrix<Dynamic, Dynamic> permMat(graphs[i].size);
		for (int j = 0; j < perms[i].size(); j++) {
			permMat.indices()[perms[i][j]] = j;
		}
		permMats.push_back(permMat);
	}

	// delete upAsgn and avgAsgn
	for (int i = 0; i < graphs.size() - 1; i++) {
		int old_nnz = graphs[i + 1].upAsgn.nonZeros();
		int sm = numReals[i + 1];
		int lg = numReals[i];
		graphs[i + 1].upAsgn = permMats[i] * graphs[i + 1].upAsgn;
		graphs[i + 1].upAsgn = graphs[i + 1].upAsgn * permMats[i + 1].inverse();
		graphs[i + 1].upAsgn = graphs[i + 1].upAsgn.topLeftCorner(lg, sm);
		assert(graphs[i + 1].upAsgn.nonZeros() == old_nnz);
		  
		graphs[i].avgPoolAsgn = permMats[i + 1] * graphs[i].avgPoolAsgn;
		graphs[i].avgPoolAsgn = graphs[i].avgPoolAsgn * permMats[i].inverse();
		graphs[i].avgPoolAsgn = graphs[i].avgPoolAsgn.topLeftCorner(sm, lg);
		assert(graphs[i].avgPoolAsgn.nonZeros() == old_nnz);
	}
}

// permute data vector i.e. exchange node ids 
// so that binary unions form the clustering tree. 
void Coarsening::perm_data(vector<float>& x, vector<int>& indices, vector<float>& xNew, int numReal){
	int M = x.size();
	int MNew = indices.size();
	assert(MNew == M);
	xNew.resize(MNew);
	assert(numReal <= M);
	
	for (int i = 0; i < numReal; i++) {
		// existing vertex. i.e. real data 
		xNew[i] = x[indices[i]];
	}
}

void Coarsening::perm_data(vector<Cell> & x, vector<int>& indices, vector<Cell>& xNew, int numReal) {
	int M = x.size();
	int MNew = indices.size();
	assert(MNew == M);
	xNew.resize(MNew);
	assert(numReal <= M);

	for (int i = 0; i < numReal; i++) {
		xNew[i] = x[indices[i]];
	}
}

// permute adjancy matrix, i.e. exchange node ids
// so that binary unions form the clustering tree.
void Coarsening::perm_adjacency(SparseMatrix<float, RowMajor>& A, vector<int>& indices, int numReal){
	/*int outS = A.outerSize();
	for (int j = A.outerIndexPtr()[0]; j < A.outerIndexPtr()[1]; j++) {
		cout << A.innerIndexPtr()[j] << " ";
	}
	cout << endl;*/

	int M = A.rows();
	int MNew = indices.size();
	assert(MNew == M);
	
	PermutationMatrix<Dynamic, Dynamic> permMat(MNew);

	for (int i = 0; i < MNew; i++) {
		permMat.indices()[indices[i]] = i;
	}
	int old_nnz = A.nonZeros();
	A = A.twistedBy(permMat);
	A = A.topLeftCorner(numReal, numReal);
	assert(A.nonZeros() == old_nnz);

	/*for (int j = A.outerIndexPtr()[0]; j < A.outerIndexPtr()[1]; j++) {
		cout << A.innerIndexPtr()[j] << " ";
	}
	cout << endl;*/
}

float Coarsening::EuclideanDist(int graphId, int front, int back)
{
	float avgSqrDistH = graphs[graphId].avgSqrDistH;
	float avgSqrDistV = graphs[graphId].avgSqrDistV;
	Cell frontCell = graphs[graphId].nodes[front];
	Cell backCell = graphs[graphId].nodes[back];
	float sqrDistH = (frontCell.x - backCell.x) * (frontCell.x - backCell.x) + (frontCell.y - backCell.y) * (frontCell.y - backCell.y) + (frontCell.z - backCell.z) * (frontCell.z - backCell.z);
	float sqrDistV = (frontCell.d - backCell.d) * (frontCell.d - backCell.d);
	return sqrt(sqrDistH / avgSqrDistH + sqrDistV/ avgSqrDistV);
}

void Coarsening::permTest(){
	parents.push_back(vector<int> {4, 1, 1, 2, 2, 3, 0, 0, 3});
	parents.push_back(vector<int> {2, 1, 0, 1, 0});
	//coarsening.parents.push_back(vector<int> {0, 1, 1, 0, 0, 1, 2, 2, 1});
	compute_perm(1);
}