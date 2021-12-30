#include "graph.h"

Graph::Graph(){
	avgSqrDistH = avgSqrDistV = 0.0;
}

Graph::Graph(vector<Cell> cells, int nVertLevels): nodes(cells), nDepths(nVertLevels){
	size = cells.size();
	for (int i = 0; i < 8; i++)
		A[i].resize(size, size);
	nCells = size / nDepths;
	assert(!(size % nDepths));
	avgSqrDistH = avgSqrDistV = 0.0;
}

void Graph::Serialize(SparseMatrix<float, RowMajor>& m, const string& filename){
	m.makeCompressed();

	fstream writeFile;
	writeFile.open(filename.c_str(), ios::binary | ios::out);

	if (writeFile.is_open()) {
		int rows, cols, nnzs, outS, innS;
		rows = m.rows();
		cols = m.cols();
		nnzs = m.nonZeros();
		outS = m.outerSize();
		innS = m.innerSize();

		writeFile.write((const char*)&(rows), sizeof(int));
		writeFile.write((const char*)&(cols), sizeof(int));
		writeFile.write((const char*)&(nnzs), sizeof(int));
		writeFile.write((const char*)&(innS), sizeof(int));
		writeFile.write((const char*)&(outS), sizeof(int));

		writeFile.write((const char *)(m.valuePtr()), sizeof(float) * m.nonZeros());
		writeFile.write((const char *)(m.outerIndexPtr()), sizeof(int) * m.outerSize());
		writeFile.write((const char *)(m.innerIndexPtr()), sizeof(int) * m.nonZeros());

		writeFile.close();
	}
}

void Graph::Deserialize(SparseMatrix<float, RowMajor>& m, const string& filename) {
	fstream readFile;
	readFile.open(filename.c_str(), ios::binary | ios::in);
	if (readFile.is_open())
	{
		int rows, cols, nnz, inSz, outSz;
		readFile.read((char*)&rows, sizeof(int));
		readFile.read((char*)&cols, sizeof(int));
		readFile.read((char*)&nnz, sizeof(int));
		readFile.read((char*)&inSz, sizeof(int));
		readFile.read((char*)&outSz, sizeof(int));

		m.resize(rows, cols);
		m.makeCompressed();
		m.resizeNonZeros(nnz);

		readFile.read((char*)(m.valuePtr()), sizeof(float) * nnz);
		readFile.read((char*)(m.outerIndexPtr()), sizeof(int) * outSz);
		readFile.read((char*)(m.innerIndexPtr()), sizeof(int) * nnz);

		m.finalize();
		readFile.close();

	} // file is open
}

void Graph::SaveMatrix(SparseMatrix<float, RowMajor> m[edgeNumAttr], const string & idxFilename, const string & valueFilename){
	int nnzs, outS;
	outS = m[0].outerSize();
	nnzs = m[0].nonZeros();
	if (nnzs) {
		vector<int> indices;
		indices.reserve(nnzs * 2);

		// save indices 
		int counter = 0;
		for (int i = 0; i < outS - 1; i++) {
			for (int j = m[0].outerIndexPtr()[i]; j < m[0].outerIndexPtr()[i + 1]; j++) {
				indices.push_back(i);
			}
		}
		for (int j = m[0].outerIndexPtr()[outS - 1]; j < nnzs; j++) {
			indices.push_back(outS - 1);
		}
		assert(indices.size() == nnzs);
		for (int i = 0; i < nnzs; i++) {
			int idx = m[0].innerIndexPtr()[i];
			assert(idx < nnzs);
			indices.push_back(idx);
		}

		cnpy::npy_save(idxFilename.c_str(), &indices[0], { (size_t)2 ,(size_t)(nnzs) }, "w");

		// save values 
		vector<float> values;
		values.resize(edgeNumAttr * nnzs);
		for (int i = 0; i < edgeNumAttr; i++) {
			memcpy(&values[i * nnzs], m[i].valuePtr(), nnzs * sizeof(float));
		}
		cnpy::npy_save(valueFilename.c_str(), &values[0], { (size_t)edgeNumAttr , (size_t)(nnzs) }, "w");
	}
}

void Graph::SaveMatrix(SparseMatrix<float, RowMajor>& m, const string& idxFilename, const string& valueFilename) {
	int nnzs, outS;
	outS = m.outerSize();
	nnzs = m.nonZeros();
	if (nnzs) {
		vector<int> indices;
		indices.reserve(nnzs * 2);

		// save indices 
		int counter = 0;
		for (int i = 0; i < outS - 1; i++) {
			for (int j = m.outerIndexPtr()[i]; j < m.outerIndexPtr()[i + 1]; j++) {
				indices.push_back(i);
			}
		}
		for (int j = m.outerIndexPtr()[outS - 1]; j < nnzs; j++) {
			indices.push_back(outS - 1);
		}
		assert(indices.size() == nnzs);
		for (int i = 0; i < nnzs; i++) {
			int idx = m.innerIndexPtr()[i];
			assert(idx < nnzs);
			indices.push_back(idx);
		}
			
		cnpy::npy_save(idxFilename.c_str(), &indices[0], { (size_t)2 ,(size_t)(nnzs) }, "w");

		// save values 
		cnpy::npy_save(valueFilename.c_str(), m.valuePtr(), { (size_t)(nnzs) }, "w");
	}
}


void Graph::SaveNodes(const string& filename){
	vector<float> posCell;
	for (int i = 0; i < nodes.size(); i++) {
		if (nodes[i].real){ 
    		posCell.push_back(nodes[i].x);
    		posCell.push_back(nodes[i].y);
    		posCell.push_back(nodes[i].z);
    		posCell.push_back(nodes[i].lat);
    		posCell.push_back(nodes[i].lon);
    		posCell.push_back(nodes[i].d);
    		posCell.push_back(nodes[i].real);
		}
	}
	// save it to file 
	cnpy::npy_save(filename.c_str(), &posCell[0], { (size_t)(posCell.size()/7), (size_t)7 }, "w");
}
