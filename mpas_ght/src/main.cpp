#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <vector>
#include "GHT.h"

using namespace std;

float threshold;
const int edgeNumAttr = 8;
vector<string> filenames;
// example: 
// 	"0047_4.99214_1377.12858_0.69051_289.17606_temperature.npy",
// 	"0048_2.88486_401.01052_0.28008_115.72901_temperature.npy",
// 	"0049_0.14181_857.12411_0.78607_197.39257_temperature.npy",
// 	"0050_2.34246_503.46576_0.62507_151.44783_temperature.npy",
// 	"0051_3.48109_667.79315_0.48774_219.78883_temperature.npy",
// 	"0052_3.91644_533.79570_0.88867_251.67885_temperature.npy",
// 	"0053_3.30245_1093.49860_0.68303_204.70439_temperature.npy",
// 	"0054_4.54106_1477.22077_0.56356_294.79028_temperature.npy",
// 	"0055_4.65705_1149.46520_0.27245_288.61040_temperature.npy",
// 	"0056_0.37095_602.75963_0.40065_123.56513_temperature.npy",
// 	"0057_4.27838_1071.53351_0.92994_237.91306_temperature.npy",
// 	"0058_3.28657_370.41000_0.67670_284.41809_temperature.npy",
// 	"0059_3.13326_1195.80632_0.59768_192.34201_temperature.npy",
// 	"0060_1.99014_718.34192_0.36125_198.64491_temperature.npy"
int numFiles = 1;
char root[1024]; // /users/PAS0027/trainsn/mpas/mpas_graph/res/EC60to30/
char ghtRoot[1024];    // ../res/EC60to30_0.5/
vector<int> graphSizes, ghtGraphSizes;
vector<double> temperature;
vector<vector<vector<int>>> avgPoolAsgnIdx, ghtAvgPoolAsgnIdx, ghtUpAsgnIdx;
vector<vector<float>> ghtAvgPoolAsgnValue, ghtUpAsgnValue;
vector<vector<vector<int>>> adjIdx, ghtAdjIdx;
vector<vector<vector<float>>> adjValue, ghtAdjValue;

void readTemperature(const string& filename) {
	fstream readFile;
	readFile.open(filename.c_str(), ios::binary | ios::in);
	streampos fileSize;
	readFile.seekg(0, std::ios::end);
	fileSize = readFile.tellg();
	readFile.seekg(0, std::ios::beg);

	temperature.resize(fileSize / sizeof(double));
	// read the data
	readFile.read((char*)(&temperature[0]), fileSize);
	readFile.close();
}

void readGraphSizes() {
	stringstream ss;
	ss << root << "graphSizes.dat";
	string filename = ss.str();

	fstream readFile;
	readFile.open(filename.c_str(), ios::binary | ios::in);
	streampos fileSize;
	readFile.seekg(0, std::ios::end);
	fileSize = readFile.tellg();
	readFile.seekg(0, std::ios::beg);

	graphSizes.resize(fileSize / sizeof(int));
	// read the data
	readFile.read((char*)(&graphSizes[0]), fileSize);
}

void readAvgPool() {
	avgPoolAsgnIdx.resize(graphSizes.size() - 1);
	for (int i = 0; i < graphSizes.size() - 1; i++) {
		stringstream ss;
		ss << root << "avgPoolAsgnIdx" << i << ".dat";
		string filename = ss.str();

		fstream readFile;
		readFile.open(filename.c_str(), ios::binary | ios::in);
		streampos fileSize;
		readFile.seekg(0, std::ios::end);
		fileSize = readFile.tellg();
		readFile.seekg(0, std::ios::beg);

		vector<int> indices;
		indices.resize(fileSize / sizeof(int));
		// read the data:
		readFile.read((char*)(&indices[0]), fileSize);

		avgPoolAsgnIdx[i].resize(2);
		avgPoolAsgnIdx[i][0].insert(avgPoolAsgnIdx[i][0].begin(), indices.begin(), indices.begin() + indices.size() / 2);
		avgPoolAsgnIdx[i][1].insert(avgPoolAsgnIdx[i][1].begin(), indices.begin() + indices.size() / 2, indices.end());
	}
}

void readAdjIdx() {
	adjIdx.resize(graphSizes.size() - 1);
	for (int i = 0; i < graphSizes.size() - 1; i++) {
		stringstream ss;
		ss << root << "adjacencyIdx" << i << ".dat";
		string filename = ss.str();

		fstream readFile;
		readFile.open(filename.c_str(), ios::binary | ios::in);
		streampos fileSize;
		readFile.seekg(0, std::ios::end);
		fileSize = readFile.tellg();
		readFile.seekg(0, std::ios::beg);

		vector<int> indices(fileSize / sizeof(int));
		// read the data:
		readFile.read((char*)(&indices[0]), fileSize);

		adjIdx[i].resize(2);
		adjIdx[i][0].insert(adjIdx[i][0].begin(), indices.begin(), indices.begin() + indices.size() / 2);
		adjIdx[i][1].insert(adjIdx[i][1].begin(), indices.begin() + indices.size() / 2, indices.end());
	}
}

void readAdjValue() {
	adjValue.resize(graphSizes.size() - 1);
	for (int i = 0; i < graphSizes.size() - 1; i++) {
		stringstream ss;
		ss << root << "adjacencyValue" << i << ".dat";
		string filename = ss.str();

		fstream readFile;
		readFile.open(filename.c_str(), ios::binary | ios::in);
		streampos fileSize;
		readFile.seekg(0, std::ios::end);
		fileSize = readFile.tellg();
		readFile.seekg(0, std::ios::beg);

		vector<float> temp;
		temp.resize(fileSize / sizeof(float));
		assert(temp.size() == adjIdx[i][0].size() * edgeNumAttr);
		// read the data:
		readFile.read((char*)(&temp[0]), fileSize);
		adjValue[i].resize(edgeNumAttr);
		for (int j = 0; j < edgeNumAttr; j++) {
			adjValue[i][j].insert(adjValue[i][j].begin(),
				temp.begin() + adjIdx[i][0].size() * j, temp.begin() + adjIdx[i][0].size() * (j + 1));
		}
	}
}

void writeGhtGraphSizes() {
	stringstream ss;
	ss << ghtRoot << "ghtGraphSizes.npy";
	string filename = ss.str();

	cnpy::npy_save(filename.c_str(), &ghtGraphSizes[0], { (size_t)ghtGraphSizes.size() }, "w");
}

void writeGhtAvgPoolIdx() {
	for (int i = 0; i < ghtGraphSizes.size() - 1; i++) {
		stringstream ss;
		ss << ghtRoot << "ghtAvgPoolAsgnIdx" << i << ".npy";
		string filename = ss.str();

		vector<int> temp;
		temp.reserve(ghtAvgPoolAsgnIdx[i][0].size() * 2);
		temp.insert(temp.end(), ghtAvgPoolAsgnIdx[i][0].begin(), ghtAvgPoolAsgnIdx[i][0].end());
		temp.insert(temp.end(), ghtAvgPoolAsgnIdx[i][1].begin(), ghtAvgPoolAsgnIdx[i][1].end());

		cnpy::npy_save(filename.c_str(), &temp[0], { (size_t)2, (size_t)ghtGraphSizes[i] }, "w");
	}
}

void writeGhtAvgPoolValue() {
	for (int i = 0; i < ghtGraphSizes.size() - 1; i++) {
		stringstream ss;
		ss << ghtRoot << "ghtAvgPoolAsgnValue" << i << ".npy";
		string filename = ss.str();

		cnpy::npy_save(filename.c_str(), &ghtAvgPoolAsgnValue[i][0], { (size_t)ghtAvgPoolAsgnValue[i].size() }, "w");
	}
}

void writeGhtUpIdx() {
	for (int i = 0; i < ghtGraphSizes.size() - 1; i++) {
		stringstream ss;
		ss << ghtRoot << "ghtUpAsgnIdx" << i + 1 << ".npy";
		string filename = ss.str();

		vector<int> temp;
		temp.reserve(ghtUpAsgnIdx[i][0].size() * 2);
		temp.insert(temp.end(), ghtUpAsgnIdx[i][0].begin(), ghtUpAsgnIdx[i][0].end());
		temp.insert(temp.end(), ghtUpAsgnIdx[i][1].begin(), ghtUpAsgnIdx[i][1].end());

		cnpy::npy_save(filename.c_str(), &temp[0], { (size_t)2, (size_t)ghtGraphSizes[i] }, "w");
	}
}

void writeGhtUpValue() {
	for (int i = 0; i < ghtGraphSizes.size() - 1; i++) {
		stringstream ss;
		ss << ghtRoot << "ghtUpAsgnValue" << i + 1 << ".npy";
		string filename = ss.str();

		cnpy::npy_save(filename.c_str(), &ghtUpAsgnValue[i][0], { (size_t)ghtUpAsgnValue[i].size() }, "w");
	}
}

void writeGhtAdjIdx() {
	for (int i = 0; i < ghtGraphSizes.size() - 1; i++) {
		stringstream ss;
		ss << ghtRoot << "ghtAdjacencyIdx" << i << ".npy";
		string filename = ss.str();

		vector<int> temp;
		temp.reserve(ghtAdjIdx[i][0].size() * 2);
		temp.insert(temp.end(), ghtAdjIdx[i][0].begin(), ghtAdjIdx[i][0].end());
		temp.insert(temp.end(), ghtAdjIdx[i][1].begin(), ghtAdjIdx[i][1].end());

		cnpy::npy_save(filename.c_str(), &temp[0], { (size_t)2, (size_t)ghtAdjIdx[i][0].size() }, "w");
	}
}

void writeGhtAdjValue() {
	for (int i = 0; i < ghtGraphSizes.size() - 1; i++) {
		stringstream ss;
		ss << ghtRoot << "ghtAdjacencyValue" << i << ".npy";
		string filename = ss.str();

		vector<float> temp;
		temp.reserve(ghtAdjValue[i][0].size() * edgeNumAttr);
		for (int j = 0; j < edgeNumAttr; j++) {
			temp.insert(temp.end(), ghtAdjValue[i][j].begin(), ghtAdjValue[i][j].end());
		}

		cnpy::npy_save(filename.c_str(), &temp[0], { (size_t)edgeNumAttr, (size_t)ghtAdjValue[i][0].size() }, "w");
	}
}

int main(int argc, char **argv) {
    sprintf(root, argv[1]);
    threshold = atof(argv[2]);
    char input_root[1024];
	sprintf(input_root, argv[3]);
    sprintf(ghtRoot, argv[4]);
	numFiles = atoi(argv[5]);
	for (int i = 6; i < 6 + numFiles; i++){
	    stringstream ss;
		ss << argv[i] << ".npy";
		string filename = ss.str();
		filenames.push_back(filename);
	}

	readGraphSizes();
	readAvgPool();

	int height = graphSizes.size();
	GHTree ghtree = GHTree(graphSizes);

	for (int h = 0; h < height - 1; h++) {
		if (!h) {
			ghtree.nodes[h].reserve(graphSizes[h]);
			for (int i = 0; i < graphSizes[h]; i++) {
				assert(avgPoolAsgnIdx[h][1][i] == i);

				GHNode ghnode = GHNode();
				ghnode.height = h;
				ghnode.newIdx = i;
				ghnode.leafHead = ghnode.leafTail = i;
				ghnode.maxMaxDiff = 0.0f;
				ghnode.merged = false;
				ghtree.nodes[h].push_back(ghnode);
			}
		}
		int st = 0, en = 1;
		int pid = avgPoolAsgnIdx[h][0][0];
		for (int i = 0; i < graphSizes[h]; i++) {
			if (i < graphSizes[h] - 1 && avgPoolAsgnIdx[h][0][i + 1] == pid) {
				en++;
			}
			else {
				GHNode ghnode = GHNode();
				ghnode.height = h + 1;
				ghnode.newIdx = ghtree.nodes[h + 1].size();
				for (int j = st; j < en; j++) {
					ghnode.children.push_back(&ghtree.nodes[h][j]);
				}
				ghnode.leafHead = ghtree.nodes[h][st].leafHead;
				ghnode.leafTail = ghtree.nodes[h][en - 1].leafTail;
				ghnode.maxMaxDiff = 0.0f;

				ghnode.merged = false;
				ghtree.nodes[h + 1].push_back(ghnode);
				for (int j = st; j < en; j++) {
					ghtree.nodes[h][j].parent = &*(ghtree.nodes[h + 1].end() - 1);
				}

				if (en < avgPoolAsgnIdx[h][0].size()) {
					st = en;
					pid = avgPoolAsgnIdx[h][0][st];
					en = st + 1;
				}
			}
		}
	}
	ghtree.nodes[height - 1][0].parent = NULL;

	for (int j = 0; j < numFiles; j++) {
		stringstream ss;
		ss << input_root << filenames[j];
		string filename = ss.str();
		readTemperature(filename);
		cout << filename << endl;

		for (int h = 0; h < height - 1; h++) {
			for (int i = 0; i < graphSizes[h + 1]; i++) {
				GHNode& ghnode = ghtree.nodes[h + 1][i];
				float mean = 0.0f;
				for (int k = ghnode.leafHead; k <= ghnode.leafTail; k++) {
					mean += temperature[k];
				}
				mean /= (ghnode.leafTail - ghnode.leafHead + 1);

				float maxDiff = 0.0f;
				for (int k = ghnode.leafHead; k <= ghnode.leafTail; k++) {
					float diff = abs(temperature[k]);
					if (diff > maxDiff) {
						maxDiff = diff;
					}
				}
				if (maxDiff > ghnode.maxMaxDiff) {
					ghnode.maxMaxDiff = maxDiff;
				}
			}
		}
	}
	cout << "finish calculating maxDiff." << endl;

	// cut the graph hierarchical tree given a threshold 
	ghtree.traverse(threshold);
	cout << "finish traversing." << endl;
	ghtree.cut();
	cout << "finish cutting." << endl;
	cout << endl;

	readAdjIdx();
	readAdjValue();

	ghtAdjIdx.resize(height);
	ghtAdjValue.resize(height);
	// create ght adjacency matrix 
	for (int h = height - 1; h >= 1; h--) {
		ghtAdjIdx[h].resize(2);
		for (int i = 0; i < 2; i++)
			ghtAdjIdx[h][i].reserve(adjIdx[h - 1][0].size() / 3);
		ghtAdjValue[h].resize(edgeNumAttr);
		for (int i = 0; i < edgeNumAttr; i++)
			ghtAdjValue[h][i].reserve(adjValue[h - 1][0].size() / 3);
		for (int i = 0; i < adjIdx[h - 1][0].size(); i++) {
			int left = adjIdx[h - 1][0][i];
			int right = adjIdx[h - 1][1][i];
			assert(left != right);
			int ghtLeft = ghtree.nodes[h - 1][left].newIdx;
			int ghtRight = ghtree.nodes[h - 1][right].newIdx;
			if (ghtLeft != ghtRight) {
				ghtAdjIdx[h][0].push_back(ghtLeft);
				ghtAdjIdx[h][1].push_back(ghtRight);
				for (int j = 0; j < edgeNumAttr; j++) {
					ghtAdjValue[h][j].push_back(adjValue[h - 1][j][i]);
				}
			}
		}
	}
	ghtAdjIdx[0] = adjIdx[0];
	ghtAdjValue[0] = adjValue[0];

	// create ght up and avgPool assignment index matrix
	ghtAvgPoolAsgnIdx.resize(height);
	ghtUpAsgnIdx.resize(height);
	for (int h = height - 1; h >= 1; h--) {
		ghtAvgPoolAsgnIdx[h].resize(2);
		ghtUpAsgnIdx[h].resize(2);
		int numGhtNodes = ghtree.nodes[h - 1].back().newIdx;
		for (int i = 0; i < 2; i++) {
			ghtAvgPoolAsgnIdx[h][i].reserve(numGhtNodes);
			ghtUpAsgnIdx[h][i].reserve(numGhtNodes);
		}
		for (int i = 0; i < graphSizes[h - 1]; i++) {
			bool add = true;
			GHNode& cur = ghtree.nodes[h - 1][i];
			if (i) {
				GHNode& pre = ghtree.nodes[h - 1][i - 1];
				if (pre.newIdx == cur.newIdx) {
					assert(pre.parent->newIdx == cur.parent->newIdx);
					add = false;
				}
			}
			if (add) {
				ghtAvgPoolAsgnIdx[h][0].push_back(cur.parent->newIdx);
				ghtAvgPoolAsgnIdx[h][1].push_back(cur.newIdx);
				ghtUpAsgnIdx[h][0].push_back(cur.newIdx);
				ghtUpAsgnIdx[h][1].push_back(cur.parent->newIdx);
			}
		}
	}
	ghtAvgPoolAsgnIdx[0].resize(2);
	ghtUpAsgnIdx[0].resize(2);
	for (int i = 0; i < 2; i++) {
		ghtAvgPoolAsgnIdx[0][i].reserve(graphSizes[0]);
		ghtUpAsgnIdx[0][i].reserve(graphSizes[0]);
	}
	for (int i = 0; i < graphSizes[0]; i++) {
		GHNode& cur = ghtree.nodes[0][i];
		ghtAvgPoolAsgnIdx[0][0].push_back(cur.newIdx);
		ghtAvgPoolAsgnIdx[0][1].push_back(i);
		ghtUpAsgnIdx[0][0].push_back(i);
		ghtUpAsgnIdx[0][1].push_back(cur.newIdx);
	}

	// set ghtGraphSize
	ghtGraphSizes.resize(height + 1);
	for (int h = 0; h < height; h++) {
		ghtGraphSizes[h] = ghtAvgPoolAsgnIdx[h][0].size();
		int M = ghtGraphSizes[h];
		int nnz = ghtAdjIdx[h][0].size();
		printf("Layer %d: M_%d = |V| = %d nodes, |E| = %d edges, avgDeg = %.2lf\n", h, h, M, nnz, (float)nnz / M);
	}
	ghtGraphSizes[height] = 1;
	printf("Layer %d: M_%d = |V| = %d nodes, |E| = %d edges, avgDeg = %.2lf\n", height, height, 1, 0, 0.0f);

	// create ght up and avgPool assignment value matrix
	ghtAvgPoolAsgnValue.resize(height);
	ghtUpAsgnValue.resize(height);
	for (int h = height - 1; h >= 0; h--) {
		ghtAvgPoolAsgnValue[h].reserve(ghtGraphSizes[h]);
		ghtUpAsgnValue[h].reserve(ghtGraphSizes[h]);
		int st = 0, en = 1;
		int pid = ghtAvgPoolAsgnIdx[h][0][0];
		for (int i = 0; i < ghtGraphSizes[h]; i++) {
			if (i < ghtGraphSizes[h] - 1 && ghtAvgPoolAsgnIdx[h][0][i + 1] == pid) {
				en++;
			}
			else {
				for (int j = st; j < en; j++) {
					ghtAvgPoolAsgnValue[h].push_back(1.0 / (en - st));
					ghtUpAsgnValue[h].push_back(1.0);
				}
				if (en < ghtGraphSizes[h]) {
					st = en;
					pid = ghtAvgPoolAsgnIdx[h][0][st];
					en = st + 1;
				}
			}
		}
		assert(ghtAvgPoolAsgnValue[h].size() == ghtGraphSizes[h]);
	}
	cout << endl;
	cout << "create ght matrices." << endl;

	// output ght matrices 
	writeGhtGraphSizes();
	writeGhtAvgPoolIdx();
	writeGhtAvgPoolValue();
	writeGhtUpIdx();
	writeGhtUpValue();
	writeGhtAdjIdx();
	writeGhtAdjValue();

	cout << endl;
}
