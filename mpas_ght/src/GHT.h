#pragma once

#include <vector> 
#include "cnpy.h"

using namespace std;

class GHNode {
public:
	int height, newIdx;
	GHNode* parent;
	vector<GHNode*> children;
	GHNode* proxy;
	int leafHead, leafTail;
	float maxMaxDiff;
	bool merged;

	GHNode();
	bool traverse(float threshold);
};

class GHTree {
public:
	int height;
	vector<vector<GHNode>> nodes;

	GHTree(const vector<int>& graphSizes);
	void traverse(float threshold);
	void cut();
};