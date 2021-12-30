#include "GHT.h"

GHNode::GHNode(){
}

bool GHNode::traverse(float threshold){
	if (!height) {
		merged = true;
	}
	else {
		merged = true; 
		for (int i = 0; i < children.size(); i++) {
			merged = merged & children[i]->traverse(threshold);
		}
		merged = merged & (maxMaxDiff < threshold);
	}
	return merged;
}

GHTree::GHTree(const vector<int>& graphSizes){
	height = graphSizes.size();
	nodes.resize(height);
	for (int i = 0; i < graphSizes.size(); i++) {
		nodes[i].reserve(graphSizes[i]);
	}
}

void GHTree::traverse(float threshold){
	assert(nodes[height - 1].size() == 1);
	nodes[height - 1][0].traverse(threshold);
}

void GHTree::cut(){
	assert(nodes[height - 1].size() == 1);
	assert(nodes[height - 1][0].parent == NULL);
	for (int i = height - 1; i >= 0; i--) {
		int count = 0;
		for (int j = 0; j < nodes[i].size(); j++) {
			GHNode& ghnode = nodes[i][j];
			if (ghnode.parent != NULL && ghnode.parent->merged) {
				ghnode.proxy = ghnode.parent->proxy;
			}
			else {
				ghnode.proxy = &ghnode;
			}

			assert(ghnode.newIdx == j);
			if (j) {
				GHNode& pre = nodes[i][j - 1];
				if (ghnode.proxy != pre.proxy){
					count++;
				}	
			}
			ghnode.newIdx = count;
		}
	}
}


