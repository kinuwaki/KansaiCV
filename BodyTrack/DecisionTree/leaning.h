#pragma once

// Regression Forest
const int NUM_TREE = 4;

struct Metric{
	double variance;
	double leftvalue[2], rightvalue[2];
	double leftvariance, rightvariance;
	int u[2];
	int v[2];
	int delta;
};

struct Node{
	int numsample;
	int depth;
	float result[2]; // result

	bool leaf;
	Metric metric;

	Node* leftNode;
	Node* rightNode;

	Node(){
		leaf = 0;
	}
};
extern Node* rootNode[NUM_TREE];

