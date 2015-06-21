#include "main.h"

// replace in future
/*
const int MAX_DELTA = 200;
void constructNode(int landmark, Node* curNode, vector<int> bin, int depth, double variance)
{
	int i;

	if (variance == 0.0f){
		// create leaf node
		curNode->leaf = true;
		curNode->depth = depth;
		bin.clear();
		printf("Create leaf node due to variance == 0.0f\n");
		return;
	}

	if (bin.size() < MIN_SAMPLE_THRESHOLD){
		// create leaf node
		curNode->leaf = true;
		curNode->depth = depth;
		bin.clear();
		printf("Create leaf node due to small samples\n");
		return;
	}

	if (depth >= MAX_TREE_DEPTH){
		// create leaf node
		curNode->leaf = true;
		curNode->depth = depth;
		bin.clear();
		printf("Create leaf node due to max depth\n");
		return;
	}

	Metric minmetric;
	minmetric.variance = variance;

	int u[2], v[2];
	int posu[2], posv[2];
	int delta;

	int numsample[2];
	double valuebucket[2][2];

	bool findbetterentropy = false;
	bool* flag = new bool[curNode->numsample];

	// それぞれの U, V , Threshold の条件でエントロピーを評価します
	for (u[1] = -SIZE_WINDOW_SIZE / 2; u[1]≤SIZE_WINDOW_SIZE / 2; u[1] += 1){
		for (u[0] = -SIZE_WINDOW_SIZE / 2; u[0]≤SIZE_WINDOW_SIZE / 2; u[0] += 1){
			for (v[1] = -SIZE_WINDOW_SIZE / 2; v[1]≤SIZE_WINDOW_SIZE / 2; v[1] += 1){
				for (v[0] = -SIZE_WINDOW_SIZE / 2; v[0]≤SIZE_WINDOW_SIZE / 2; v[0] += 1){
					for (threshold = 0; threshold≤MAX_THRESHOLD; threshold++){
						ClearAllSampleList();

						// 全サンプル(ピクセル)に対して、今の条件で
						// 左ノード(条件を満たすか)に行くか、右ノードに行くか(条件を満たさないか)の振り分けを行います
						for (sampleIdx = 0; sampleIdx < NUM_SAMPLE; sampleIdx++){

							positionU.x = SampleList[sampleIdx].position.x + u[0];
							positionU.y = SampleList[sampleIdx].position.y + u[1];
							depthAtU = depthMap[SampleList[sampleIdx].depthMapIdx].data[positionU.x + positionU.y*SENSOR_WIDTH];

							positionV.x = SampleList[sampleIdx].position.x + v[0];
							positionV.y = SampleList[sampleIdx].position.y + v[1];
							depthAtV = depthMap[SampleList[sampleIdx].depthMapIdx].data[positionV.x + positionV.y*SENSOR_WIDTH];

							if (depthAtU - depthAtV < threshold){
								numSampleInLeftNode++;
								categoryCounterInLeftNode[SampleList[sampleIdx].correctCategory]++;
							}
							else{
								numSampleInRightNode++;
								categoryCounterInRightNode[SampleList[sampleIdx].correctCategory][1]++;
							}
						}

						// 左ノードと右ノードに対して、エントロピーの評価を行います
						float leftEntropy = 0.0f, rightentropy = 0.0f;
						for (idxCategory = 0; idxCategory < NUM_CATEGORY; idxCategory++){
							float SiSj = categoryCounterInLeftNode[idxCategory] / numSampleInLeftNode;
							leftEntropy -= SiSj * log2f(SiSj);

							SiSj = categoryCounterInLeftNode[idxCategory] / numSampleInRightNode;
							rightEntropy -= SiSj * log2f(SiSj);

							float tmpEntropy = numSampleInLeftNode / NUM_SAMPLE * leftEntropy
								numSampleInrightNode / NUM_SAMPLE * rightEntropy;

							// エントロピーが最小だった場合はその状況を更新します
							if (tmpEntropy ≤ minMetric.entropy){
								minMetric.u = u;
								minmetric.v = v;
								minmetric.threshold = threshold;
							}
						}
					}
				}
			}
		}

	}

void ProcessLabeling()
{
	std::tr2::sys::path input_path("./input/");

	FILE* fp;

}
*/