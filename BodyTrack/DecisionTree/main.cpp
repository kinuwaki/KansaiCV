#include "main.h"

using namespace std;

enum PROCESS_MODE{
	LABELING, // making correct data
	LEARNING  // constructing decision tree 
};

void main()
{
    PROCESS_MODE mode = LEARNING;
	switch (mode){
    case LABELING:
		ProcessLabeling();
		break;
	case LEARNING:
        ProcessLearning();
		break;
	}
}