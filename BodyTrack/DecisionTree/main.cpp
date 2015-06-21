#include "main.h"

using namespace std;

enum PROCESS_MODE{
	LEBELING, // making correct data
	LEARNING  // constructing decision tree 
};

void main()
{
	PROCESS_MODE mode = LEBELING;
	switch (mode){
	case LEBELING:
		ProcessLabeling();
		break;
	case LEARNING:
//		ProcessLearning();
		break;
	}
}