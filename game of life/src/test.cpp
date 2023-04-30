#include "test.h"



// n = size of array (float arrays must of of same size)
void add(int n, float* x, float* y)
{
	std::cout << "Add function called" << std::endl;
	std::cout << "Array size: " << n << std::endl;
	for (int i = 0; i < n; i++)
	{
		y[i] += x[i];
	}
}

int _main()
{
	std::cout << "Main function called" << std::endl;
	// CPU implementation
	std::cout << "CPU addition" << std::endl;
	std::cout << "Initialising arrays" << std::endl;
	float n = 1<<20; // 1m elements
	float* x = new float[n];
	float* y = new float[n];

	std::cout << "Populating arrays" << std::endl;
	for (int i = 0; i < n; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Run kernel on 1M elements on the CPU
	std::cout << "Performing array addition" << std::endl;
	add(n, x, y);

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < n; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;

	// free memory
	delete[] x;
	delete[] y;

	std::cout << "GPU addition" << maxError << std::endl;
	// GPU implementation
	
	return 0;// g_main();
}