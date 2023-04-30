#include "Solver.h"
#include <iostream>
#include <chrono>
#include <vector>
Solver::Solver(float width, float height)
	: WINDOW_WIDTH(width), WINDOW_HEIGHT(height) {
	std::cout << WINDOW_HEIGHT << " ; " << WINDOW_WIDTH << std::endl;
};

void Solver::wallCollision(float* quadAttribIdx, float* velocityIdx, float p_size)
{
	if (*(quadAttribIdx + 0) < 0)
	{
		*(quadAttribIdx + 0) = 0;
		*(velocityIdx + 0) *= -1;
	}
	if (*(quadAttribIdx + 0) > WINDOW_WIDTH - p_size)
	{
		*(quadAttribIdx + 0) = WINDOW_WIDTH - p_size;
		*(velocityIdx + 0) *= -1;
	}

	if (*(quadAttribIdx + 1) < 0)
	{
		*(quadAttribIdx + 1) = 0;
		*(velocityIdx + 1) *= -1;
	}
	if (*(quadAttribIdx + 1) > WINDOW_HEIGHT - p_size)
	{
		*(quadAttribIdx + 1) = WINDOW_HEIGHT - p_size;
		*(velocityIdx + 1) *= -1;
	}
}

__global__
void mytest(std::vector<std::vector<std::vector<float*>>>* grid, float* positions, int p_count, int p_size, int stride)
{
	for (int k = 0; k < p_count; k++)
	{
		std::vector<float*> nearby;
		const int cell_col = static_cast<int>(positions[k * stride + 0] / (1200.0f / 600)); // cols
		const int cell_row = static_cast<int>(positions[k * stride + 0] / (700.0f / 350)); // rows
		for (int i = -1; i < 2; i++)
		{
			for (int j = -1; j < 2; j++)
			{
				if (cell_row + i < 0 || cell_row + i >= 350 || cell_col + j < 0 || cell_col + j >= 600);
					continue;
				for (auto v : (*grid)[cell_row + i][cell_col + j])
				{
					nearby.push_back(v);
				}
			}
		}
		//std::vector<float*> nearby = grid->FindNear(&(positions[i * stride]));
		int count = 0;
		for (float* other : nearby)
		{
			// compute distance between current particle and nearby particles to test for collisions
			if ((positions[k * stride + 0] - other[0]) * (positions[k * stride + 0] - other[0]) +
				(positions[k * stride + 1] - other[1]) * (positions[k * stride + 1] - other[1]) <= p_size * p_size)
			{
				count++;
			}
		}
		if (count > 1)
		{
			positions[k * stride + 4] = 1.0f;
			positions[k * stride + 9] = 1.0f;
			positions[k * stride + 14] = 1.0f;
			positions[k * stride + 19] = 1.0f;
		}
		else
		{
			positions[k * stride + 4] = 0.0f;
			positions[k * stride + 9] = 0.0f;
			positions[k * stride + 14] = 0.0f;
			positions[k * stride + 19] = 0.0f;
		}
	}
}

void Solver::particleCollision2(float* positions, int p_count, int p_size, int stride, Grid* grid)
{
	//pointer to raw grid
	mytest <<<1, 1>>>(grid->getPGrid(), positions, p_count, p_size, stride);
	cudaDeviceSynchronize();
}


void Solver::particleCollision(float* quadAttribIdx, int p_count, int p_size, int stride, Grid* grid)
{
	//mytest <<<1, 1>>>();
	std::vector<float*> nearby = grid->FindNear(quadAttribIdx);
	int count = 0;
	for (float* other : nearby)
	{
		// compute distance between current particle and nearby particles to test for collisions
		if ((*(quadAttribIdx + 0) - *(other + 0)) * (*(quadAttribIdx + 0) - *(other + 0)) +
			(*(quadAttribIdx + 1) - *(other + 1)) * (*(quadAttribIdx + 1) - *(other + 1)) <= p_size * p_size)
		{
			count++;
		}
	}
	if (count > 1)
	{
		*(quadAttribIdx + 4) = 1.0f;
		*(quadAttribIdx + 9) = 1.0f;
		*(quadAttribIdx + 14) = 1.0f;
		*(quadAttribIdx + 19) = 1.0f;
	}
	else
	{
		*(quadAttribIdx + 4) = 0.0f;
		*(quadAttribIdx + 9) = 0.0f;
		*(quadAttribIdx + 14) = 0.0f;
		*(quadAttribIdx + 19) = 0.0f;
	}
}

void Solver::updatePosition(float* quadAttribIdx, float* velocityIdx, float p_size, float dt)
{
	*(quadAttribIdx + 0) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0);
	*(quadAttribIdx + 1) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1);
	*(quadAttribIdx + 5) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0) + p_size;
	*(quadAttribIdx + 6) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1);
	*(quadAttribIdx + 10) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0) + p_size;
	*(quadAttribIdx + 11) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1) + p_size;
	*(quadAttribIdx + 15) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0);
	*(quadAttribIdx + 16) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1) + p_size;
}
