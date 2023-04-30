#include "Solver.h"
#include <iostream>
#include <chrono>>
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

void Solver::particleCollision(float* quadAttribIdx, int p_count, int p_size, int stride, Grid* grid)
{
	std::vector<float*> nearby = grid->FindNear(quadAttribIdx);
	int count = 0;
	for (float* other : nearby)
	{
		// compute distance between current particle and nearby particles to test for collisions
		if ((*(quadAttribIdx + 0) - *(other + 0)) * (*(quadAttribIdx + 0) - *(other + 0)) +
			(*(quadAttribIdx + 1) - *(other + 1)) * (*(quadAttribIdx + 1) - *(other + 1)) < p_size * p_size)
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
		*(quadAttribIdx + 4)  = 0.0f;
		*(quadAttribIdx + 9)  = 0.0f;
		*(quadAttribIdx + 14) = 0.0f;
		*(quadAttribIdx + 19) = 0.0f;
	}
}

void Solver::updatePosition(float* quadAttribIdx, float* velocityIdx, float p_size, float dt)
{
	*(quadAttribIdx + 0)  = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0);
	*(quadAttribIdx + 1)  = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1);
	*(quadAttribIdx + 5)  = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0) + p_size;
	*(quadAttribIdx + 6)  = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1);     
	*(quadAttribIdx + 10) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0) + p_size;
	*(quadAttribIdx + 11) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1) + p_size;
	*(quadAttribIdx + 15) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0);
	*(quadAttribIdx + 16) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1) + p_size;
}
