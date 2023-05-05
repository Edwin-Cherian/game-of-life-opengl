#include "Solver.h"
#include <iostream>
#include <chrono>
#include <vector>
Solver::Solver(float width, float height, int rows, int cols, int stride)
	: WINDOW_WIDTH(width), WINDOW_HEIGHT(height), ROWS(rows), COLS(cols), STRIDE(stride) {
	std::cout << WINDOW_HEIGHT << " ; " << WINDOW_WIDTH << std::endl;
};

void Solver::setCellState(float* positions, int row, int col, float state)
{
	positions[(row * COLS + col) * STRIDE +  2] = state;
	positions[(row * COLS + col) * STRIDE +  5] = state;
	positions[(row * COLS + col) * STRIDE +  8] = state;
	positions[(row * COLS + col) * STRIDE + 11] = state;
}

void Solver::resetCells(float* positions)
{
	for (int i = 0; i < ROWS * COLS; i++)
	{
		positions[i * STRIDE + 2] = 0.0f;
		positions[i * STRIDE + 5] = 0.0f;
		positions[i * STRIDE + 8] = 0.0f;
		positions[i * STRIDE + 11] = 0.0f;
	}
}



void Solver::updateCells(float* positions, float* positions_buffer)
{
	g_updateCells<<<256, 256>>>(positions, positions_buffer, STRIDE, ROWS, COLS);
	cudaDeviceSynchronize();
	g_copyFromBuffer<<<256, 256>>>(positions, positions_buffer, STRIDE, ROWS, COLS);
	cudaDeviceSynchronize();
}

int Solver::getNeighbours(float* positions, int idx)
{
	int count = -1 * positions[idx];
	for (int i = -1; i < 2; i++)
	{
		for (int j = -1; j < 2; j++)
		{
			if (idx + STRIDE * (i * COLS + j) >= 0 && idx + STRIDE * (i * COLS + j) <= ROWS * COLS * STRIDE)
				count += positions[idx + STRIDE * (i * COLS + j)];
		}
	}
	return count;
}

__global__
void g_updateCells(float* positions, float* positions_buffer, int stride, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_stride = blockDim.x * gridDim.x;
	for (int i = idx; i < rows * cols; i += thread_stride) {
		int neighbours = -1 * positions[i * stride + 2];
		for (int j = -1; j < 2; j++)
		{
			for (int k = -1; k < 2; k++)
			{
				if (stride * (i + j * cols + k) + 2 >= 0 && stride * (1 + j * cols + k) + 2 <= rows * cols * 12)
					neighbours += positions[stride * (i + j * cols + k) + 2];
			}
		}
		if ((positions[i * stride + 2] == 1.0f && neighbours == 2) || neighbours == 3) {
			positions_buffer[i * stride +  2] = 1.0f;
			positions_buffer[i * stride +  5] = 1.0f;
			positions_buffer[i * stride +  8] = 1.0f;
			positions_buffer[i * stride + 11] = 1.0f;
		}
		else {
			positions_buffer[i * stride +  2] = 0.0f;
			positions_buffer[i * stride +  5] = 0.0f;
			positions_buffer[i * stride +  8] = 0.0f;
			positions_buffer[i * stride + 11] = 0.0f;
		}
	}
}

__global__
void g_copyFromBuffer(float* positions, float* positions_buffer, int stride, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_stride = blockDim.x * gridDim.x;
	for (int i = idx; i < rows * cols; i += thread_stride) {
		positions[i * stride +  2] = positions_buffer[i * stride +  2];
		positions[i * stride +  5] = positions_buffer[i * stride +  5];
		positions[i * stride +  8] = positions_buffer[i * stride +  8];
		positions[i * stride + 11] = positions_buffer[i * stride + 11];
	}
}



//void Solver::wallCollision(float* quadAttribIdx, float* velocityIdx, float p_size)
//{
//	if (*(quadAttribIdx + 0) < 0)
//	{
//		*(quadAttribIdx + 0) = 0;
//		*(velocityIdx + 0) *= -1;
//	}
//	if (*(quadAttribIdx + 0) > WINDOW_WIDTH - p_size)
//	{
//		*(quadAttribIdx + 0) = WINDOW_WIDTH - p_size;
//		*(velocityIdx + 0) *= -1;
//	}
//
//	if (*(quadAttribIdx + 1) < 0)
//	{
//		*(quadAttribIdx + 1) = 0;
//		*(velocityIdx + 1) *= -1;
//	}
//	if (*(quadAttribIdx + 1) > WINDOW_HEIGHT - p_size)
//	{
//		*(quadAttribIdx + 1) = WINDOW_HEIGHT - p_size;
//		*(velocityIdx + 1) *= -1;
//	}
//}
//
//__global__
//void mytest(std::vector<std::vector<std::vector<float*>>>* grid, float* positions, int p_count, int p_size, int stride)
//{
//	// empty kernel
//}
//
//void Solver::particleCollision2(float* positions, int p_count, int p_size, int stride, Grid* grid)
//{
//	for (int i = 0; i < p_count; i++)
//	{
//		std::vector<float*> nearby = grid->FindNear(&(positions[i * stride]));
//		int count = 0;
//		for (float* other : nearby)
//		{
//			// compute distance between current particle and nearby particles to test for collisions
//			if ((positions[i * stride + 0] - other[0]) * (positions[i * stride + 0] - other[0]) +
//				(positions[i * stride + 1] - other[1]) * (positions[i * stride + 1] - other[1]) <= p_size * p_size)
//			{
//				count++;
//			}
//		}
//		if (count > 1)
//		{
//			positions[i * stride + 4] = 1.0f;
//			positions[i * stride + 9] = 1.0f;
//			positions[i * stride + 14] = 1.0f;
//			positions[i * stride + 19] = 1.0f;
//		}
//		else
//		{
//			positions[i * stride + 4] = 0.0f;
//			positions[i * stride + 9] = 0.0f;
//			positions[i * stride + 14] = 0.0f;
//			positions[i * stride + 19] = 0.0f;
//		}
//	}
//}
//
//
//void Solver::particleCollision(float* quadAttribIdx, int p_count, int p_size, int stride, Grid* grid)
//{
//	//mytest <<<1, 1>>>();
//	std::vector<float*> nearby = grid->FindNear(quadAttribIdx);
//	int count = 0;
//	for (float* other : nearby)
//	{
//		// compute distance between current particle and nearby particles to test for collisions
//		if ((*(quadAttribIdx + 0) - *(other + 0)) * (*(quadAttribIdx + 0) - *(other + 0)) +
//			(*(quadAttribIdx + 1) - *(other + 1)) * (*(quadAttribIdx + 1) - *(other + 1)) <= p_size * p_size)
//		{
//			count++;
//		}
//	}
//	if (count > 1)
//	{
//		*(quadAttribIdx + 4) = 1.0f;
//		*(quadAttribIdx + 9) = 1.0f;
//		*(quadAttribIdx + 14) = 1.0f;
//		*(quadAttribIdx + 19) = 1.0f;
//	}
//	else
//	{
//		*(quadAttribIdx + 4) = 0.0f;
//		*(quadAttribIdx + 9) = 0.0f;
//		*(quadAttribIdx + 14) = 0.0f;
//		*(quadAttribIdx + 19) = 0.0f;
//	}
//}
//
//void Solver::updatePosition(float* quadAttribIdx, float* velocityIdx, float p_size, float dt)
//{
//	*(quadAttribIdx + 0) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0);
//	*(quadAttribIdx + 1) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1);
//	*(quadAttribIdx + 5) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0) + p_size;
//	*(quadAttribIdx + 6) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1);
//	*(quadAttribIdx + 10) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0) + p_size;
//	*(quadAttribIdx + 11) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1) + p_size;
//	*(quadAttribIdx + 15) = *(quadAttribIdx + 0) + 1 * *(velocityIdx + 0);
//	*(quadAttribIdx + 16) = *(quadAttribIdx + 1) + 1 * *(velocityIdx + 1) + p_size;
//}

