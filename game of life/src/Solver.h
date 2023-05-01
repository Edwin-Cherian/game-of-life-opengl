#pragma once
#include "Grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <iostream>
#include <chrono>

static class Solver
{
private:
	float WINDOW_WIDTH;
	float WINDOW_HEIGHT;
	int ROWS;
	int COLS;
	int STRIDE;
public:
	Solver(float width, float height, int rows, int cols, int stride);
	//void wallCollision(float* quadAttribIdx, float* velocityIdx, float size);
	//void updatePosition(float* quadAttribIdx, float* velocityIdx, float size, float dt);
	////__global__
	//void particleCollision(float* quadAttribIdx, int p_count, int p_size, int quad_vacount, Grid* grid);
	//void particleCollision2(float* quadAttribIdx, int p_count, int p_size, int quad_vacount, Grid* grid);
	void setCellState(float* positions, int row, int col, float state);
	void resetCells(float* positions);
	void updateCells(float* positions, float* positions_buffer);
	int getNeighbours(float* positions, int idx);
};

//__global__
//void mytest(std::vector<std::vector<std::vector<float*>>>* grid, float* positions, int p_count, int p_size, int stride);
