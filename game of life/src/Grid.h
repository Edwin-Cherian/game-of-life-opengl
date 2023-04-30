#pragma once

#include <vector>
#include <cuda_runtime.h>

class Grid
{
private:
	static const int rows = 350;
	static const int cols = 600;
	//std::vector<float*> grid [rows][cols];
	std::vector<std::vector<std::vector<float*>>> grid { rows, std::vector<std::vector<float*>>{cols} };
	std::vector<std::vector<std::vector<float*>>>* pgrid = &grid;


public:
	Grid();
	~Grid();

	void AddObject(float* objPtr);
	void RemoveObject(float* objPtr);
	std::vector<float*> FindNear(float* objPtr);
	void ReadData(float* objPtr);
	inline auto getRawGrid() { return &grid; };
	inline auto getPGrid() { return pgrid; };
	inline int getRows() { return rows; };
	inline int getCols() { return cols; };
};